"""POST /chat (README B7).

Header ``X-Employee-Id``, body ``{"session_id": str, "message": str}``.

Order of checks (the first that fails decides the status):
    400  the body is not a JSON object with non-empty string ``session_id`` and ``message``
    401  the header is missing or empty
    503  startup ingestion has not finished
    401  the header is not an existing employee_id
    403  the session was claimed first by a different employee
Everything after that is HTTP 200, including refusals, clarifications, tool errors and
failures: a timeout (one deadline for the whole request, ``REQUEST_TIMEOUT_SECONDS``, under
30 s) and any unexpected exception become a 200 with route "no_answer", a plain apology and
empty citations and tool calls. Error bodies are fixed text: never an echo of the input, a
stack trace or an internal detail.

The role, region and identity of the requester come from the employee record in the
database, loaded here from the header; nothing in the message can change them.

Limitation: a worker thread cannot be cancelled. If the deadline passes while a tool is
mid-write, the user gets the apology but that write may still complete (the tools write in a
single transaction, so it is all or nothing).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import Settings
from app.db import SESSION_FORBIDDEN, Database, Employee, claim_or_check_session, get_employee
from app.ingestion.state import IngestionState
from app.llm import deadline as llm_deadline
from app.orchestrator.response import (
    NO_ANSWER_MESSAGE,
    ROUTE_NO_ANSWER,
    TIMEOUT_MESSAGE,
    ChatResult,
)
from app.orchestrator.service import ChatService

logger = logging.getLogger("app.api.chat")

MAX_MESSAGE_CHARS: Final = 4000
MAX_SESSION_ID_CHARS: Final = 128
MAX_BODY_BYTES: Final = 64 * 1024

DETAIL_BAD_REQUEST: Final = (
    "The request body must be a JSON object with non-empty text fields 'session_id' and 'message'."
)
DETAIL_UNAUTHORIZED: Final = "A valid X-Employee-Id header is required."
DETAIL_FORBIDDEN: Final = "This session belongs to a different employee."
DETAIL_UNAVAILABLE: Final = "The service is still starting. Please try again shortly."


class CitationModel(BaseModel):
    doc_id: str
    locator: str


class ToolCallModel(BaseModel):
    name: str
    args: dict[str, str]
    status: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: str
    citations: list[CitationModel]
    tool_calls: list[ToolCallModel]
    pending_confirmation: bool


_REQUEST_BODY_DOC: Final = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["session_id", "message"],
                    "properties": {"session_id": {"type": "string"}, "message": {"type": "string"}},
                }
            }
        },
    }
}


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def _text_field(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text and len(text) <= limit else None


async def _read_body(request: Request) -> tuple[str, str] | None:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    session_id = _text_field(data.get("session_id"), MAX_SESSION_ID_CHARS)
    message = _text_field(data.get("message"), MAX_MESSAGE_CHARS)
    if session_id is None or message is None:
        return None
    return session_id, message


@dataclass(frozen=True)
class _Auth:
    status: str  # "ok" | "unauthorized" | "forbidden"
    employee: Employee | None = None


def _authenticate(db: Database, employee_id: str, session_id: str) -> _Auth:
    """Load the employee from the DB, then claim or check the session. Blocking (SQLite)."""
    with db.read() as conn:
        employee = get_employee(conn, employee_id)
    if employee is None:
        return _Auth("unauthorized")
    with db.transaction() as conn:
        if claim_or_check_session(conn, session_id, employee.employee_id) == SESSION_FORBIDDEN:
            return _Auth("forbidden")
    return _Auth("ok", employee)


def _handle(
    service: ChatService, employee: Employee, session_id: str, message: str, request_id: str, seconds: float
) -> ChatResult:
    """Runs in the worker thread: gives the provider adapters the time left for this request,
    so no LLM call (or its retry) can outlive the request's deadline."""
    llm_deadline.set_deadline(seconds)
    try:
        return service.handle(employee, session_id, message, request_id)
    finally:
        llm_deadline.clear_deadline()


def _payload(request_id: str, result: ChatResult) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "answer": result.answer,
        "route": result.route,
        "citations": [{"doc_id": c.doc_id, "locator": c.locator} for c in result.citations],
        "tool_calls": [{"name": c.name, "args": dict(c.args), "status": c.status} for c in result.tool_calls],
        "pending_confirmation": result.pending_confirmation,
    }


def _log(request_id: str, employee_id: str, result: ChatResult) -> None:
    # ids, route, tool names and statuses only: never a name, email, phone or message text
    tools = ",".join(f"{c.name}:{c.status}" for c in result.tool_calls) or "-"
    logger.info(
        "chat_request request_id=%s employee=%s route=%s tools=%s", request_id, employee_id, result.route, tools
    )


def register_chat(app: FastAPI, *, settings: Settings, state: IngestionState, db: Database, service: ChatService) -> None:
    @app.get("/me")
    async def me(request: Request) -> Any:
        """Directory fields that are not personal data: id, role, region, level, join date."""
        employee_id = (request.headers.get("X-Employee-Id") or "").strip()
        if not employee_id:
            return _error(401, DETAIL_UNAUTHORIZED)
        with db.read() as conn:
            employee = get_employee(conn, employee_id)
        if employee is None:
            return _error(401, DETAIL_UNAUTHORIZED)
        return {
            "employee_id": employee.employee_id,
            "role": employee.role,
            "region": employee.region,
            "level": employee.level,
            "join_date": employee.join_date,
        }

    @app.post("/chat", response_model=ChatResponse, openapi_extra=_REQUEST_BODY_DOC)
    async def chat(request: Request) -> Any:
        started = time.monotonic()
        request_id = str(uuid.uuid4())

        body = await _read_body(request)
        if body is None:
            logger.info("chat_rejected request_id=%s status=400", request_id)
            return _error(400, DETAIL_BAD_REQUEST)
        session_id, message = body

        employee_id = (request.headers.get("X-Employee-Id") or "").strip()
        if not employee_id:
            logger.info("chat_rejected request_id=%s status=401", request_id)
            return _error(401, DETAIL_UNAUTHORIZED)
        if not state.ready:
            return _error(503, DETAIL_UNAVAILABLE)

        def remaining() -> float:
            return max(0.0, settings.request_timeout_seconds - (time.monotonic() - started))

        who = "-"
        try:
            auth = await asyncio.wait_for(asyncio.to_thread(_authenticate, db, employee_id, session_id), remaining())
            if auth.status == "forbidden":
                logger.info("chat_rejected request_id=%s status=403", request_id)
                return _error(403, DETAIL_FORBIDDEN)
            employee = auth.employee
            if auth.status != "ok" or employee is None:
                logger.info("chat_rejected request_id=%s status=401", request_id)
                return _error(401, DETAIL_UNAUTHORIZED)
            who = employee.employee_id
            result = await asyncio.wait_for(
                asyncio.to_thread(_handle, service, employee, session_id, message, request_id, remaining()),
                remaining(),
            )
        except asyncio.TimeoutError:
            logger.warning("chat_timeout request_id=%s employee=%s", request_id, who)
            result = ChatResult(answer=TIMEOUT_MESSAGE, route=ROUTE_NO_ANSWER)
        except Exception as exc:  # never a stack trace or internal detail for the user
            logger.error("chat_failed request_id=%s employee=%s error=%s", request_id, who, type(exc).__name__)
            result = ChatResult(answer=NO_ANSWER_MESSAGE, route=ROUTE_NO_ANSWER)

        _log(request_id, who, result)
        return _payload(request_id, result)
