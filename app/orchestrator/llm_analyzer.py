"""The LLM analyzer: ONE call per message, NO corpus text (README B7, Phase 7b).

Input to the model: the message (untrusted, delimited), the summary of the session's pending
proposal if there is one, and ONLY the requester's role and region (both from the database).
No names, emails, phones, balances or documents.

Output: JSON validated against ``ANALYSIS_SCHEMA``. Anything that is not valid (not JSON,
wrong shape, unknown intent) raises ``InvalidModelOutput``. The provider being down raises
``LLMUnavailable``. In both cases /chat answers ``no_answer`` and never writes.

The call goes through ``LLMClient`` and so through ``guard.call_with_corpus`` with an EMPTY
corpus: it is logged with ``corpus_tokens=0`` like every other call.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import replace
from typing import Any, Final

from app.llm.client import LLMClient
from app.llm.schema import InvalidModelOutput, parse_and_validate
from app.orchestrator.analyzer import (
    INTENT_BALANCE,
    INTENT_LIST,
    MISSING_EMPLOYEE,
    Analyzer,
    SessionContext,
    Understanding,
)
from app.orchestrator.prompts import ANALYSIS_SCHEMA, ANALYZER_SYSTEM
from app.orchestrator.tool_intent import extract_mentioned_employee_id, match_unambiguous_tool

logger = logging.getLogger("app.orchestrator.llm_analyzer")

_MAX_FIELD_CHARS: Final = 200
_CLOSING_TAG: Final = re.compile(r"<\s*/\s*message\s*>", re.IGNORECASE)
_PENDING_KEYS: Final = ("leave_type", "start_date", "end_date", "request_id")


def _clean_field(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value[:_MAX_FIELD_CHARS] if value else None


def build_user_message(message: str, context: SessionContext) -> str:
    """Role, region, the pending proposal summary (if any), then the untrusted message."""
    if context.pending_action:
        details = ", ".join(
            f"{k}={context.pending_details[k]}" for k in _PENDING_KEYS if k in context.pending_details
        )
        pending = f"{context.pending_action}" + (f" ({details})" if details else "")
    else:
        pending = "none"
    safe = _CLOSING_TAG.sub("[/message]", message)  # the message cannot end its own delimiter
    return (
        f"requester_role: {context.role or 'unknown'}\n"
        f"requester_region: {context.region or 'unknown'}\n"
        f"pending_proposal: {pending}\n\n"
        f"<message>\n{safe}\n</message>"
    )


def understanding_from(data: dict[str, Any]) -> Understanding:
    """The validated JSON as an ``Understanding`` (which normalises it once more)."""
    return Understanding(
        intent=data["intent"],
        leave_type=_clean_field(data["leave_type"]),
        dates_text=_clean_field(data["dates_text"]),
        target_employee_id=_clean_field(data["target_employee_id"]),
        request_id=_clean_field(data["request_id"]),
        status=_clean_field(data["status"]),
        missing_fields=tuple(data["missing_fields"]),
        needs_leave_data=data["needs_leave_data"],
    )


class LLMAnalyzer(Analyzer):
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def analyze(self, message: str, session_context: SessionContext) -> Understanding:
        request_id = session_context.request_id or str(uuid.uuid4())
        shortcut = match_unambiguous_tool(message)
        if shortcut is not None:
            logger.info("analyze_skipped request_id=%s reason=unambiguous_tool intent=%s", request_id, shortcut.intent)
            return shortcut
        call = self._client.complete(
            purpose="analyze",
            request_id=request_id,
            system=ANALYZER_SYSTEM,
            messages=[{"role": "user", "content": build_user_message(message, session_context)}],
            json_schema=ANALYSIS_SCHEMA,
        )
        try:
            data = parse_and_validate(call.response, ANALYSIS_SCHEMA)
            understanding = understanding_from(data)
        except InvalidModelOutput:
            # Invalid JSON is not a classification. The service turns this into HTTP 200
            # route no_answer; we must not invent an intent (that could propose a write).
            logger.warning("analysis_invalid request_id=%s", request_id)
            raise
        return _with_mentioned_employee(understanding, message)


def _with_mentioned_employee(understanding: Understanding, message: str) -> Understanding:
    """If the message already wrote an id, do not ask for one again."""
    if understanding.intent not in (INTENT_BALANCE, INTENT_LIST):
        return understanding
    target = understanding.target_employee_id or extract_mentioned_employee_id(message)
    if not target:
        return understanding
    missing = tuple(f for f in understanding.missing_fields if f != MISSING_EMPLOYEE)
    if target == understanding.target_employee_id and missing == understanding.missing_fields:
        return understanding
    return replace(understanding, target_employee_id=target, missing_fields=missing)
