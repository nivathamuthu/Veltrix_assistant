"""The chat result and the route invariants (README B7).

``enforce_invariants`` is applied to EVERY response before it is returned:

    rag        citations >= 1, no tool calls
    tool       tool calls >= 1, no citations
    hybrid     both >= 1
    clarify / no_answer / refuse    neither

When a response breaks its route's rule it is downgraded to the nearest honest route (rag
with no citations is a no_answer; hybrid with only citations is a rag; ...) and the function
never raises: whatever it is given, the caller gets a valid ``ChatResult`` back.

Citations are de-duplicated and limited to documents that were actually sent to the LLM in
this turn AND that the requester may read (a last line of defence: unreadable documents are
filtered out before retrieval, so they should never get this far).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final, Iterable

from app.tools.common import STATUS_ERROR, STATUS_PENDING_CONFIRMATION, STATUS_SUCCESS, ToolResult

logger = logging.getLogger("app.orchestrator.response")

ROUTE_RAG: Final = "rag"
ROUTE_TOOL: Final = "tool"
ROUTE_HYBRID: Final = "hybrid"
ROUTE_CLARIFY: Final = "clarify"
ROUTE_NO_ANSWER: Final = "no_answer"
ROUTE_REFUSE: Final = "refuse"
ROUTES: Final = (ROUTE_RAG, ROUTE_TOOL, ROUTE_HYBRID, ROUTE_CLARIFY, ROUTE_NO_ANSWER, ROUTE_REFUSE)

NO_ANSWER_MESSAGE: Final = (
    "I'm sorry, I couldn't find an answer to that. Please try rephrasing your question, "
    "or contact HR for help."
)
UNAVAILABLE_MESSAGE: Final = (
    "I'm sorry, I can't answer that right now. Please try again in a moment, or contact HR for help."
)
TIMEOUT_MESSAGE: Final = "I'm sorry, that took too long to answer. Please try again in a moment."

_TOOL_STATUSES: Final = (STATUS_SUCCESS, STATUS_ERROR, STATUS_PENDING_CONFIRMATION)


@dataclass(frozen=True)
class Citation:
    doc_id: str
    locator: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, str]
    status: str


@dataclass(frozen=True)
class ChatResult:
    """What the orchestrator produced for one message (before the API adds the request id)."""

    answer: str
    route: str
    citations: tuple[Citation, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    pending_confirmation: bool = False
    # doc_ids whose text was sent to the LLM this turn: the only ones that may be cited
    sent_doc_ids: frozenset[str] = field(default_factory=frozenset)


def tool_call_of(result: ToolResult, status: str | None = None) -> ToolCall:
    """The response's view of a tool result: name, B6 args and status (never the data)."""
    return ToolCall(result.name, dict(result.args), status or result.status)


def _items(values: object) -> list[Any]:
    try:
        return list(values)  # type: ignore[call-overload]
    except Exception:  # not iterable, or an iterator that fails: treat as empty
        return []


def _clean_citations(
    values: object, sent: frozenset[str], readable: frozenset[str]
) -> tuple[Citation, ...]:
    kept: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for item in _items(values):
        if not isinstance(item, Citation):
            continue
        doc_id, locator = item.doc_id, item.locator
        if not (isinstance(doc_id, str) and doc_id.strip() and isinstance(locator, str) and locator.strip()):
            continue
        if doc_id not in sent or doc_id not in readable:
            continue
        key = (doc_id, locator)
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return tuple(kept)


def _clean_tool_calls(values: object) -> tuple[ToolCall, ...]:
    kept: list[ToolCall] = []
    for item in _items(values):
        if not isinstance(item, ToolCall):
            continue
        if not (isinstance(item.name, str) and item.name and item.status in _TOOL_STATUSES):
            continue
        args = dict(item.args) if isinstance(item.args, dict) else {}
        kept.append(ToolCall(item.name, args, item.status))
    return tuple(kept)


def _string_set(values: object) -> frozenset[str]:
    return frozenset(v for v in _items(values) if isinstance(v, str))


def _fallback() -> ChatResult:
    return ChatResult(answer=NO_ANSWER_MESSAGE, route=ROUTE_NO_ANSWER)


def enforce_invariants(result: ChatResult, *, readable_doc_ids: Iterable[str] | None) -> ChatResult:
    """Return a ChatResult that satisfies the route table, downgrading when it does not."""
    try:
        return _enforce(result, readable_doc_ids)
    except Exception:  # never raise to the user
        logger.error("invariant enforcement failed; answering with a safe no_answer")
        return _fallback()


def _enforce(result: ChatResult, readable_doc_ids: Iterable[str] | None) -> ChatResult:
    if not isinstance(result, ChatResult):
        return _fallback()

    sent = _string_set(result.sent_doc_ids)
    readable = _string_set(readable_doc_ids)
    citations = _clean_citations(result.citations, sent, readable)
    tool_calls = _clean_tool_calls(result.tool_calls)
    answer = result.answer if isinstance(result.answer, str) and result.answer.strip() else NO_ANSWER_MESSAGE
    route = result.route if result.route in ROUTES else ROUTE_NO_ANSWER
    if route != result.route:
        answer = NO_ANSWER_MESSAGE

    cited, called = bool(citations), bool(tool_calls)
    if route == ROUTE_RAG:
        if not cited:
            route = ROUTE_NO_ANSWER
        elif called:
            route = ROUTE_HYBRID
    elif route == ROUTE_TOOL:
        if not called:
            route = ROUTE_NO_ANSWER
        elif cited:
            route = ROUTE_HYBRID
    elif route == ROUTE_HYBRID:
        if cited and called:
            pass
        elif cited:
            route = ROUTE_RAG
        elif called:
            route = ROUTE_TOOL
        else:
            route = ROUTE_NO_ANSWER

    if route == ROUTE_NO_ANSWER and (result.route in (ROUTE_RAG, ROUTE_TOOL, ROUTE_HYBRID)):
        answer = NO_ANSWER_MESSAGE  # an unsupported claim is not passed on
    if route not in (ROUTE_RAG, ROUTE_HYBRID):
        citations = ()
    if route not in (ROUTE_TOOL, ROUTE_HYBRID):
        tool_calls = ()

    pending = any(call.status == STATUS_PENDING_CONFIRMATION for call in tool_calls)
    return ChatResult(
        answer=answer,
        route=route,
        citations=citations,
        tool_calls=tool_calls,
        pending_confirmation=pending,
        sent_doc_ids=sent,
    )
