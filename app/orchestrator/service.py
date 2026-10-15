"""The orchestrator: one message in, one ``ChatResult`` out (README B7-B10).

Order of events for a message from an authenticated requester in a session they own:

 1. A clear "yes" with a proposal pending runs that proposal (apply_leave / cancel_leave
    re-validate inside the tool). No analyzer, no LLM.
 2. Any other message first discards the session's pending proposal.
 3. Deterministic guards (bulk data, salaries, contact details, system prompt, injection):
    refused here, before the analyzer.
 4. The analyzer understands the message (intent plus details). It is untrusted input.
 5. Leave-data intents call the tools with the REQUESTER (loaded from the database). The tool
    checks permission itself. apply / cancel only ever PROPOSE; a leave type, dates or request
    id that is missing or unclear gets ONE clarify message listing everything missing.
 6. Policy and holiday questions go to the responder (a placeholder until Phase 7b).
 7. ``enforce_invariants`` runs on whatever came out.

Nothing here reads a clock (``today`` is given), logs a message, or trusts a role, region or
identity claim from the text.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from datetime import date
from typing import Any, Callable, Final

from app.db import Database, Employee, readable_doc_ids
from app.llm.providers.base import LLMUnavailable
from app.llm.schema import InvalidModelOutput
from app.orchestrator.analyzer import (
    INTENT_APPLY,
    INTENT_BALANCE,
    INTENT_CANCEL,
    INTENT_HOLIDAY,
    INTENT_LIST,
    INTENT_OFF_TOPIC,
    INTENT_POLICY,
    INTENT_TEAM,
    MISSING_DATES,
    MISSING_EMPLOYEE,
    MISSING_LEAVE_TYPE,
    MISSING_REQUEST,
    Analyzer,
    SessionContext,
    Understanding,
)
from app.orchestrator.confirmation import (
    discard_pending,
    execute_pending,
    get_pending,
    is_confirmation,
    propose_apply,
    propose_cancel,
)
from app.orchestrator.dates import (
    REASON_AMBIGUOUS,
    REASON_END_BEFORE_START,
    REASON_HALF_DAY,
    REASON_INVALID,
    REASON_OUTSIDE_YEAR,
    REASON_PAST,
    parse_leave_dates,
)
from app.orchestrator.guards import GUARD_REFUSAL_MESSAGE, OFF_TOPIC_MESSAGE, check_guards
from app.orchestrator.policy import PolicyRequest, Responder
from app.orchestrator.response import (
    NO_ANSWER_MESSAGE,
    ROUTE_CLARIFY,
    ROUTE_NO_ANSWER,
    ROUTE_REFUSE,
    UNAVAILABLE_MESSAGE,
    ChatResult,
    enforce_invariants,
)
from app.orchestrator.tool_answers import tool_answer
from app.retrieval.policy_resolver import readable_access_levels
from app.tools import (
    LEAVE_TYPES,
    STATUS_PENDING_CONFIRMATION,
    get_leave_balance,
    get_team_leave,
    list_leave_requests,
)

logger = logging.getLogger("app.orchestrator.service")

_DETAIL_KEYS: Final = ("leave_type", "start_date", "end_date", "request_id")

GENERIC_CLARIFY: Final = (
    "I can help with questions about Veltrix leave and HR policy, your leave balance and leave "
    "requests, and applying for or cancelling your own leave. What would you like to do?"
)

_DATE_HINTS: Final = {
    REASON_AMBIGUOUS: "that date is ambiguous",
    REASON_PAST: "those dates are in the past",
    REASON_OUTSIDE_YEAR: "leave can only be requested within 2026",
    REASON_END_BEFORE_START: "the end date is before the start date",
    REASON_INVALID: "that isn't a valid date",
    REASON_HALF_DAY: "half days aren't supported",
}
_DATES_ASK: Final = "a start date and an end date, like 'from Monday to Wednesday'"


class ChatService:
    def __init__(
        self,
        *,
        db: Database,
        analyzer: Analyzer,
        responder: Responder,
        today: Callable[[], date],
        index: Callable[[], Any] = lambda: None,
    ) -> None:
        self.db = db
        self.analyzer = analyzer
        self.responder = responder
        self._today = today
        self._index = index

    # --- entry point ----------------------------------------------------------------------------------------------

    def handle(
        self, requester: Employee, session_id: str, message: str, request_id: str | None = None
    ) -> ChatResult:
        """``request_id`` only labels the log lines of this request's LLM calls."""
        request_id = request_id or str(uuid.uuid4())
        today = self._today()
        pending = get_pending(session_id, db=self.db)

        if pending is not None and is_confirmation(message):
            executed = execute_pending(requester, db=self.db, today=today, session_id=session_id)
            if executed is not None:
                return self._finish(requester, executed)
            pending = None  # nothing valid to run (and it is gone): an ordinary message

        context = SessionContext(
            pending_action=pending.action if pending else None,
            pending_details={
                k: v for k, v in (pending.payload if pending else {}).items()
                if k in _DETAIL_KEYS and isinstance(v, str)
            },
            role=requester.role,  # from the database record, never from the message
            region=requester.region,
            request_id=request_id,
        )
        if pending is not None:
            discard_pending(session_id, db=self.db)  # any other message drops the proposal first

        if check_guards(message).refuse:
            return self._finish(requester, ChatResult(answer=GUARD_REFUSAL_MESSAGE, route=ROUTE_REFUSE))

        try:
            understanding = self.analyzer.analyze(message, context)
        except (LLMUnavailable, InvalidModelOutput):
            # Provider down, timed out, or a reply that is not valid JSON: apology, no write.
            return self._finish(requester, ChatResult(answer=UNAVAILABLE_MESSAGE, route=ROUTE_NO_ANSWER))
        if not isinstance(understanding, Understanding):
            raise TypeError("the analyzer must return an Understanding")
        return self._finish(
            requester, self._route(requester, session_id, message, understanding, today, request_id)
        )

    def _finish(self, requester: Employee, result: ChatResult) -> ChatResult:
        with self.db.read() as conn:
            readable = readable_doc_ids(conn, readable_access_levels(requester.role))
        return enforce_invariants(result, readable_doc_ids=readable)

    # --- routing --------------------------------------------------------------------------------------------------

    def _route(
        self, requester: Employee, session_id: str, message: str, u: Understanding, today: date,
        request_id: str | None = None,
    ) -> ChatResult:
        intent = u.intent
        if intent == INTENT_OFF_TOPIC:
            return ChatResult(answer=OFF_TOPIC_MESSAGE, route=ROUTE_REFUSE)
        if intent in (INTENT_POLICY, INTENT_HOLIDAY):
            return self._policy(requester, message, u, today, request_id)
        if intent == INTENT_BALANCE:
            if MISSING_EMPLOYEE in u.missing_fields:
                return _clarify("look that up", [_ASK_EMPLOYEE])
            return tool_answer(get_leave_balance(requester, u.target_employee_id, u.leave_type, db=self.db))
        if intent == INTENT_LIST:
            if MISSING_EMPLOYEE in u.missing_fields:
                return _clarify("look that up", [_ASK_EMPLOYEE])
            return tool_answer(list_leave_requests(requester, u.target_employee_id, u.status, db=self.db))
        if intent == INTENT_TEAM:
            return self._team(requester, u, today)
        if intent == INTENT_APPLY:
            return self._apply(requester, session_id, u, today)
        if intent == INTENT_CANCEL:
            if u.request_id is None or MISSING_REQUEST in u.missing_fields:
                return _clarify("cancel a leave request", [_ASK_REQUEST])
            return propose_cancel(requester, u.request_id, db=self.db, today=today, session_id=session_id)
        return ChatResult(answer=GENERIC_CLARIFY, route=ROUTE_CLARIFY)

    def _policy(
        self, requester: Employee, message: str, u: Understanding, today: date, request_id: str | None = None
    ) -> ChatResult:
        try:
            result = self.responder.respond(
                PolicyRequest(message, requester, u, today, self.db, self._index(), request_id)
            )
        except LLMUnavailable:
            return ChatResult(answer=UNAVAILABLE_MESSAGE, route=ROUTE_NO_ANSWER)
        except Exception as exc:  # the responder failing must not break the request
            logger.error("responder_failed employee=%s error=%s", requester.employee_id, type(exc).__name__)
            return ChatResult(answer=NO_ANSWER_MESSAGE, route=ROUTE_NO_ANSWER)
        if not isinstance(result, ChatResult):
            return ChatResult(answer=NO_ANSWER_MESSAGE, route=ROUTE_NO_ANSWER)
        # only the confirmation step may produce a pending_confirmation call
        calls = tuple(
            c for c in _safe_tuple(result.tool_calls) if getattr(c, "status", None) != STATUS_PENDING_CONFIRMATION
        )
        return replace(result, tool_calls=calls, pending_confirmation=False)

    def _team(self, requester: Employee, u: Understanding, today: date) -> ChatResult:
        parsed = parse_leave_dates(u.dates_text or "", today)
        if MISSING_DATES in u.missing_fields or not (parsed.ok or parsed.reason == REASON_PAST):
            return _clarify("show your team's leave", [_ask_dates(parsed.reason if u.dates_text else None)])
        assert parsed.start is not None and parsed.end is not None  # ok or past: both are set
        return tool_answer(
            get_team_leave(requester, parsed.start.isoformat(), parsed.end.isoformat(), db=self.db)
        )

    def _apply(self, requester: Employee, session_id: str, u: Understanding, today: date) -> ChatResult:
        missing: list[str] = []
        if u.leave_type not in LEAVE_TYPES or MISSING_LEAVE_TYPE in u.missing_fields:
            missing.append(_ASK_LEAVE_TYPE)
        parsed = parse_leave_dates(u.dates_text or "", today)
        if not parsed.ok or MISSING_DATES in u.missing_fields:
            missing.append(_ask_dates(parsed.reason if u.dates_text and not parsed.ok else None))
        if MISSING_EMPLOYEE in u.missing_fields:
            missing.append(_ASK_EMPLOYEE)
        if MISSING_REQUEST in u.missing_fields:
            missing.append(_ASK_REQUEST)
        if missing:
            return _clarify("request leave", missing)  # everything missing, in ONE message
        assert parsed.start is not None and parsed.end is not None
        return propose_apply(
            requester, u.leave_type, parsed.start.isoformat(), parsed.end.isoformat(),
            db=self.db, today=today, session_id=session_id,
        )


# --- clarify ------------------------------------------------------------------------------------------------------

_ASK_LEAVE_TYPE: Final = f"the leave type ({' or '.join(LEAVE_TYPES)})"
_ASK_EMPLOYEE: Final = "which employee you mean (their employee id)"
_ASK_REQUEST: Final = "which request you mean (its request id; ask me to list your requests to see them)"


def _ask_dates(reason: str | None) -> str:
    hint = _DATE_HINTS.get(reason or "")
    if hint:
        return f"the dates ({hint}; please give {_DATES_ASK})"
    return f"the dates ({_DATES_ASK})"


def _clarify(action: str, missing: list[str]) -> ChatResult:
    if len(missing) == 1:
        listing = missing[0]
    else:
        listing = ", ".join(missing[:-1]) + f" and {missing[-1]}"
    return ChatResult(answer=f"To {action} I still need {listing}.", route=ROUTE_CLARIFY)


def _safe_tuple(values: object) -> tuple[Any, ...]:
    try:
        return tuple(values)  # type: ignore[call-overload]
    except Exception:
        return ()
