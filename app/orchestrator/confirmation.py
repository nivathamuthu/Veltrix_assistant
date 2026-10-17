"""Propose, then confirm (README B8): no write ever happens without an explicit "yes".

Flow
    propose_apply / propose_cancel   run the tool's validate_* (no writes). Rejected: a tool
        result with status "error" and the reason, and NO pending proposal (an older one for
        the session is cleared too). Valid: ONE pending proposal for the session is stored
        (a new proposal replaces the old one) and the answer asks the user to confirm; the
        tool call has status "pending_confirmation", the only place that status is produced.
    is_confirmation   only a clear yes ("yes", "confirm", "go ahead", ...). Anything longer or
        mixed ("yes but make it Friday") is not one; neither is "no", "ok" or "skip
        confirmation".
    execute_pending   with a proposal pending, take it out of the table in one atomic step
        (so two concurrent "yes" messages run it once), then run apply_leave / cancel_leave,
        which re-validate inside their own transaction. Whatever the outcome, the proposal is
        gone afterwards.
    discard_pending   any other message first discards the pending proposal.

What is executed is the STORED payload (leave type, dates or request id), never the text of
the confirming message, and only for the employee who made the proposal.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Final

from app.db import Database, Employee
from app.db.proposals import (
    ACTION_APPLY,
    ACTION_CANCEL,
    Proposal,
    claim_proposal,
    delete_proposal,
    get_proposal,
    save_proposal,
)
from app.orchestrator.response import ROUTE_TOOL, ChatResult, ToolCall, tool_call_of
from app.tools import apply_leave, cancel_leave, validate_apply_leave, validate_cancel_leave
from app.tools.common import STATUS_ERROR, STATUS_PENDING_CONFIRMATION, safe_args

NAME_APPLY: Final = "apply_leave"
NAME_CANCEL: Final = "cancel_leave"

_CONFIRMATION = re.compile(r"(?:yes(?:,?\s*please)?|confirm(?:ed)?|go\s+ahead|do\s+it|proceed)[.!]*")
_MAX_CONFIRMATION_CHARS: Final = 40

CONFIRM_HINT: Final = "Reply 'yes' to confirm, or send any other message to cancel this."


def is_confirmation(message: object) -> bool:
    """True only for a clear yes, with simple punctuation and any case."""
    if not isinstance(message, str):
        return False
    text = message.strip().casefold()
    if not text or len(text) > _MAX_CONFIRMATION_CHARS:
        return False
    return _CONFIRMATION.fullmatch(text) is not None


def _days(n: int | None) -> str:
    return f"{n} working day{'' if n == 1 else 's'}"


def _article(word: str) -> str:
    """'an' before a vowel sound, 'a' otherwise (leave types are single English words)."""
    return "an" if (word[:1].casefold() in "aeiou") else "a"


def _when(start: str, end: str) -> str:
    return f"on {start}" if start == end else f"from {start} to {end}"


def _rejected(name: str, args: dict[str, str], reason: str) -> ChatResult:
    return ChatResult(
        answer=f"I can't go ahead with that: {reason}",
        route=ROUTE_TOOL,
        tool_calls=(ToolCall(name, args, STATUS_ERROR),),
        pending_confirmation=False,
    )


def _store(db: Database, session_id: str, action: str, payload: dict[str, Any]) -> None:
    with db.transaction() as conn:
        save_proposal(conn, session_id, action, payload)  # replaces any earlier proposal


def get_pending(session_id: str, *, db: Database) -> Proposal | None:
    with db.read() as conn:
        return get_proposal(conn, session_id)


def discard_pending(session_id: str, *, db: Database) -> None:
    with db.transaction() as conn:
        delete_proposal(conn, session_id)


# --- propose ---------------------------------------------------------------------------------------------------


def propose_apply(
    requester: Employee,
    leave_type: str | None,
    start_date: str | None,
    end_date: str | None,
    *,
    db: Database,
    today: date,
    session_id: str,
) -> ChatResult:
    """Validate an apply_leave request and, if it is valid, ask for confirmation. Writes only
    the proposal row; the leave tables are untouched."""
    args = safe_args(leave_type=leave_type, start_date=start_date, end_date=end_date)
    check = validate_apply_leave(requester, leave_type, start_date, end_date, db=db, today=today)  # type: ignore[arg-type]
    if not check.ok:
        discard_pending(session_id, db=db)
        return _rejected(NAME_APPLY, args, check.reason or "the request is not valid.")

    _store(
        db, session_id, ACTION_APPLY,
        {"employee_id": requester.employee_id, "leave_type": leave_type, "start_date": start_date, "end_date": end_date},
    )
    balance = check.balance_days
    after = f" Your {leave_type} balance would go from {balance:g} to {balance - check.working_days:g} days." \
        if balance is not None and check.working_days is not None else ""
    return ChatResult(
        answer=(
            f"Please confirm: submit {_article(leave_type)} {leave_type} leave request {_when(start_date, end_date)} "  # type: ignore[arg-type]
            f"({_days(check.working_days)}).{after} {CONFIRM_HINT}"
        ),
        route=ROUTE_TOOL,
        tool_calls=(ToolCall(NAME_APPLY, args, STATUS_PENDING_CONFIRMATION),),
        pending_confirmation=True,
    )


def propose_cancel(
    requester: Employee,
    request_id: str | None,
    *,
    db: Database,
    today: date,
    session_id: str,
) -> ChatResult:
    """Validate a cancel_leave request and, if it is valid, ask for confirmation."""
    args = safe_args(request_id=request_id)
    check = validate_cancel_leave(requester, request_id, db=db, today=today)  # type: ignore[arg-type]
    if not check.ok:
        discard_pending(session_id, db=db)
        return _rejected(NAME_CANCEL, args, check.reason or "the request is not valid.")

    _store(db, session_id, ACTION_CANCEL, {"employee_id": requester.employee_id, "request_id": check.request_id})
    return ChatResult(
        answer=(
            f"Please confirm: cancel your {check.leave_type} leave request {check.request_id} "
            f"{_when(check.start_date, check.end_date)} ({_days(check.working_days)}); "  # type: ignore[arg-type]
            f"those days would go back onto your balance. {CONFIRM_HINT}"
        ),
        route=ROUTE_TOOL,
        tool_calls=(ToolCall(NAME_CANCEL, args, STATUS_PENDING_CONFIRMATION),),
        pending_confirmation=True,
    )


# --- execute ---------------------------------------------------------------------------------------------------


def _strings(payload: dict[str, Any], *keys: str) -> list[str] | None:
    values = [payload.get(k) for k in keys]
    return values if all(isinstance(v, str) and v for v in values) else None  # type: ignore[return-value]


def execute_pending(requester: Employee, *, db: Database, today: date, session_id: str) -> ChatResult | None:
    """Run the session's pending proposal for ``requester``. Returns None when there is
    nothing (valid) to run: then the caller treats the message as an ordinary one."""
    with db.transaction() as conn:  # BEGIN IMMEDIATE: of two concurrent "yes" only one gets it
        proposal = claim_proposal(conn, session_id)
    if proposal is None:
        return None
    payload = proposal.payload
    if payload.get("employee_id") != requester.employee_id:
        return None  # not this employee's proposal: discarded (it was claimed), never executed

    if proposal.action == ACTION_APPLY:
        fields = _strings(payload, "leave_type", "start_date", "end_date")
        if fields is None:
            return None
        result = apply_leave(requester, *fields, db=db, today=today)
        if result.status == STATUS_ERROR:
            answer = f"I couldn't submit that request: {result.reason} Nothing was changed."
        else:
            d = result.data
            answer = (
                f"Done. Your {d['leave_type']} leave request {_when(d['start_date'], d['end_date'])} "
                f"({_days(d['working_days'])}) has been submitted and is pending approval. "
                f"Your {d['leave_type']} balance is now {d['balance_days']:g} days."
            )
        return ChatResult(answer=answer, route=ROUTE_TOOL, tool_calls=(tool_call_of(result),))

    if proposal.action == ACTION_CANCEL:
        fields = _strings(payload, "request_id")
        if fields is None:
            return None
        result = cancel_leave(requester, fields[0], db=db, today=today)
        if result.status == STATUS_ERROR:
            answer = f"I couldn't cancel that request: {result.reason} Nothing was changed."
        else:
            d = result.data
            answer = (
                f"Done. Your {d['leave_type']} leave request {d['request_id']} {_when(d['start_date'], d['end_date'])} "
                f"has been cancelled and {_days(d['working_days'])} went back onto your balance. "
                f"Your {d['leave_type']} balance is now {d['balance_days']:g} days."
            )
        return ChatResult(answer=answer, route=ROUTE_TOOL, tool_calls=(tool_call_of(result),))

    return None
