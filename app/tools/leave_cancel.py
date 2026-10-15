"""cancel_leave(request_id) for the requester's own requests (README B6).

Rules: the request belongs to the requester, its status is pending or approved, and its
start_date is after today. In ONE transaction (BEGIN IMMEDIATE) the status becomes
"cancelled" and the request's working days go back onto the balance.

A request id that does not exist and one that belongs to someone else get the SAME error:
ownership is part of the lookup itself (``get_own_request``), so both are just "no row".
``validate_cancel_leave`` runs the same checks without writing, for the propose step.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from app.db import Database, Employee, get_balance, get_own_request
from app.db.models import LeaveRequest
from app.tools.common import (
    ToolResult,
    error,
    finish,
    is_valid_id,
    parse_iso_date,
    require_requester,
    require_today,
    safe_args,
    success,
)

NAME = "cancel_leave"

STATUS_CANCELLED = "cancelled"
CANCELLABLE = ("pending", "approved")

# The same text for "no such request" and "someone else's request".
REASON_NOT_FOUND = "I couldn't find a leave request with that id among your requests."
REASON_ALREADY_CANCELLED = "That leave request is already cancelled."
REASON_BAD_STATUS = "Only pending or approved requests can be cancelled; this one is {status}."
REASON_STARTED = (
    "Only leave that starts after today can be cancelled: that request has already started "
    "or starts today."
)
REASON_BAD_DATES = "The dates of that request are not valid, so it can't be cancelled."
REASON_NO_BALANCE_ROW = (
    "There is no {leave_type} leave balance to add the days back to, so the request can't be cancelled."
)
REASON_NOT_CHANGED = "The request could not be completed, so nothing was changed. Please try again."


@dataclass(frozen=True)
class CancelValidation:
    """Outcome of ``validate_cancel_leave``: a summary of the request to cancel, or the
    rejection reason. On a rejection every other field is None (it says nothing about
    whether the id exists)."""

    ok: bool
    request_id: str | None = None
    leave_type: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    working_days: int | None = None
    reason: str | None = None


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _evaluate(
    conn: sqlite3.Connection, requester: Employee, request_id: object, today: date
) -> LeaveRequest:
    """The request to cancel, or ``_Rejected``. Used for the first look, the final look
    inside the transaction, and validation."""
    if not is_valid_id(request_id):
        raise _Rejected(REASON_NOT_FOUND)
    request = get_own_request(conn, requester.employee_id, request_id)
    if request is None:
        raise _Rejected(REASON_NOT_FOUND)
    if request.status == STATUS_CANCELLED:
        raise _Rejected(REASON_ALREADY_CANCELLED)
    if request.status not in CANCELLABLE:
        raise _Rejected(REASON_BAD_STATUS.format(status=request.status))
    start = parse_iso_date(request.start_date)
    if start is None:
        raise _Rejected(REASON_BAD_DATES)
    if start <= today:
        raise _Rejected(REASON_STARTED)
    if not get_balance(conn, requester.employee_id, request.leave_type):
        raise _Rejected(REASON_NO_BALANCE_ROW.format(leave_type=request.leave_type))
    return request


def validate_cancel_leave(
    requester: Employee, request_id: str, *, db: Database, today: date
) -> CancelValidation:
    """Run every cancel_leave check without writing anything (read-only connection)."""
    requester = require_requester(requester)
    today = require_today(today)
    try:
        with db.read() as conn:
            request = _evaluate(conn, requester, request_id, today)
    except _Rejected as rejected:
        return CancelValidation(ok=False, reason=rejected.reason)
    return CancelValidation(
        ok=True,
        request_id=request.request_id,
        leave_type=request.leave_type,
        start_date=request.start_date,
        end_date=request.end_date,
        working_days=request.working_days,
    )


def _mark_cancelled(conn: sqlite3.Connection, employee_id: str, request_id: str) -> None:
    changed = conn.execute(
        "UPDATE leave_requests SET status = ? "
        "WHERE request_id = ? AND employee_id = ? AND status IN ('pending', 'approved')",
        (STATUS_CANCELLED, request_id, employee_id),
    ).rowcount
    if changed != 1:  # the re-check passed, so this is a safety net, not a normal path
        raise _Rejected(REASON_NOT_FOUND)


def _credit_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str, days: int) -> None:
    changed = conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days + ? "
        "WHERE employee_id = ? AND leave_type = ?",
        (days, employee_id, leave_type),
    ).rowcount
    if changed != 1:
        raise _Rejected(REASON_NO_BALANCE_ROW.format(leave_type=leave_type))


def cancel_leave(
    requester: Employee, request_id: str, *, db: Database, today: date
) -> ToolResult:
    """Cancel one of the requester's own pending or approved requests that starts after
    ``today`` (APP_TODAY), and give its working days back, atomically."""
    requester = require_requester(requester)
    today = require_today(today)
    args = safe_args(request_id=request_id)

    try:
        with db.read() as conn:  # first look: keeps rejected requests from taking the write lock
            _evaluate(conn, requester, request_id, today)

        try:
            with db.transaction() as conn:  # BEGIN IMMEDIATE ... COMMIT, ROLLBACK on any exception
                request = _evaluate(conn, requester, request_id, today)  # final look
                _mark_cancelled(conn, requester.employee_id, request.request_id)
                _credit_balance(conn, requester.employee_id, request.leave_type, request.working_days)
                balances = get_balance(conn, requester.employee_id, request.leave_type)
                new_balance = balances[0].balance_days if balances else 0.0
        except sqlite3.Error:
            return finish(requester, error(NAME, args, REASON_NOT_CHANGED))
    except _Rejected as rejected:
        return finish(requester, error(NAME, args, rejected.reason))

    return finish(
        requester,
        success(
            NAME,
            args,
            {
                "request_id": request.request_id,
                "status": STATUS_CANCELLED,
                "leave_type": request.leave_type,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "working_days": request.working_days,
                "balance_days": new_balance,
            },
        ),
    )
