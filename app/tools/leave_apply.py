"""apply_leave(leave_type, start_date, end_date) for the requester only (README B6).

``validate_apply_leave`` and ``apply_leave`` run the SAME checks (``_evaluate``), so a
proposal that validates cannot be rejected for a different reason at execution, and the
reason text is identical.

``apply_leave`` writes in ONE transaction opened with BEGIN IMMEDIATE. The checks are
repeated inside it: between the first look and the write another request may have spent
the balance or taken the dates, and the write lock is what makes the second look final.
The request is inserted first and the balance debited second; if anything after the
BEGIN fails, everything is rolled back.

Working days = Monday to Friday minus the PUBLIC holidays of the calendar that applies to
the requester's region, from the Phase 3c tables (never from code). No applicable
calendar means a rejection with the lookup's reason, never "count every weekday".
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time

from app.db import Database, Employee, find_overlapping_requests, get_balance
from app.retrieval.holiday_lookup import NoApplicableCalendar, public_holidays_for
from app.retrieval.policy_resolver import readable_access_levels
from app.tools.common import (
    LEAVE_TYPES,
    LEAVE_YEAR,
    ToolResult,
    error,
    finish,
    is_leave_type,
    parse_iso_date,
    require_requester,
    require_today,
    safe_args,
    success,
)
from app.working_days import working_days

NAME = "apply_leave"

STATUS_PENDING = "pending"

REASON_BAD_LEAVE_TYPE = f"leave_type must be one of: {', '.join(repr(t) for t in LEAVE_TYPES)}."
REASON_BAD_DATE = "{name} must be a valid calendar date in YYYY-MM-DD format."
REASON_NOT_IN_YEAR = f"Leave can only be requested for {LEAVE_YEAR}: both dates must be in {LEAVE_YEAR}."
REASON_IN_THE_PAST = "start_date must not be in the past: it must be today or later."
REASON_END_BEFORE_START = "end_date must not be before start_date."
REASON_NO_WORKING_DAYS = (
    "That range has no working days: it covers only weekends and public holidays."
)
REASON_NO_BALANCE_ROW = "You have no {leave_type} leave balance, so this request can't be made."
REASON_OVER_BALANCE = (
    "This request needs {days} working day(s) but your {leave_type} leave balance is {balance:g}."
)
REASON_OVERLAP = "These dates overlap your {status} {leave_type} leave request from {start} to {end}."
REASON_NOT_CHANGED = "The request could not be completed, so nothing was changed. Please try again."


@dataclass(frozen=True)
class LeaveValidation:
    """Outcome of ``validate_apply_leave``: the working days, or the rejection reason."""

    ok: bool
    working_days: int | None = None
    reason: str | None = None
    balance_days: float | None = None  # the balance the days would come off


class _Rejected(Exception):
    """A rule failed. Carries the requester-safe reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _parse(leave_type: object, start_date: object, end_date: object, today: date) -> tuple[date, date]:
    """The checks that need no database."""
    if not is_leave_type(leave_type):
        raise _Rejected(REASON_BAD_LEAVE_TYPE)
    start = parse_iso_date(start_date)
    if start is None:
        raise _Rejected(REASON_BAD_DATE.format(name="start_date"))
    end = parse_iso_date(end_date)
    if end is None:
        raise _Rejected(REASON_BAD_DATE.format(name="end_date"))
    if start.year != LEAVE_YEAR or end.year != LEAVE_YEAR:
        raise _Rejected(REASON_NOT_IN_YEAR)
    if start < today:
        raise _Rejected(REASON_IN_THE_PAST)
    if end < start:
        raise _Rejected(REASON_END_BEFORE_START)
    return start, end


def _evaluate(
    conn: sqlite3.Connection, requester: Employee, leave_type: str, start: date, end: date, today: date
) -> tuple[int, float]:
    """The checks that read the database: calendar, working days, balance, overlap.
    Returns ``(working_days, balance_days)`` or raises ``_Rejected``."""
    lookup = public_holidays_for(
        conn, requester.region, today, allowed_access_levels=readable_access_levels(requester.role)
    )
    if isinstance(lookup, NoApplicableCalendar):
        raise _Rejected(lookup.message)

    days = working_days(start, end, lookup.dates)
    if days == 0:
        raise _Rejected(REASON_NO_WORKING_DAYS)

    balances = get_balance(conn, requester.employee_id, leave_type)
    if not balances:
        raise _Rejected(REASON_NO_BALANCE_ROW.format(leave_type=leave_type))
    balance = balances[0].balance_days
    if days > balance:
        raise _Rejected(REASON_OVER_BALANCE.format(days=days, leave_type=leave_type, balance=balance))

    clashes = find_overlapping_requests(conn, requester.employee_id, start.isoformat(), end.isoformat())
    if clashes:
        first = clashes[0]
        raise _Rejected(
            REASON_OVERLAP.format(
                status=first.status, leave_type=first.leave_type, start=first.start_date, end=first.end_date
            )
        )
    return days, balance


def validate_apply_leave(
    requester: Employee,
    leave_type: str,
    start_date: str,
    end_date: str,
    *,
    db: Database,
    today: date,
) -> LeaveValidation:
    """Run every apply_leave check without writing anything (read-only connection).
    For the propose step of the confirmation flow."""
    requester = require_requester(requester)
    today = require_today(today)
    try:
        start, end = _parse(leave_type, start_date, end_date, today)
        with db.read() as conn:
            days, balance = _evaluate(conn, requester, leave_type, start, end, today)
    except _Rejected as rejected:
        return LeaveValidation(ok=False, reason=rejected.reason)
    return LeaveValidation(ok=True, working_days=days, balance_days=balance)


def _debit_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str, days: int) -> None:
    changed = conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days - ? "
        "WHERE employee_id = ? AND leave_type = ? AND balance_days >= ?",
        (days, employee_id, leave_type, days),
    ).rowcount
    if changed != 1:  # the re-check passed, so this is a safety net, not a normal path
        raise _Rejected(REASON_OVER_BALANCE.format(days=days, leave_type=leave_type, balance=0))


def _created_at(today: date, now: datetime | None) -> str:
    if now is None:
        return datetime.combine(today, time.min).isoformat(timespec="seconds")
    if not isinstance(now, datetime):
        raise TypeError("now must be a datetime.datetime when given")
    return now.isoformat(timespec="seconds")


def apply_leave(
    requester: Employee,
    leave_type: str,
    start_date: str,
    end_date: str,
    *,
    db: Database,
    today: date,
    now: datetime | None = None,
) -> ToolResult:
    """Create a PENDING leave request for the requester and subtract its working days from
    their balance, atomically. Never for anyone else: there is no employee argument.

    ``today`` is APP_TODAY from config. ``now`` (optional) is the caller's timestamp for
    ``created_at``; without it the timestamp is ``today`` at 00:00:00.
    """
    requester = require_requester(requester)
    today = require_today(today)
    created_at = _created_at(today, now)
    args = safe_args(leave_type=leave_type, start_date=start_date, end_date=end_date)

    try:
        start, end = _parse(leave_type, start_date, end_date, today)
        with db.read() as conn:  # first look: keeps rejected requests from taking the write lock
            _evaluate(conn, requester, leave_type, start, end, today)

        try:
            with db.transaction() as conn:  # BEGIN IMMEDIATE ... COMMIT, ROLLBACK on any exception
                days, _ = _evaluate(conn, requester, leave_type, start, end, today)  # final look
                request_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO leave_requests (request_id, employee_id, leave_type, start_date, "
                    "end_date, working_days, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (request_id, requester.employee_id, leave_type, start.isoformat(), end.isoformat(),
                     days, STATUS_PENDING, created_at),
                )
                _debit_balance(conn, requester.employee_id, leave_type, days)
                balances = get_balance(conn, requester.employee_id, leave_type)
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
                "request_id": request_id,
                "leave_type": leave_type,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "working_days": days,
                "status": STATUS_PENDING,
                "balance_days": new_balance,
            },
        ),
    )
