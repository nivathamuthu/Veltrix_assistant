"""Read functions over the existing tables. Every function takes an open
connection first, so tools can run them inside ``Database.transaction()``.

THIS IS NOT AN AUTHORISATION LAYER. These functions return whatever the
database holds for the id they are given. Who may see what is decided in tool
code (Phase 6) from the ``X-Employee-Id`` header and the requester's DB
record, *before* any of these is called for a target employee.

What they do guarantee: a target id that does not exist yields ``None``
(never an exception, never a different shape), so a missing employee cannot be
told apart from one the caller was not allowed to ask about. The team
functions, whose subject is the *requester*, return ``[]`` for a nonexistent
manager, a non-manager and a manager with no reports alike.

All SQL is a constant string with ``?`` placeholders.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import date

from app.db.models import Employee, LeaveBalance, LeaveRequest

_EMPLOYEE_COLUMNS = (
    "employee_id, full_name, email, phone, region, join_date, level, role, manager_id"
)
_REQUEST_COLUMNS = (
    "r.request_id, r.employee_id, r.leave_type, r.start_date, r.end_date, "
    "r.working_days, r.status, r.created_at"
)

_SQL_EMPLOYEE = "SELECT " + _EMPLOYEE_COLUMNS + " FROM employees WHERE employee_id = ?"

_SQL_DIRECT_REPORTS = (
    "SELECT " + _EMPLOYEE_COLUMNS + " FROM employees WHERE manager_id = ? ORDER BY employee_id"
)

_SQL_BALANCES = (
    "SELECT employee_id, leave_type, balance_days FROM leave_balances "
    "WHERE employee_id = ? AND (? IS NULL OR leave_type = ?) ORDER BY leave_type"
)

_SQL_EMPLOYEE_EXISTS = "SELECT 1 FROM employees WHERE employee_id = ?"

_SQL_REQUESTS = (
    "SELECT " + _REQUEST_COLUMNS + " FROM leave_requests r "
    "WHERE r.employee_id = ? AND (? IS NULL OR r.status = ?) "
    "ORDER BY r.start_date, r.end_date, r.request_id"
)

# A request overlaps [start, end] when it starts on/before `end` and ends
# on/after `start` (inclusive on both sides). Dates are YYYY-MM-DD text, which
# sorts chronologically.
_SQL_TEAM_REQUESTS = (
    "SELECT " + _REQUEST_COLUMNS + " FROM leave_requests r "
    "JOIN employees e ON e.employee_id = r.employee_id "
    "WHERE e.manager_id = ? AND r.status IN ('pending', 'approved') "
    "AND r.start_date <= ? AND r.end_date >= ? "
    "ORDER BY r.start_date, r.employee_id, r.request_id"
)

_SQL_PII = "SELECT full_name, email, phone FROM employees"

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _check_date(name: str, value: str) -> str:
    """Strict YYYY-MM-DD (``date.fromisoformat`` alone accepts other forms)."""
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise ValueError(name + " must be YYYY-MM-DD")
    date.fromisoformat(value)  # rejects 2026-13-45
    return value


def _employee(row: sqlite3.Row) -> Employee:
    return Employee(**{key: row[key] for key in row.keys()})


def _request(row: sqlite3.Row) -> LeaveRequest:
    return LeaveRequest(**{key: row[key] for key in row.keys()})


def get_employee(conn: sqlite3.Connection, employee_id: str) -> Employee | None:
    """The employee with exactly this id (ids are case-sensitive), else None."""
    row = conn.execute(_SQL_EMPLOYEE, (employee_id,)).fetchone()
    return _employee(row) if row else None


def get_direct_reports(conn: sqlite3.Connection, manager_id: str) -> list[Employee]:
    """Employees whose ``manager_id`` equals ``manager_id``; [] if none."""
    return [_employee(r) for r in conn.execute(_SQL_DIRECT_REPORTS, (manager_id,))]


def get_balance(
    conn: sqlite3.Connection, employee_id: str, leave_type: str | None = None
) -> list[LeaveBalance] | None:
    """Current balances (already net of pending/approved days), by leave type.

    None when there is nothing to return: the employee does not exist, or has
    no balance row (for ``leave_type`` if one is given). The two cases are
    deliberately indistinguishable.
    """
    rows = conn.execute(_SQL_BALANCES, (employee_id, leave_type, leave_type)).fetchall()
    if not rows:
        return None
    return [
        LeaveBalance(r["employee_id"], r["leave_type"], r["balance_days"]) for r in rows
    ]


def list_requests(
    conn: sqlite3.Connection, employee_id: str, status: str | None = None
) -> list[LeaveRequest] | None:
    """The employee's requests (optionally one status), oldest start first.

    None if the employee does not exist; [] if they exist but nothing matches.
    Callers must have passed their permission check before asking about a
    target employee, so this difference is never visible to a requester.
    """
    if conn.execute(_SQL_EMPLOYEE_EXISTS, (employee_id,)).fetchone() is None:
        return None
    return [_request(r) for r in conn.execute(_SQL_REQUESTS, (employee_id, status, status))]


def get_team_requests(
    conn: sqlite3.Connection, manager_id: str, start_date: str, end_date: str
) -> list[LeaveRequest]:
    """Pending and approved requests of ``manager_id``'s direct reports that
    overlap ``[start_date, end_date]`` (both inclusive).

    Only direct reports are ever included: the manager's own requests, other
    teams' requests and cancelled requests are not.
    """
    start = _check_date("start_date", start_date)
    end = _check_date("end_date", end_date)
    if end < start:
        raise ValueError("end_date must not be before start_date")
    return [_request(r) for r in conn.execute(_SQL_TEAM_REQUESTS, (manager_id, end, start))]


def get_pii_values(
    conn: sqlite3.Connection,
) -> tuple[list[str], list[str], list[str]]:
    """(names, emails, phones) of all employees, for the log redactor only.

    Never pass the result to the LLM, an answer or a log line.
    """
    rows = conn.execute(_SQL_PII).fetchall()
    return (
        [r["full_name"] for r in rows],
        [r["email"] for r in rows],
        [r["phone"] for r in rows],
    )
