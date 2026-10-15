"""Shared synthetic data for the Phase 6 tool tests. Everything here is invented.

Org chart (on top of tests/db_helpers.py)
    H1 (hr, IN)          no manager
    M1 (manager, IN)     reports E1, E2, M3
    M2 (manager, US)     report  E3
    M3 (manager, IN)     report  E5            (so E5 is M1's report's report: not M1's report)
    E4 (employee, DE)    no manager
    N1 (employee, IN)    no manager, NO balance rows, no requests

TODAY is Thursday 2026-10-15: 10-17 is a Saturday, 10-18 a Sunday, 10-19 a Monday.

Holiday calendars (loaded with the real Phase 3c tables, no calendar for DE):
    CAL-IN  regions IN  public: 2026-10-24 (a Saturday), 2026-11-11 (Wed), 2026-12-25 (Fri)
                        optional (must NOT count): 2026-11-12
    CAL-US  regions US  public: 2026-11-26 (Thu)
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from app.db import (
    KIND_OPTIONAL,
    KIND_PUBLIC,
    CalendarRow,
    Database,
    HolidayRow,
    get_employee,
    replace_holiday_data,
    run_migrations,
)
from tests.db_helpers import build_synthetic_db

TODAY = date(2026, 10, 15)
CORE_TABLES = ("employees", "leave_balances", "leave_requests")

EXTRA_EMPLOYEES = [
    ("E5", "Ezra Holt", "ezra.holt@veltrix.example", "+91 90000 00006", "IN", "2022-01-01", "L1", "employee", "M3"),
    ("N1", "Nia Brandt", "nia.brandt@veltrix.example", "+91 90000 00007", "IN", "2022-02-01", "L1", "employee", None),
]
EXTRA_BALANCES = [
    ("H1", "annual", 8.0), ("H1", "sick", 2.0),
    ("M2", "annual", 8.0), ("M2", "sick", 2.0),
    ("M3", "annual", 8.0), ("M3", "sick", 2.0),
    ("E4", "annual", 6.0), ("E4", "sick", 3.0),
    ("E5", "annual", 8.0), ("E5", "sick", 2.0),
]
EXTRA_REQUESTS = [
    ("R8", "E5", "annual", "2026-11-17", "2026-11-18", 2, "approved", "2026-10-08T09:00:00"),
]


def _calendar(doc_id, regions, status="ok", access="all", effective="2026-01-01", public=1):
    return CalendarRow(doc_id, effective, tuple(regions), access, status, public_count=public)


def _holiday(doc_id, region, day, kind=KIND_PUBLIC):
    return HolidayRow(region, day, "Some day", doc_id, kind, "2026-01-01", (region,))


def load_test_holidays(db: Database) -> None:
    """The synthetic holiday calendars (see the module docstring), via the real tables.

    Separate from ``build_tools_db`` so API tests can add them AFTER startup ingestion,
    which rebuilds the holiday tables from the (empty) test corpus."""
    with db.transaction() as conn:
        replace_holiday_data(
            conn,
            [_calendar("CAL-IN", ["IN"], public=3), _calendar("CAL-US", ["US"])],
            [
                _holiday("CAL-IN", "IN", "2026-10-24"),
                _holiday("CAL-IN", "IN", "2026-11-11"),
                _holiday("CAL-IN", "IN", "2026-12-25"),
                _holiday("CAL-IN", "IN", "2026-11-12", KIND_OPTIONAL),
                _holiday("CAL-US", "US", "2026-11-26"),
            ],
        )


def build_tools_db(path: Path, *, holidays: bool = True) -> Database:
    build_synthetic_db(path)
    db = Database(path)
    run_migrations(db)
    with db.transaction() as conn:
        conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)", EXTRA_EMPLOYEES)
        conn.executemany("INSERT INTO leave_balances VALUES (?,?,?)", EXTRA_BALANCES)
        conn.executemany("INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)", EXTRA_REQUESTS)
    if holidays:
        load_test_holidays(db)
    return db


def who(db: Database, employee_id: str):
    """The requester, loaded from the DB like the caller (the API layer) will."""
    with db.read() as conn:
        employee = get_employee(conn, employee_id)
    assert employee is not None, employee_id
    return employee


def run_sql(db: Database, *statements: str) -> None:
    """Direct test-only SQL (setup, triggers, simulated competing writers)."""
    conn = sqlite3.connect(db.path)
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def balance_of(db: Database, employee_id: str, leave_type: str) -> float | None:
    conn = sqlite3.connect(f"file:{db.path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
            (employee_id, leave_type),
        ).fetchone()
        return None if row is None else row[0]
    finally:
        conn.close()


def request_rows(db: Database, employee_id: str | None = None) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db.path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        if employee_id is None:
            return conn.execute("SELECT * FROM leave_requests ORDER BY request_id").fetchall()
        return conn.execute(
            "SELECT * FROM leave_requests WHERE employee_id = ? ORDER BY request_id", (employee_id,)
        ).fetchall()
    finally:
        conn.close()


def pii_values(db: Database) -> tuple[list[str], list[str], list[str]]:
    from app.db import get_pii_values

    with db.read() as conn:
        return get_pii_values(conn)
