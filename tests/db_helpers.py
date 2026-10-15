"""Synthetic database with the exact schema from the brief.

Hand-built rows chosen to hit edge cases. Everything is invented; nothing
here comes from the corpus or the sample database.

Org chart
    H1 (hr, IN)
    M1 (manager, IN) -> reports E1, E2, M3
    M2 (manager, US) -> report  E3
    M3 (manager, IN) -> no reports
    E4 (employee, DE), no manager, no balance rows, no requests
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

CORE_SCHEMA = (
    """CREATE TABLE employees (
      employee_id TEXT PRIMARY KEY,
      full_name   TEXT NOT NULL,
      email       TEXT NOT NULL,
      phone       TEXT NOT NULL,
      region      TEXT NOT NULL,
      join_date   TEXT NOT NULL,
      level       TEXT NOT NULL,
      role        TEXT NOT NULL,
      manager_id  TEXT REFERENCES employees(employee_id)
    )""",
    """CREATE TABLE leave_balances (
      employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
      leave_type   TEXT NOT NULL,
      balance_days REAL NOT NULL,
      PRIMARY KEY (employee_id, leave_type)
    )""",
    """CREATE TABLE leave_requests (
      request_id   TEXT PRIMARY KEY,
      employee_id  TEXT NOT NULL REFERENCES employees(employee_id),
      leave_type   TEXT NOT NULL,
      start_date   TEXT NOT NULL,
      end_date     TEXT NOT NULL,
      working_days INTEGER NOT NULL,
      status       TEXT NOT NULL,
      created_at   TEXT NOT NULL
    )""",
)

# (employee_id, full_name, email, phone, region, join_date, level, role, manager_id)
EMPLOYEES = [
    ("H1", "Hana Quill", "hana.quill@veltrix.example", "+91 90000 00001", "IN", "2019-01-01", "L6", "hr", None),
    ("M1", "Marcus Vane", "marcus.vane@veltrix.example", "+91 90000 00002", "IN", "2019-02-01", "L5", "manager", "H1"),
    ("M2", "Mira Stroud", "mira.stroud@veltrix.example", "+1 512-555-0102", "US", "2019-03-01", "L5", "manager", "H1"),
    ("M3", "Milo Ashby", "milo.ashby@veltrix.example", "+91 90000 00003", "IN", "2020-01-01", "L4", "manager", "M1"),
    ("E1", "Eli Fenn", "eli.fenn@veltrix.example", "+91 90000 00004", "IN", "2021-01-01", "L2", "employee", "M1"),
    ("E2", "Esme Rook", "esme.rook@veltrix.example", "+91 90000 00005", "IN", "2021-02-01", "L2", "employee", "M1"),
    ("E3", "Evan Pike", "evan.pike@veltrix.example", "+1 512-555-0103", "US", "2021-03-01", "L2", "employee", "M2"),
    ("E4", "Edda Lorn", "edda.lorn@veltrix.example", "+49 30 5550 0104", "DE", "2021-04-01", "L2", "employee", None),
]

BALANCES = [
    ("E1", "annual", 10.0), ("E1", "sick", 5.0),
    ("E2", "annual", 3.5),
    ("E3", "annual", 12.0), ("E3", "sick", 4.0),
    ("M1", "annual", 15.0), ("M1", "sick", 6.0),
]

# (request_id, employee_id, leave_type, start, end, working_days, status, created_at)
REQUESTS = [
    ("R1", "E1", "annual", "2026-11-02", "2026-11-04", 3, "pending",   "2026-10-01T09:00:00"),
    ("R2", "E1", "annual", "2026-12-21", "2026-12-24", 4, "approved",  "2026-10-02T09:00:00"),
    ("R3", "E1", "sick",   "2026-11-10", "2026-11-11", 2, "cancelled", "2026-10-03T09:00:00"),
    ("R4", "E2", "annual", "2026-11-04", "2026-11-06", 3, "approved",  "2026-10-04T09:00:00"),
    ("R5", "E3", "annual", "2026-11-03", "2026-11-03", 1, "pending",   "2026-10-05T09:00:00"),
    ("R6", "M1", "annual", "2026-11-03", "2026-11-03", 1, "approved",  "2026-10-06T09:00:00"),
    ("R7", "M3", "annual", "2026-11-05", "2026-11-05", 1, "pending",   "2026-10-07T09:00:00"),
]


def build_synthetic_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    for ddl in CORE_SCHEMA:
        conn.execute(ddl)
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?)", EMPLOYEES)
    conn.executemany("INSERT INTO leave_balances VALUES (?,?,?)", BALANCES)
    conn.executemany("INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)", REQUESTS)
    conn.commit()
    conn.close()
    return path


def dump_tables(path: Path, tables: tuple[str, ...]) -> dict[str, list[tuple]]:
    """Every row of the given tables, sorted, for before/after comparison."""
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return {
            t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall(), key=repr)  # test-only
            for t in tables
        }
    finally:
        conn.close()


def schema_of(path: Path, tables: tuple[str, ...]) -> dict[str, str]:
    """The stored CREATE statement of each given table."""
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return {
            name: sql
            for name, sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'")
            if name in tables
        }
    finally:
        conn.close()
