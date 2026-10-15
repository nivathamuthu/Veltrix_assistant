"""Developer report: run each of the five tools once and print the ToolResult.

    python -m app.tools.tool_report

Works on a temporary COPY of DB_PATH (apply_leave and cancel_leave write), so the real
database is never touched. The people are picked from the DB by role and relationship,
nothing is hard-coded; "today" is APP_TODAY (or the system date) from config. The
holiday tables must already be filled (the service fills them at startup).

get_team_leave shows full names of the manager's own reports: that is the one place a
tool returns a name, because the requester may see their own team. Nothing prints an
email or a phone.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

from app.config import load_settings
from app.db import Database, Employee, get_employee
from app.tools import (
    ToolResult,
    apply_leave,
    cancel_leave,
    get_leave_balance,
    get_team_leave,
    list_leave_requests,
    validate_apply_leave,
)

_SQL_ONE_MANAGER_AND_REPORT = (
    "SELECT m.employee_id AS manager_id, e.employee_id AS report_id FROM employees m "
    "JOIN employees e ON e.manager_id = m.employee_id "
    "JOIN leave_balances b ON b.employee_id = e.employee_id AND b.leave_type = 'annual' "
    "WHERE m.role = 'manager' AND e.role = 'employee' AND b.balance_days >= 2 "
    "ORDER BY m.employee_id, e.employee_id LIMIT 1"
)
_SQL_OUTSIDER = (
    "SELECT employee_id FROM employees WHERE employee_id NOT IN (?, ?) AND manager_id IS NOT ? "
    "AND role = 'employee' ORDER BY employee_id LIMIT 1"
)
_SQL_FIRST_HR = "SELECT employee_id FROM employees WHERE role = 'hr' ORDER BY employee_id LIMIT 1"


def _show(title: str, result: ToolResult) -> None:
    print(f"\n-- {title}")
    print(json.dumps(dataclasses.asdict(result), indent=2, default=str))


def _load(db: Database, employee_id: str) -> Employee:
    with db.read() as conn:
        employee = get_employee(conn, employee_id)
    assert employee is not None
    return employee


def _free_week(db: Database, who: Employee, today: date) -> tuple[str, str] | None:
    """The first Monday-Tuesday after today that apply_leave would accept for ``who``."""
    monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
    for week in range(60):
        start = monday + timedelta(days=7 * week)
        end = start + timedelta(days=1)
        check = validate_apply_leave(who, "annual", start.isoformat(), end.isoformat(), db=db, today=today)
        if check.ok:
            return start.isoformat(), end.isoformat()
    return None


def main() -> int:
    settings = load_settings()
    today = settings.today()
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "veltrix_copy.db"
        shutil.copyfile(settings.db_path, copy)
        db = Database(copy)

        with db.read() as conn:
            pair = conn.execute(_SQL_ONE_MANAGER_AND_REPORT).fetchone()
            if pair is None:
                print("no manager with a report who has annual balance in this database")
                return 1
            manager_id, report_id = pair["manager_id"], pair["report_id"]
            outsider = conn.execute(_SQL_OUTSIDER, (report_id, manager_id, manager_id)).fetchone()
            hr_row = conn.execute(_SQL_FIRST_HR).fetchone()
        manager, report = _load(db, manager_id), _load(db, report_id)

        print(f"today = {today}   (roles picked from the DB by relationship; ids are real, data is a copy)")
        print(f"manager = {manager_id} ({manager.region}), report = {report_id} ({report.region})")

        _show("get_leave_balance (manager, for a direct report)",
              get_leave_balance(manager, report_id, db=db))
        _show("list_leave_requests (employee, own, pending)",
              list_leave_requests(report, None, "pending", db=db))
        _show("get_team_leave (manager, whole of 2026)",
              get_team_leave(manager, f"{today.year}-01-01", f"{today.year}-12-31", db=db))

        week = _free_week(db, report, today)
        if week is None:
            print("\nno free Monday-Tuesday left in 2026 for apply_leave to demonstrate")
        else:
            applied = apply_leave(report, "annual", *week, db=db, today=today)
            _show("apply_leave (employee, own, a free Monday-Tuesday)", applied)
            if applied.status == "success":
                _show("cancel_leave (employee, the request just created)",
                      cancel_leave(report, applied.data["request_id"], db=db, today=today))
        _show("apply_leave rejection (leave type missing)",
              apply_leave(report, "", f"{today.year}-12-01", f"{today.year}-12-02", db=db, today=today))

        if outsider is not None:
            _show("get_leave_balance REFUSED (employee asks about a colleague)",
                  get_leave_balance(report, outsider["employee_id"], db=db))
            _show("get_leave_balance REFUSED (same, id that does not exist)",
                  get_leave_balance(report, "NOSUCH", db=db))
        if hr_row is not None:
            _show("get_leave_balance (hr, any one employee)",
                  get_leave_balance(_load(db, hr_row["employee_id"]), report_id, db=db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
