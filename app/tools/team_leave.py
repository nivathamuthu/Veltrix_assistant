"""get_team_leave(start_date, end_date): a manager's direct reports on leave (README B3, B6).

Managers only (hr is not a manager here). The tool accepts no employee list: the team is
read from the DB as "employees whose manager_id is the requester". Cancelled requests are
excluded and the range is inclusive on both ends.
"""
from __future__ import annotations

from app.db import Database, Employee, get_direct_reports, get_team_requests
from app.tools.access import is_manager
from app.tools.common import (
    ToolResult,
    error,
    finish,
    parse_iso_date,
    require_requester,
    safe_args,
    success,
)

NAME = "get_team_leave"

REASON_NOT_A_MANAGER = "Team leave is available to managers only."
REASON_BAD_DATE = "{name} must be a valid calendar date in YYYY-MM-DD format."
REASON_BAD_RANGE = "end_date must not be before start_date."


def get_team_leave(
    requester: Employee,
    start_date: str,
    end_date: str,
    *,
    db: Database,
) -> ToolResult:
    """Pending and approved leave of the requester's DIRECT REPORTS overlapping
    ``[start_date, end_date]``, each with the report's employee_id and full_name (the only
    place a name is returned: the requester is allowed to see their own team)."""
    requester = require_requester(requester)
    args = safe_args(start_date=start_date, end_date=end_date)

    if not is_manager(requester):
        return finish(requester, error(NAME, args, REASON_NOT_A_MANAGER))
    start = parse_iso_date(start_date)
    if start is None:
        return finish(requester, error(NAME, args, REASON_BAD_DATE.format(name="start_date")))
    end = parse_iso_date(end_date)
    if end is None:
        return finish(requester, error(NAME, args, REASON_BAD_DATE.format(name="end_date")))
    if end < start:
        return finish(requester, error(NAME, args, REASON_BAD_RANGE))

    with db.read() as conn:
        names = {e.employee_id: e.full_name for e in get_direct_reports(conn, requester.employee_id)}
        requests = get_team_requests(conn, requester.employee_id, start.isoformat(), end.isoformat())

    data = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "requests": [
            {
                "request_id": r.request_id,
                "employee_id": r.employee_id,
                "full_name": names[r.employee_id],
                "leave_type": r.leave_type,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "working_days": r.working_days,
                "status": r.status,
            }
            for r in requests
            if r.employee_id in names  # a direct report, always; the guard keeps it explicit
        ],
    }
    return finish(requester, success(NAME, args, data))
