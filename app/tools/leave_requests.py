"""list_leave_requests(employee_id?, status?): leave requests of one employee (README B6)."""
from __future__ import annotations

from app.db import Database, Employee, list_requests
from app.tools.access import may_view_employee
from app.tools.common import (
    REQUEST_STATUSES,
    ToolResult,
    error,
    finish,
    is_request_status,
    is_valid_id,
    refusal,
    require_requester,
    safe_args,
    success,
)

NAME = "list_leave_requests"

REASON_BAD_EMPLOYEE_ID = "employee_id must be a single employee id."
REASON_BAD_STATUS = f"status must be one of: {', '.join(repr(s) for s in REQUEST_STATUSES)}."
# Reachable only for hr (who may see every employee), for an id that is not an employee.
REASON_NO_SUCH_EMPLOYEE = "There is no employee with that id."


def list_leave_requests(
    requester: Employee,
    employee_id: str | None = None,
    status: str | None = None,
    *,
    db: Database,
) -> ToolResult:
    """The requests of ONE employee (the requester when ``employee_id`` is None), oldest
    start first, optionally of one status. Same access rules as ``get_leave_balance``."""
    requester = require_requester(requester)
    args = safe_args(employee_id=employee_id, status=status)

    if employee_id is not None and not is_valid_id(employee_id):
        return finish(requester, error(NAME, args, REASON_BAD_EMPLOYEE_ID))
    target = requester.employee_id if employee_id is None else employee_id

    with db.read() as conn:
        if not may_view_employee(conn, requester, target):
            return finish(requester, refusal(NAME, args))
        # only now may the target's data be touched
        if status is not None and not is_request_status(status):
            return finish(requester, error(NAME, args, REASON_BAD_STATUS))
        requests = list_requests(conn, target, status)

    if requests is None:
        return finish(requester, error(NAME, args, REASON_NO_SUCH_EMPLOYEE))
    data = {
        "employee_id": target,
        "requests": [
            {
                "request_id": r.request_id,
                "leave_type": r.leave_type,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "working_days": r.working_days,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in requests
        ],
    }
    return finish(requester, success(NAME, args, data))
