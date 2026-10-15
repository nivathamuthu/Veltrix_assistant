"""get_leave_balance(employee_id?, leave_type?): the current balance (README B6)."""
from __future__ import annotations

from app.db import Database, Employee, get_balance
from app.tools.access import may_view_employee
from app.tools.common import (
    LEAVE_TYPES,
    ToolResult,
    error,
    finish,
    is_leave_type,
    is_valid_id,
    refusal,
    require_requester,
    safe_args,
    success,
)

NAME = "get_leave_balance"

REASON_BAD_EMPLOYEE_ID = "employee_id must be a single employee id."
REASON_BAD_LEAVE_TYPE = f"leave_type must be one of: {', '.join(repr(t) for t in LEAVE_TYPES)}."
REASON_NO_BALANCE = "No leave balance is recorded for that employee."


def get_leave_balance(
    requester: Employee,
    employee_id: str | None = None,
    leave_type: str | None = None,
    *,
    db: Database,
) -> ToolResult:
    """Balance days per leave type for ONE employee: the requester when ``employee_id`` is
    None, otherwise someone the requester is allowed to see (self; a manager's direct
    report; any one employee for hr). The balance already has pending and approved days
    subtracted."""
    requester = require_requester(requester)
    args = safe_args(employee_id=employee_id, leave_type=leave_type)

    if employee_id is not None and not is_valid_id(employee_id):
        return finish(requester, error(NAME, args, REASON_BAD_EMPLOYEE_ID))
    target = requester.employee_id if employee_id is None else employee_id

    with db.read() as conn:
        if not may_view_employee(conn, requester, target):
            return finish(requester, refusal(NAME, args))
        # only now may the target's data be touched
        if leave_type is not None and not is_leave_type(leave_type):
            return finish(requester, error(NAME, args, REASON_BAD_LEAVE_TYPE))
        balances = get_balance(conn, target, leave_type)

    if not balances:
        return finish(requester, error(NAME, args, REASON_NO_BALANCE))
    data = {
        "employee_id": target,
        "balances": [{"leave_type": b.leave_type, "balance_days": b.balance_days} for b in balances],
    }
    return finish(requester, success(NAME, args, data))
