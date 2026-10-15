"""The five leave tools (README B3, B6). Plain functions: no LLM and no orchestrator here.

Each takes the authenticated requester (an ``Employee`` loaded from the DB by the caller
from ``X-Employee-Id``) first, then only the B6 argument names, then keyword-only
infrastructure (``db``, and ``today`` where a date rule needs it). Permission checks run
inside the tool, before the target's data is read, and never depend on LLM arguments.
"""
from app.tools.common import (
    B6_ARGUMENTS,
    LEAVE_TYPES,
    LEAVE_YEAR,
    REFUSAL_MESSAGE,
    REQUEST_STATUSES,
    STATUS_ERROR,
    STATUS_PENDING_CONFIRMATION,
    STATUS_SUCCESS,
    ToolResult,
)
from app.tools.leave_apply import LeaveValidation, apply_leave, validate_apply_leave
from app.tools.leave_balance import get_leave_balance
from app.tools.leave_cancel import CancelValidation, cancel_leave, validate_cancel_leave
from app.tools.leave_requests import list_leave_requests
from app.tools.team_leave import get_team_leave

__all__ = [
    "B6_ARGUMENTS",
    "CancelValidation",
    "LEAVE_TYPES",
    "LEAVE_YEAR",
    "LeaveValidation",
    "REFUSAL_MESSAGE",
    "REQUEST_STATUSES",
    "STATUS_ERROR",
    "STATUS_PENDING_CONFIRMATION",
    "STATUS_SUCCESS",
    "ToolResult",
    "apply_leave",
    "cancel_leave",
    "get_leave_balance",
    "get_team_leave",
    "list_leave_requests",
    "validate_apply_leave",
    "validate_cancel_leave",
]
