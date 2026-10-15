"""Shared pieces of the five leave tools (README B3, B6).

Every tool takes the authenticated requester (an ``Employee`` the caller loaded from the
DB by the ``X-Employee-Id`` header) as its first argument, then ONLY the B6 argument
names. Nothing here trusts identity, role or region from anywhere else (rules 4 and 5).

No clock is read in this package: "today" is always a parameter (APP_TODAY from config).
Nothing here logs a name, an email or a phone: only employee ids and the status.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.db import Employee

logger = logging.getLogger("app.tools")

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
_STATUSES = (STATUS_SUCCESS, STATUS_ERROR)  # "pending_confirmation" is added by a later step

# One fixed text for "you may not see that employee", identical whether or not the
# employee exists. It names nobody.
REFUSAL_MESSAGE = "I can't share information about that employee."

LEAVE_TYPES = ("annual", "sick")
REQUEST_STATUSES = ("pending", "approved", "cancelled")
LEAVE_YEAR = 2026  # README B6/B9: leave outside 2026 is out of scope

# The only argument names a ToolResult may carry (README B6).
B6_ARGUMENTS = frozenset(
    {"employee_id", "leave_type", "status", "start_date", "end_date", "request_id"}
)

INVALID_ARG = "[invalid]"

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}", re.ASCII)
_ARG_VALUE = re.compile(r"[A-Za-z0-9_.:\-]{1,64}", re.ASCII)
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns.

    ``args`` holds only B6 argument names with ids, types and dates: never a name,
    email or phone (see ``safe_args``). ``data`` is empty for an error and ``reason`` is
    the specific, requester-safe explanation. ``status`` is "success" or "error".
    """

    name: str
    args: dict[str, str] = field(default_factory=dict)
    status: str = STATUS_SUCCESS
    data: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}")
        unknown = set(self.args) - B6_ARGUMENTS
        if unknown:
            raise ValueError("args may only use B6 argument names")
        if self.status == STATUS_ERROR and not (isinstance(self.reason, str) and self.reason):
            raise ValueError("an error needs a reason")
        if self.status == STATUS_SUCCESS and self.reason is not None:
            raise ValueError("a success has no reason")
        object.__setattr__(self, "args", dict(self.args))
        object.__setattr__(self, "data", dict(self.data))


def safe_args(**values: object) -> dict[str, str]:
    """The arguments to echo in a result: those that were given, and only if they look like
    an id, type or date. Anything else (free text, a name, an email, a number) becomes
    ``[invalid]``, so personal data an LLM put into an argument never reaches a log."""
    echoed: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, str) and _ARG_VALUE.fullmatch(value):
            echoed[key] = value
        else:
            echoed[key] = INVALID_ARG
    return echoed


def success(name: str, args: dict[str, str], data: dict[str, Any]) -> ToolResult:
    return ToolResult(name, args, STATUS_SUCCESS, data, None)


def error(name: str, args: dict[str, str], reason: str) -> ToolResult:
    return ToolResult(name, args, STATUS_ERROR, {}, reason)


def refusal(name: str, args: dict[str, str]) -> ToolResult:
    """The one answer for every target the requester may not see, existing or not."""
    return error(name, args, REFUSAL_MESSAGE)


def finish(requester: Employee, result: ToolResult) -> ToolResult:
    """Log one line (tool, requester id, status) and return the result."""
    logger.info("tool_call tool=%s requester=%s status=%s", result.name, requester.employee_id, result.status)
    return result


def require_requester(requester: object) -> Employee:
    if not isinstance(requester, Employee):
        raise TypeError("requester must be the Employee record loaded from the database")
    return requester


def require_today(today: object) -> date:
    if isinstance(today, datetime) or not isinstance(today, date):
        raise TypeError("today must be a datetime.date (APP_TODAY), passed in by the caller")
    return today


def is_valid_id(value: object) -> bool:
    """An employee id or request id shape. Rejects bulk markers, lists, free text."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def parse_iso_date(value: object) -> date | None:
    """A real calendar date written exactly YYYY-MM-DD, else None."""
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def is_leave_type(value: object) -> bool:
    return isinstance(value, str) and value in LEAVE_TYPES


def is_request_status(value: object) -> bool:
    return isinstance(value, str) and value in REQUEST_STATUSES
