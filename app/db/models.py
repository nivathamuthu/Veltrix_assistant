"""Row types returned by the read functions.

``full_name``, ``email`` and ``phone`` are personal data. They are excluded
from ``repr`` so an accidental ``log.info("%s", employee)`` prints only the
employee ID.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Employee:
    employee_id: str
    full_name: str = field(repr=False)
    email: str = field(repr=False)
    phone: str = field(repr=False)
    region: str
    join_date: str
    level: str
    role: str
    manager_id: str | None


@dataclass(frozen=True)
class LeaveBalance:
    employee_id: str
    leave_type: str
    balance_days: float


@dataclass(frozen=True)
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    working_days: int
    status: str
    created_at: str
