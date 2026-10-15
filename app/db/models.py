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
class Segment:
    """Where one source section sits inside a chunk's text (character offsets).

    Merged chunks hold several sections; knowing each one's locator lets a
    search result cite the section that actually matches the query.
    """

    locator: str
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage. Metadata is copied from the manifest entry."""

    chunk_id: str
    doc_id: str
    locator: str  # nearest heading, else "p. N" (first section if merged)
    regions: tuple[str, ...]
    effective_date: str
    doc_type: str
    access_level: str
    text: str
    token_count: int
    segments: tuple[Segment, ...] = ()  # in memory only; not stored in the table


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
