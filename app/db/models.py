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


# Holiday kinds. Only KIND_PUBLIC may ever count as a non-working day; KIND_UNKNOWN
# is what a table gets when its heading does not say which it is (never used).
KIND_PUBLIC = "public"
KIND_OPTIONAL = "optional"
KIND_UNKNOWN = "unknown"

# Outcome of extracting one calendar document.
CALENDAR_OK = "ok"  # parsed, at least one public holiday found
CALENDAR_NO_PUBLIC = "no_public_holidays"  # parsed, but no usable public rows
CALENDAR_UNAVAILABLE = "unavailable"  # in the manifest, file missing or unparseable


@dataclass(frozen=True)
class HolidayRow:
    """One holiday of one calendar document, for one of its regions."""

    region: str
    date: str  # ISO yyyy-mm-dd
    name: str
    doc_id: str
    kind: str  # KIND_PUBLIC | KIND_OPTIONAL | KIND_UNKNOWN
    effective_date: str  # of the calendar document (from the manifest)
    regions: tuple[str, ...]  # of the calendar document (from the manifest)


@dataclass(frozen=True)
class CalendarRow:
    """A calendar document from the manifest and how its extraction went.

    Kept even when no holiday could be read, so a lookup can tell "the newest
    calendar is unusable" from "there is no calendar" and never falls back to
    an older one in silence.
    """

    doc_id: str
    effective_date: str
    regions: tuple[str, ...]
    access_level: str
    status: str  # CALENDAR_OK | CALENDAR_NO_PUBLIC | CALENDAR_UNAVAILABLE
    public_count: int = 0
    optional_count: int = 0
    unknown_count: int = 0
    skipped_rows: int = 0


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
