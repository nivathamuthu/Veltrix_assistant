"""Which holiday calendar applies to a requester, and its public holidays.

Section 5 (README B4) applied to calendar documents, in code, not by an LLM:

1. Eligibility. A calendar can be used only if the requester may read its
   access_level, its regions include the requester's region or GLOBAL, and its
   effective_date is on or before the reference date.
2. Precedence, in order: authority (every calendar is a calendar, so equal),
   scope (a calendar for the requester's region outranks a GLOBAL one), then
   recency (the later effective_date wins).
3. If that still leaves two calendars, there is no basis to pick one: the
   result is "no applicable calendar" (reason: conflict), not a guess.

The dates come from the winning calendar ONLY; nothing is merged from older,
newer or other-region calendars. Only rows marked ``public`` are returned.

The result is never a bare (possibly empty) set: callers get either an
``ApplicableCalendar`` or a ``NoApplicableCalendar`` carrying the reason, so
apply_leave can reject with a reason instead of silently counting every
weekday as a working day.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Collection

from app.db.holidays import get_public_dates, list_calendars
from app.db.models import CALENDAR_OK, CalendarRow

GLOBAL_REGION = "GLOBAL"  # scope value meaning "applies everywhere" (README B3/B4)

REASON_NO_CALENDAR = "no_calendar"  # no readable calendar covers the region
REASON_NOT_YET_EFFECTIVE = "not_yet_effective"  # calendars exist, all start later
REASON_CONFLICT = "conflict"  # two equally ranked calendars; will not pick one
REASON_UNUSABLE = "unusable"  # the winning calendar yielded no public holidays

_MESSAGES = {
    REASON_NO_CALENDAR: "No holiday calendar applies to your region.",
    REASON_NOT_YET_EFFECTIVE: "No holiday calendar applies to your region on that date.",
    REASON_CONFLICT: "The holiday calendars for your region conflict, so none can be used.",
    REASON_UNUSABLE: "The applicable holiday calendar could not be read.",
}


@dataclass(frozen=True)
class ApplicableCalendar:
    doc_id: str
    effective_date: date
    regions: tuple[str, ...]
    dates: frozenset[date]  # public holidays of THIS calendar only


@dataclass(frozen=True)
class NoApplicableCalendar:
    reason: str  # one of the REASON_* values
    message: str  # fixed text, safe to show to the requester

    def __bool__(self) -> bool:  # `if lookup:` is True only for a real calendar
        return False


HolidayLookup = ApplicableCalendar | NoApplicableCalendar


def _no_calendar(reason: str) -> NoApplicableCalendar:
    return NoApplicableCalendar(reason, _MESSAGES[reason])


def _regions(calendar: CalendarRow) -> frozenset[str]:
    return frozenset(r.upper() for r in calendar.regions)


def public_holidays_for(
    conn: sqlite3.Connection,
    requester_region: str,
    reference_date: date,
    *,
    allowed_access_levels: Collection[str],
) -> HolidayLookup:
    """The calendar that applies to ``requester_region`` on ``reference_date`` and
    its public holiday dates.

    ``allowed_access_levels`` is required: a calendar the requester may not read
    must not influence their working-day count (README rule 2 applies to dates
    handed to the LLM as much as to text). It comes from the requester's role.
    """
    if isinstance(allowed_access_levels, str):
        raise TypeError("allowed_access_levels must be a collection of access levels")
    allowed = frozenset(allowed_access_levels)
    region = (requester_region or "").strip().upper()
    reference = reference_date.date() if hasattr(reference_date, "date") else reference_date

    eligible = [
        c
        for c in list_calendars(conn)
        if c.access_level in allowed and (region in _regions(c) or GLOBAL_REGION in _regions(c))
    ]
    if not region or not eligible:
        return _no_calendar(REASON_NO_CALENDAR)

    in_force = [c for c in eligible if date.fromisoformat(c.effective_date) <= reference]
    if not in_force:
        return _no_calendar(REASON_NOT_YET_EFFECTIVE)

    # Scope: a calendar naming the requester's region outranks a GLOBAL one.
    regional = [c for c in in_force if region in _regions(c)]
    candidates = regional or in_force

    # Recency: the latest effective_date; a tie cannot be resolved.
    newest = max(c.effective_date for c in candidates)
    winners = [c for c in candidates if c.effective_date == newest]
    if len(winners) > 1:
        return _no_calendar(REASON_CONFLICT)

    winner = winners[0]
    if winner.status != CALENDAR_OK:
        return _no_calendar(REASON_UNUSABLE)  # never fall back to an older calendar
    dates = frozenset(date.fromisoformat(d) for d in get_public_dates(conn, winner.doc_id))
    if not dates:
        return _no_calendar(REASON_UNUSABLE)
    return ApplicableCalendar(
        doc_id=winner.doc_id,
        effective_date=date.fromisoformat(winner.effective_date),
        regions=winner.regions,
        dates=dates,
    )
