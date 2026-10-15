"""Working-day arithmetic. Pure: no database, no clock, no I/O."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

_WEEKDAYS_PER_WEEK = 5  # Monday to Friday


def working_days(start: date, end: date, holidays: Iterable[date] = ()) -> int:
    """Monday-to-Friday days from ``start`` to ``end`` (both inclusive), minus
    ``holidays`` that fall on one of those days.

    * A holiday on a Saturday or Sunday changes nothing (it is not a working
      day to begin with), and a holiday outside the range is ignored.
    * A date listed twice is subtracted once.
    * ``end`` before ``start`` is a caller bug, not "zero days": it raises
      ``ValueError`` so apply_leave must validate the range first.
    * ``datetime`` objects are rejected, so a stray time of day cannot make a
      comparison silently wrong.
    """
    holiday_set = set(holidays)
    for value in (start, end, *holiday_set):
        if not isinstance(value, date) or isinstance(value, datetime):
            raise TypeError("working_days works on datetime.date values only")
    if end < start:
        raise ValueError("end date is before the start date")

    total_days = (end - start).days + 1
    full_weeks, remainder = divmod(total_days, 7)
    count = full_weeks * _WEEKDAYS_PER_WEEK + sum(
        1 for offset in range(remainder) if (start + timedelta(days=offset)).weekday() < 5
    )
    holidays_on_working_days = sum(
        1 for h in holiday_set if start <= h <= end and h.weekday() < 5
    )
    return count - holidays_on_working_days
