"""working_days(): Monday-Friday minus the given holidays, both ends inclusive."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.working_days import working_days

# Synthetic calendar facts only: 2026-03-02 is a Monday.
MON = date(2026, 3, 2)


def brute_force(start: date, end: date, holidays) -> int:
    holidays = set(holidays)
    day, total = start, 0
    while day <= end:
        if day.weekday() < 5 and day not in holidays:
            total += 1
        day += timedelta(days=1)
    return total


def test_a_plain_working_week_is_five_days():
    assert working_days(MON, MON + timedelta(days=4)) == 5


def test_both_ends_are_inclusive():
    assert working_days(MON, MON) == 1
    assert working_days(MON, MON + timedelta(days=1)) == 2


def test_a_weekend_only_range_is_zero():
    saturday = MON + timedelta(days=5)
    assert working_days(saturday, saturday + timedelta(days=1)) == 0
    assert working_days(saturday, saturday) == 0


def test_a_range_spanning_a_weekend_skips_it():
    friday = MON + timedelta(days=4)
    assert working_days(friday, friday + timedelta(days=3)) == 2  # Fri, Mon


def test_two_full_weeks_are_ten_days():
    assert working_days(MON, MON + timedelta(days=13)) == 10


def test_holidays_inside_the_range_reduce_the_count():
    wednesday, thursday = MON + timedelta(days=2), MON + timedelta(days=3)
    assert working_days(MON, MON + timedelta(days=4), [wednesday]) == 4
    assert working_days(MON, MON + timedelta(days=4), [wednesday, thursday]) == 3


def test_a_holiday_on_a_weekend_does_not_change_the_count():
    saturday, sunday = MON + timedelta(days=5), MON + timedelta(days=6)
    week = (MON, MON + timedelta(days=6))
    assert working_days(*week, [saturday, sunday]) == working_days(*week) == 5


def test_a_holiday_outside_the_range_is_ignored():
    outside = [MON - timedelta(days=1), MON + timedelta(days=30)]
    assert working_days(MON, MON + timedelta(days=4), outside) == 5


def test_a_holiday_on_the_first_or_last_day_counts():
    friday = MON + timedelta(days=4)
    assert working_days(MON, friday, [MON, friday]) == 3


def test_a_date_listed_twice_is_subtracted_once():
    wednesday = MON + timedelta(days=2)
    assert working_days(MON, MON + timedelta(days=4), [wednesday, wednesday]) == 4


def test_every_working_day_a_holiday_gives_zero():
    days = [MON + timedelta(days=i) for i in range(5)]
    assert working_days(MON, days[-1], days) == 0


def test_holidays_may_be_any_iterable():
    wednesday = MON + timedelta(days=2)
    assert working_days(MON, MON + timedelta(days=4), iter([wednesday])) == 4
    assert working_days(MON, MON + timedelta(days=4), frozenset({wednesday})) == 4


def test_matches_a_day_by_day_count_over_many_ranges():
    holidays = [date(2026, 1, 1) + timedelta(days=n) for n in (0, 5, 13, 14, 40, 41, 100, 200)]
    start = date(2025, 12, 20)
    for offset in range(0, 400, 7):
        for length in (0, 1, 2, 5, 6, 7, 8, 13, 14, 29, 30, 31, 90):
            first = start + timedelta(days=offset)
            last = first + timedelta(days=length)
            assert working_days(first, last, holidays) == brute_force(first, last, holidays), (first, last)


def test_a_leap_day_and_a_year_boundary_are_handled():
    assert working_days(date(2028, 2, 28), date(2028, 3, 1)) == 3  # Mon-Wed, leap day included
    assert working_days(date(2025, 12, 29), date(2026, 1, 2)) == 5


def test_end_before_start_is_an_error_not_zero():
    with pytest.raises(ValueError):
        working_days(MON + timedelta(days=1), MON)


def test_datetimes_are_rejected():
    with pytest.raises(TypeError):
        working_days(datetime(2026, 3, 2, 9), datetime(2026, 3, 3, 9))
    with pytest.raises(TypeError):
        working_days(MON, MON, [datetime(2026, 3, 2, 9)])


def test_non_dates_are_rejected():
    with pytest.raises(TypeError):
        working_days("2026-03-02", MON)
    with pytest.raises(TypeError):
        working_days(MON, MON, ["2026-03-02"])
