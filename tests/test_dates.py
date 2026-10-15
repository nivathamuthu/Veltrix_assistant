"""Tolerant date parsing that never guesses (Phase 3c)."""
from __future__ import annotations

from datetime import date

import pytest

from app.ingestion.dates import parse_date, parse_date_cell, parse_weekday

EXPECTED = date(2026, 1, 15)  # a Thursday


@pytest.mark.parametrize(
    "text",
    [
        # day month year
        "15 January 2026", "15 january 2026", "15 JANUARY 2026", "15 Jan 2026", "15 Jan. 2026",
        "15-Jan-2026", "15/Jan/2026", "15th January 2026", "15th of January, 2026", "the 15th of January 2026",
        "  15   January   2026  ", "15\u00a0January\u00a02026",  # non-breaking spaces
        # month day year
        "Jan 15, 2026", "January 15 2026", "January 15th, 2026", "Jan. 15, 2026",
        # ISO and friends
        "2026-01-15", "2026/01/15", "2026.01.15", "2026-1-15",
        # numeric, unambiguous (day above 12)
        "15/01/2026", "15.01.2026", "15-01-2026", "01/15/2026",
        # with a weekday, in every place it is found
        "Thursday, 15 January 2026", "Thursday 15 January 2026", "Thu, 15 Jan 2026",
        "Thu. 15 Jan 2026", "15 January 2026 (Thursday)", "15 January 2026, Thursday",
        "Thursday, January 15, 2026", "Thursday 2026-01-15", "15 Jan 2026 (Thu)",
    ],
)
def test_supported_formats_all_give_the_same_date(text):
    assert parse_date(text) == EXPECTED


def test_september_has_two_common_abbreviations():
    assert parse_date("3 Sep 2026") == parse_date("3 Sept 2026") == date(2026, 9, 3)


def test_a_cell_that_also_names_the_weekday_reports_it():
    parsed = parse_date_cell("Thursday, 15 January 2026")
    assert (parsed.value, parsed.weekday) == (EXPECTED, 3)
    assert parse_date_cell("15 January 2026").weekday is None


@pytest.mark.parametrize(
    "text",
    [
        None, "", "   ", "-", "TBC", "n/a",
        "15 January",  # no year: never assume one
        "January 2026", "2026",
        "15/01/26", "15 January 26",  # two-digit year
        "03/04/2026", "04/03/2026", "12.11.2026",  # day-first or month-first: unknowable
        "24-26 December 2026", "24 to 26 December 2026",  # a range is not one date
        "30 February 2026", "31 April 2026", "2026-13-01", "0/0/2026", "32 January 2026",
        "15 Januar 2026", "15 Foo 2026",  # not an English month
        "15 January 2026 and 16 January 2026",
        "15 January 2026 (subject to confirmation)",  # extra words: not exactly a date
        "see section 5", "Row: Date", "Republic Day",
        "Monday 15 January 2026 Tuesday",  # two different weekdays
        "1000-01-01", "3000-01-01",  # outside any plausible calendar
    ],
)
def test_anything_unclear_is_rejected_not_guessed(text):
    assert parse_date(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("03/03/2026", date(2026, 3, 3)),  # same either way
        ("25/12/2026", date(2026, 12, 25)),
        ("12/25/2026", date(2026, 12, 25)),
        ("1.1.2026", date(2026, 1, 1)),
    ],
)
def test_numeric_dates_are_accepted_only_when_the_order_cannot_matter(text, expected):
    assert parse_date(text) == expected


def test_a_stale_weekday_does_not_change_the_date():
    """The parser reports the weekday; it never uses it to 'fix' the date."""
    parsed = parse_date_cell("Monday, 15 January 2026")  # really a Thursday
    assert parsed.value == EXPECTED and parsed.weekday == 0


@pytest.mark.parametrize(
    "text, expected",
    [("Thursday", 3), ("thu", 3), ("Thu.", 3), ("MONDAY", 0), ("Sun", 6), (" Saturday ", 5)],
)
def test_parse_weekday(text, expected):
    assert parse_weekday(text) == expected


@pytest.mark.parametrize("text", [None, "", "Thursday 15", "Holiday", "Day", "weekday"])
def test_parse_weekday_rejects_everything_else(text):
    assert parse_weekday(text) is None
