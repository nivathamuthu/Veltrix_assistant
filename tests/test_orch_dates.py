"""Leave-date parsing (README B9): deterministic, given today; never guesses.

Today in these tests is Thursday 2026-10-15 unless a case says otherwise.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.orchestrator.dates import (
    REASON_AMBIGUOUS,
    REASON_END_BEFORE_START,
    REASON_HALF_DAY,
    REASON_INVALID,
    REASON_MISSING,
    REASON_OUTSIDE_YEAR,
    REASON_PAST,
    DateResult,
    parse_leave_dates,
)

TODAY = date(2026, 10, 15)  # a Thursday


def d(month, day, year=2026):
    return date(year, month, day)


def parse(text, today=TODAY):
    return parse_leave_dates(text, today)


def test_reason_strings_are_the_agreed_words():
    assert REASON_AMBIGUOUS == "ambiguous"
    assert REASON_PAST == "in the past"
    assert REASON_OUTSIDE_YEAR == "outside 2026"
    assert REASON_MISSING == "no date"
    assert REASON_END_BEFORE_START == "end before start"
    assert REASON_INVALID == "invalid date"
    assert REASON_HALF_DAY == "half day"


# --- single dates: start = end ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-10-19", d(10, 19)),
        ("19 October", d(10, 19)),
        ("19 october", d(10, 19)),
        ("19 Oct", d(10, 19)),
        ("19th October", d(10, 19)),
        ("the 19th of October", d(10, 19)),
        ("October 19", d(10, 19)),
        ("Oct 19", d(10, 19)),
        ("Oct. 19th", d(10, 19)),
        ("Oct 19 2026", d(10, 19)),
        ("Oct 19, 2026", d(10, 19)),
        ("19 October 2026", d(10, 19)),
        ("on 19 Oct", d(10, 19)),
        ("1 November", d(11, 1)),  # no year means 2026
        ("1st of November", d(11, 1)),
        ("Dec 25", d(12, 25)),
        ("31 December", d(12, 31)),
        ("15 October", d(10, 15)),  # today is allowed
        ("today", d(10, 15)),
        ("tomorrow", d(10, 16)),
        ("day after tomorrow", d(10, 17)),
        ("TOMORROW!", d(10, 16)),
        ("  tomorrow  ", d(10, 16)),
    ],
)
def test_a_single_date_gives_start_equal_to_end(text, expected):
    result = parse(text)
    assert result.ok, result
    assert (result.start, result.end, result.reason) == (expected, expected, None)


def test_a_date_inside_a_sentence_is_found():
    assert parse("I would like annual leave on 19 October please").start == d(10, 19)
    assert parse("can you book me off tomorrow?").start == d(10, 16)


def test_the_word_may_is_a_month_only_before_or_after_a_day_number():
    assert parse("may I take leave on 19 October").start == d(10, 19)
    assert parse("May 19").reason == REASON_PAST  # a real date: 19 May 2026


def test_no_year_means_2026_not_the_current_year():
    far = date(2031, 1, 1)
    assert parse_leave_dates("1 November", far).reason == REASON_PAST  # 2026-11-01 is before 2031
    assert parse_leave_dates("1 November", date(2026, 1, 1)).start == d(11, 1)


# --- weekdays -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("on Monday", d(10, 19)),
        ("Monday", d(10, 19)),
        ("on mon", d(10, 19)),
        ("next Monday", d(10, 19)),
        ("this Monday", d(10, 19)),
        ("coming Monday", d(10, 19)),
        ("on Tuesday", d(10, 20)),
        ("Wednesday", d(10, 21)),
        ("Friday", d(10, 16)),  # the first Friday after today
        ("this Friday", d(10, 16)),
        ("next Friday", d(10, 16)),  # README B9: the first Friday after today
        ("next Thursday", d(10, 22)),  # today is Thursday: the first one AFTER today
        ("Saturday", d(10, 17)),  # a weekend day parses; apply_leave rejects zero working days
    ],
)
def test_weekday_names_mean_the_first_such_day_after_today(text, expected):
    result = parse(text)
    assert result.ok, result
    assert result.start == result.end == expected


@pytest.mark.parametrize("text", ["on Thursday", "Thursday", "this Thursday", "on thu"])
def test_naming_todays_weekday_without_next_is_ambiguous(text):
    result = parse(text)
    assert not result.ok and result.reason == REASON_AMBIGUOUS
    assert result.start is None and result.end is None


def test_weeks_start_on_monday():
    sunday = date(2026, 10, 18)  # next Monday is tomorrow
    assert parse_leave_dates("next week", sunday) == parse_leave_dates("next week", sunday)
    assert parse_leave_dates("next week", sunday).start == d(10, 19)
    assert parse_leave_dates("next week", date(2026, 10, 19)).start == d(10, 26)  # a Monday: the week AFTER
    assert parse_leave_dates("next week", date(2026, 10, 24)).start == d(10, 26)  # a Saturday


def test_next_week_is_monday_to_friday_of_the_following_week():
    result = parse("next week")
    assert result.ok and (result.start, result.end) == (d(10, 19), d(10, 23))


@pytest.mark.parametrize(
    "text, expected",
    [("Friday next week", d(10, 23)), ("next week Wednesday", d(10, 21)), ("on Monday of next week", d(10, 19)),
     ("next week on Tuesday", d(10, 20))],
)
def test_a_weekday_of_next_week(text, expected):
    result = parse(text)
    assert result.ok and result.start == result.end == expected


# --- ranges --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, start, end",
    [
        ("from 19 October to 21 October", d(10, 19), d(10, 21)),
        ("19 Oct to 21 Oct", d(10, 19), d(10, 21)),
        ("19 Oct - 21 Oct", d(10, 19), d(10, 21)),
        ("19 Oct – 21 Oct", d(10, 19), d(10, 21)),  # en dash
        ("19 Oct till 21 Oct", d(10, 19), d(10, 21)),
        ("19 Oct until 21 Oct", d(10, 19), d(10, 21)),
        ("19 Oct through 23 Oct", d(10, 19), d(10, 23)),
        ("from 19 to 21 October", d(10, 19), d(10, 21)),
        ("19-21 October", d(10, 19), d(10, 21)),
        ("19th - 21st October", d(10, 19), d(10, 21)),
        ("Oct 19-21", d(10, 19), d(10, 21)),
        ("October 19 to 21", d(10, 19), d(10, 21)),
        ("2026-10-19 to 2026-10-23", d(10, 19), d(10, 23)),
        ("2026-10-19 - 2026-10-23", d(10, 19), d(10, 23)),
        ("from 28 October to 3 November", d(10, 28), d(11, 3)),  # across a month end
        ("next Mon to Wed", d(10, 19), d(10, 21)),
        ("from Monday to Wednesday", d(10, 19), d(10, 21)),
        ("Monday till Friday", d(10, 19), d(10, 23)),
        ("next Monday to next Wednesday", d(10, 19), d(10, 21)),
        ("Mon - Fri", d(10, 19), d(10, 23)),
        ("from tomorrow to Monday", d(10, 16), d(10, 19)),
        ("from tomorrow to Friday next week", d(10, 16), d(10, 23)),
        ("today to tomorrow", d(10, 15), d(10, 16)),
        ("FROM 19 OCTOBER TO 21 OCTOBER.", d(10, 19), d(10, 21)),
        ("19 October 2026 to 21 October 2026", d(10, 19), d(10, 21)),
        ("19 Oct to 19 Oct", d(10, 19), d(10, 19)),  # one day written as a range
    ],
)
def test_ranges(text, start, end):
    result = parse(text)
    assert result.ok, result
    assert (result.start, result.end, result.reason) == (start, end, None)


def test_the_end_weekday_of_a_range_is_the_first_one_on_or_after_the_start():
    assert parse("Monday to Monday") == parse("Monday")  # the same day
    r = parse("next Friday to Monday")
    assert (r.start, r.end) == (d(10, 16), d(10, 19))


# --- failures: never guess ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "19/10/2026", "10/11/2026", "19/10", "5/6", "19.10.2026", "19-10-2026", "1/2/26",
        "from 19/10/2026 to 21/10/2026",
        "next month", "this month", "this week", "next weekend", "soon", "sometime next year",
        "in November", "during December", "for March",
        "the 19th", "on the 21st",
        "19 October and 21 October", "19 Oct, 21 Oct, 23 Oct", "19 Oct 21 Oct",
        "19 Oct to 21", "19 Oct to",
        "for 3 days starting 19 October", "19 October for a week", "19 October for 3 days", "19 Oct for two days",
        "next week to the week after", "from next week to Friday",
        "Thursday to Friday",  # today is Thursday: the start is ambiguous
    ],
)
def test_unclear_input_is_ambiguous_and_gives_no_dates(text):
    result = parse(text)
    assert not result.ok, (text, result)
    assert result.reason == REASON_AMBIGUOUS, (text, result)
    assert result.start is None and result.end is None


@pytest.mark.parametrize("text", ["", "   ", "I need some time off", "apply for annual leave", "hello", "yes"])
def test_text_without_a_date_gives_no_date(text):
    result = parse(text)
    assert not result.ok and result.reason == REASON_MISSING
    assert result.start is None and result.end is None


@pytest.mark.parametrize(
    "text, start",
    [
        ("yesterday", d(10, 14)),
        ("14 October", d(10, 14)),
        ("1 August", d(8, 1)),
        ("last Friday", d(10, 9)),
        ("2026-10-14", d(10, 14)),
        ("from 12 October to 14 October", d(10, 12)),
        ("from 12 October to 20 October", d(10, 12)),  # starts in the past even though it ends later
    ],
)
def test_dates_before_today_are_in_the_past(text, start):
    result = parse(text)
    assert not result.ok and result.reason == REASON_PAST
    assert result.start == start  # still reported, so a read-only view (team leave) may use it


@pytest.mark.parametrize(
    "text",
    ["2027-01-04", "4 January 2027", "Oct 19 2025", "19 October 2027", "from 30 December to 2 January 2027",
     "2025-12-31", "from 28 December 2026 to 5 January 2027"],
)
def test_dates_outside_2026_are_rejected(text):
    result = parse(text)
    assert not result.ok and result.reason == REASON_OUTSIDE_YEAR


def test_relative_dates_that_land_outside_2026_are_rejected():
    assert parse_leave_dates("tomorrow", date(2026, 12, 31)).reason == REASON_OUTSIDE_YEAR
    assert parse_leave_dates("next week", date(2026, 12, 28)).reason == REASON_OUTSIDE_YEAR
    assert parse_leave_dates("next Monday", date(2026, 12, 29)).reason == REASON_OUTSIDE_YEAR


def test_outside_2026_is_reported_before_in_the_past():
    assert parse("2025-03-01").reason == REASON_OUTSIDE_YEAR


@pytest.mark.parametrize("text", ["from 21 October to 19 October", "23 Oct - 19 Oct", "2026-10-23 to 2026-10-19",
                                  "from Friday next week to Monday next week"])
def test_an_end_before_the_start_is_rejected(text):
    result = parse(text)
    assert not result.ok and result.reason == REASON_END_BEFORE_START


@pytest.mark.parametrize("text", ["31 February", "30 February 2026", "2026-02-30", "31 April", "from 19 Oct to 31 Nov"])
def test_impossible_calendar_dates_are_invalid(text):
    result = parse(text)
    assert not result.ok and result.reason == REASON_INVALID


@pytest.mark.parametrize("text", ["half day on 19 October", "half-day tomorrow", "a half day next Monday",
                                  "tomorrow for half a day", "half a day on Friday", "19 October, half days please"])
def test_half_days_are_not_supported(text):
    assert parse(text).reason == REASON_HALF_DAY


# --- shape and determinism ---------------------------------------------------------------------------------------------------


def test_the_result_is_a_value_object_and_ok_means_dates_and_no_reason():
    good, bad = parse("tomorrow"), parse("19/10/2026")
    assert isinstance(good, DateResult) and good.ok and good.reason is None
    assert not bad.ok and bad.reason
    with pytest.raises(Exception):
        good.start = None  # type: ignore[misc]


def test_the_same_input_gives_the_same_answer_and_depends_only_on_today():
    assert parse("next Friday") == parse("next Friday")
    assert parse_leave_dates("next Friday", date(2026, 10, 16)).start == d(10, 23)  # today IS a Friday
    assert parse_leave_dates("tomorrow", date(2026, 3, 1)).start == d(3, 2)


def test_only_a_date_object_is_accepted_as_today():
    with pytest.raises(TypeError):
        parse_leave_dates("tomorrow", "2026-10-15")  # type: ignore[arg-type]


@pytest.mark.parametrize("text", [None, 5, ["tomorrow"]])
def test_non_text_gives_no_date(text):
    assert parse_leave_dates(text, TODAY).reason == REASON_MISSING  # type: ignore[arg-type]


def test_the_module_reads_no_clock():
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "app" / "orchestrator" / "dates.py").read_text("utf-8")
    for forbidden in ("datetime.now", "date.today", "datetime.today", "time.time(", "utcnow"):
        assert forbidden not in source


def test_long_input_is_handled_quickly():
    import time

    text = ("please " * 2000) + "on 19 October"
    start = time.monotonic()
    assert parse(text).start == d(10, 19)
    assert time.monotonic() - start < 1.0
