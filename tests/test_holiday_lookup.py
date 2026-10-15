"""public_holidays_for(): which calendar applies (Section 5) and its public dates."""
from __future__ import annotations

from datetime import date

import pytest

from app.db import (
    CALENDAR_NO_PUBLIC,
    CALENDAR_OK,
    KIND_OPTIONAL,
    KIND_PUBLIC,
    KIND_UNKNOWN,
    CalendarRow,
    HolidayRow,
    replace_holiday_data,
    run_migrations,
)
from app.retrieval.holiday_lookup import (
    REASON_CONFLICT,
    REASON_NO_CALENDAR,
    REASON_NOT_YET_EFFECTIVE,
    REASON_UNUSABLE,
    ApplicableCalendar,
    NoApplicableCalendar,
    public_holidays_for,
)

ALL = ("all",)


def calendar(doc_id, regions, effective, *, access="all", status=CALENDAR_OK, public=1):
    return CalendarRow(doc_id, effective, tuple(regions), access, status, public_count=public)


def holiday(doc_id, regions, effective, day, kind=KIND_PUBLIC, name="Some day"):
    return [HolidayRow(r, day, name, doc_id, kind, effective, tuple(regions)) for r in regions]


def load(db, calendars, holidays):
    run_migrations(db)
    with db.transaction() as conn:
        replace_holiday_data(conn, calendars, [h for group in holidays for h in group])


def lookup(db, region, reference, levels=ALL):
    with db.read() as conn:
        return public_holidays_for(conn, region, reference, allowed_access_levels=levels)


def dates(result) -> set[date]:
    assert isinstance(result, ApplicableCalendar), result
    return set(result.dates)


@pytest.fixture()
def two_in_calendars(synthetic_db):
    """An older and a newer calendar for the same region, with different dates."""
    load(
        synthetic_db,
        [calendar("OLD", ["IN"], "2025-01-01"), calendar("NEW", ["IN"], "2026-01-01")],
        [
            holiday("OLD", ["IN"], "2025-01-01", "2025-01-10"),
            holiday("OLD", ["IN"], "2025-01-01", "2025-06-06"),
            holiday("NEW", ["IN"], "2026-01-01", "2026-01-12"),
        ],
    )
    return synthetic_db


# --- which calendar --------------------------------------------------------


def test_the_newer_calendar_wins(two_in_calendars):
    result = lookup(two_in_calendars, "IN", date(2026, 6, 1))
    assert result.doc_id == "NEW"
    assert dates(result) == {date(2026, 1, 12)}  # nothing mixed in from OLD


def test_a_reference_date_before_the_newer_effective_date_picks_the_older(two_in_calendars):
    result = lookup(two_in_calendars, "IN", date(2025, 12, 31))
    assert result.doc_id == "OLD"
    assert dates(result) == {date(2025, 1, 10), date(2025, 6, 6)}


def test_the_effective_date_itself_counts_as_in_force(two_in_calendars):
    assert lookup(two_in_calendars, "IN", date(2026, 1, 1)).doc_id == "NEW"
    assert lookup(two_in_calendars, "IN", date(2025, 1, 1)).doc_id == "OLD"


def test_before_every_calendar_there_is_no_applicable_calendar(two_in_calendars):
    result = lookup(two_in_calendars, "IN", date(2024, 12, 31))
    assert isinstance(result, NoApplicableCalendar)
    assert result.reason == REASON_NOT_YET_EFFECTIVE


def test_a_region_without_a_calendar_has_no_applicable_calendar(two_in_calendars):
    result = lookup(two_in_calendars, "DE", date(2026, 6, 1))
    assert isinstance(result, NoApplicableCalendar) and result.reason == REASON_NO_CALENDAR
    assert result.message and "DE" not in result.message


@pytest.mark.parametrize("region", ["", None, "  "])
def test_a_missing_region_never_matches(two_in_calendars, region):
    assert lookup(two_in_calendars, region, date(2026, 6, 1)).reason == REASON_NO_CALENDAR


def test_the_region_match_ignores_case_and_padding(two_in_calendars):
    assert lookup(two_in_calendars, " in ", date(2026, 6, 1)).doc_id == "NEW"


def test_no_calendars_at_all_is_a_result_not_an_empty_set(synthetic_db):
    load(synthetic_db, [], [])
    result = lookup(synthetic_db, "IN", date(2026, 6, 1))
    assert isinstance(result, NoApplicableCalendar)
    assert not result  # falsy, so `if not result:` rejects
    assert result.reason == REASON_NO_CALENDAR


def test_a_found_calendar_is_truthy_even_when_it_holds_one_date(two_in_calendars):
    assert lookup(two_in_calendars, "IN", date(2026, 6, 1))


# --- scope: regional beats GLOBAL ---------------------------------------------


def test_a_regional_calendar_outranks_a_newer_global_one(synthetic_db):
    load(
        synthetic_db,
        [calendar("REG", ["IN"], "2025-01-01"), calendar("GLB", ["GLOBAL"], "2026-01-01")],
        [
            holiday("REG", ["IN"], "2025-01-01", "2026-02-02"),
            holiday("GLB", ["GLOBAL"], "2026-01-01", "2026-02-03"),
        ],
    )
    result = lookup(synthetic_db, "IN", date(2026, 6, 1))
    assert result.doc_id == "REG" and dates(result) == {date(2026, 2, 2)}


def test_a_global_calendar_is_used_when_no_regional_one_exists(synthetic_db):
    load(
        synthetic_db,
        [calendar("GLB", ["GLOBAL"], "2026-01-01")],
        [holiday("GLB", ["GLOBAL"], "2026-01-01", "2026-02-03")],
    )
    assert lookup(synthetic_db, "US", date(2026, 6, 1)).doc_id == "GLB"


def test_a_calendar_covering_several_regions_serves_each_of_them(synthetic_db):
    load(
        synthetic_db,
        [calendar("MULTI", ["IN", "DE"], "2026-01-01")],
        [holiday("MULTI", ["IN", "DE"], "2026-01-01", "2026-02-02")],
    )
    for region in ("IN", "DE"):
        assert dates(lookup(synthetic_db, region, date(2026, 6, 1))) == {date(2026, 2, 2)}
    assert lookup(synthetic_db, "US", date(2026, 6, 1)).reason == REASON_NO_CALENDAR


def test_other_regions_calendars_are_never_mixed_in(synthetic_db):
    load(
        synthetic_db,
        [calendar("IN-CAL", ["IN"], "2026-01-01"), calendar("DE-CAL", ["DE"], "2026-03-01")],
        [
            holiday("IN-CAL", ["IN"], "2026-01-01", "2026-02-02"),
            holiday("DE-CAL", ["DE"], "2026-03-01", "2026-04-04"),
        ],
    )
    assert dates(lookup(synthetic_db, "IN", date(2026, 6, 1))) == {date(2026, 2, 2)}
    assert dates(lookup(synthetic_db, "DE", date(2026, 6, 1))) == {date(2026, 4, 4)}


# --- only public rows -------------------------------------------------------------


def test_optional_and_unknown_rows_are_never_returned(synthetic_db):
    load(
        synthetic_db,
        [calendar("CAL", ["IN"], "2026-01-01")],
        [
            holiday("CAL", ["IN"], "2026-01-01", "2026-02-02", KIND_PUBLIC),
            holiday("CAL", ["IN"], "2026-01-01", "2026-03-03", KIND_OPTIONAL),
            holiday("CAL", ["IN"], "2026-01-01", "2026-04-06", KIND_UNKNOWN),
            holiday("CAL", ["IN"], "2026-01-01", "2026-05-05", "something-else"),
        ],
    )
    assert dates(lookup(synthetic_db, "IN", date(2026, 6, 1))) == {date(2026, 2, 2)}


# --- refusing instead of guessing ---------------------------------------------


def test_two_equally_ranked_calendars_conflict(synthetic_db):
    load(
        synthetic_db,
        [calendar("A", ["IN"], "2026-01-01"), calendar("B", ["IN"], "2026-01-01")],
        [
            holiday("A", ["IN"], "2026-01-01", "2026-02-02"),
            holiday("B", ["IN"], "2026-01-01", "2026-02-03"),
        ],
    )
    result = lookup(synthetic_db, "IN", date(2026, 6, 1))
    assert isinstance(result, NoApplicableCalendar) and result.reason == REASON_CONFLICT


def test_a_tie_among_older_calendars_does_not_matter_when_a_newer_one_wins(synthetic_db):
    load(
        synthetic_db,
        [calendar("A", ["IN"], "2025-01-01"), calendar("B", ["IN"], "2025-01-01"),
         calendar("C", ["IN"], "2026-01-01")],
        [holiday("C", ["IN"], "2026-01-01", "2026-02-02")],
    )
    assert lookup(synthetic_db, "IN", date(2026, 6, 1)).doc_id == "C"


def test_an_unreadable_newest_calendar_does_not_fall_back_to_an_older_one(synthetic_db):
    load(
        synthetic_db,
        [calendar("OLD", ["IN"], "2025-01-01"),
         calendar("NEW", ["IN"], "2026-01-01", status=CALENDAR_NO_PUBLIC, public=0)],
        [holiday("OLD", ["IN"], "2025-01-01", "2025-02-03")],
    )
    result = lookup(synthetic_db, "IN", date(2026, 6, 1))
    assert isinstance(result, NoApplicableCalendar) and result.reason == REASON_UNUSABLE
    # ...but an earlier reference date still reaches the older calendar
    assert lookup(synthetic_db, "IN", date(2025, 6, 1)).doc_id == "OLD"


def test_a_calendar_marked_ok_with_no_public_rows_is_unusable(synthetic_db):
    load(synthetic_db, [calendar("CAL", ["IN"], "2026-01-01")],
         [holiday("CAL", ["IN"], "2026-01-01", "2026-03-03", KIND_OPTIONAL)])
    assert lookup(synthetic_db, "IN", date(2026, 6, 1)).reason == REASON_UNUSABLE


# --- permissions ---------------------------------------------------------------


def test_a_calendar_the_requester_cannot_read_is_not_used(synthetic_db):
    load(
        synthetic_db,
        [calendar("OPEN", ["IN"], "2025-01-01"),
         calendar("HIDDEN", ["IN"], "2026-01-01", access="hr_only")],
        [
            holiday("OPEN", ["IN"], "2025-01-01", "2025-02-03"),
            holiday("HIDDEN", ["IN"], "2026-01-01", "2026-02-02"),
        ],
    )
    as_employee = lookup(synthetic_db, "IN", date(2026, 6, 1), levels=("all",))
    assert as_employee.doc_id == "OPEN"
    as_hr = lookup(synthetic_db, "IN", date(2026, 6, 1), levels=("all", "hr_only"))
    assert as_hr.doc_id == "HIDDEN"


def test_with_no_readable_calendar_the_reason_does_not_reveal_a_hidden_one(synthetic_db):
    load(synthetic_db, [calendar("HIDDEN", ["IN"], "2026-01-01", access="hr_only")],
         [holiday("HIDDEN", ["IN"], "2026-01-01", "2026-02-02")])
    result = lookup(synthetic_db, "IN", date(2026, 6, 1), levels=("all",))
    assert result.reason == REASON_NO_CALENDAR
    assert "HIDDEN" not in repr(result) and "hr_only" not in repr(result)


def test_access_levels_are_required_and_must_be_a_collection(two_in_calendars):
    with two_in_calendars.read() as conn:
        with pytest.raises(TypeError):
            public_holidays_for(conn, "IN", date(2026, 6, 1))  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            public_holidays_for(conn, "IN", date(2026, 6, 1), allowed_access_levels="all")
        # an empty collection means "may read nothing"
        result = public_holidays_for(conn, "IN", date(2026, 6, 1), allowed_access_levels=())
    assert result.reason == REASON_NO_CALENDAR


def test_a_datetime_reference_is_treated_as_its_date(two_in_calendars):
    from datetime import datetime

    assert lookup(two_in_calendars, "IN", datetime(2026, 1, 1, 23, 59)).doc_id == "NEW"
