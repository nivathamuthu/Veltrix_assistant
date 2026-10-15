"""apply_leave and validate_apply_leave (README B6): rules, working days, one transaction."""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime

import pytest

import app.tools.leave_apply as leave_apply
from app.db import Database
from app.tools import ToolResult, apply_leave, validate_apply_leave
from app.tools.leave_apply import LeaveValidation
from tests.db_helpers import dump_tables
from tests.tool_helpers import (
    CORE_TABLES,
    TODAY,
    balance_of,
    pii_values,
    request_rows,
    run_sql,
    who,
)


def apply(db, requester_id, leave_type, start, end, **kw):
    return apply_leave(who(db, requester_id), leave_type, start, end, db=db, today=TODAY, **kw)


def validate(db, requester_id, leave_type, start, end):
    return validate_apply_leave(who(db, requester_id), leave_type, start, end, db=db, today=TODAY)


def snapshot(db):
    return dump_tables(db.path, CORE_TABLES)


# Dates used below (today is Thu 2026-10-15; IN public holidays 10-24 Sat, 11-11 Wed, 12-25 Fri;
# optional 11-12 Thu does not count; E1 has R1 pending 11-02..11-04, R2 approved 12-21..12-24,
# R3 cancelled sick 11-10..11-11; E2 annual 3.5; E1 annual 10, sick 5).


# --- working days -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2026-10-19", "2026-10-23", 5),  # Monday to Friday
        ("2026-10-15", "2026-10-15", 1),  # today itself (start >= today)
        ("2026-10-16", "2026-10-20", 3),  # Fri + Mon + Tue: the weekend inside is skipped
        ("2026-10-19", "2026-10-30", 10),  # two weeks; the Saturday holiday on 10-24 changes nothing
        ("2026-11-09", "2026-11-13", 4),  # the public holiday on Wed 11-11 reduces it; optional 11-12 does not
        ("2026-12-28", "2026-12-31", 4),
        ("2026-11-05", "2026-11-06", 2),  # starts the day after E1's pending request ends
    ],
)
def test_working_days_are_weekdays_minus_public_holidays(tools_db, start, end, expected):
    result = validate(tools_db, "E1", "annual", start, end)
    assert result.ok and result.working_days == expected, result


@pytest.mark.parametrize(
    "start, end",
    [
        ("2026-10-17", "2026-10-18"),  # Saturday and Sunday
        ("2026-10-17", "2026-10-17"),
        ("2026-10-24", "2026-10-24"),  # a public holiday that is also a Saturday
        ("2026-11-11", "2026-11-11"),  # a weekday public holiday
    ],
)
def test_a_range_with_no_working_days_is_rejected(tools_db, start, end):
    before = snapshot(tools_db)
    result = apply(tools_db, "E1", "annual", start, end)
    assert result.status == "error"
    assert "working day" in result.reason.lower()
    assert snapshot(tools_db) == before


def test_a_holiday_only_range_made_of_a_holiday_and_a_weekend_is_rejected(tools_db):
    # Fri 12-25 is a public holiday, 12-26 and 12-27 are the weekend: nothing left to take
    result = apply(tools_db, "E1", "annual", "2026-12-25", "2026-12-27")
    assert result.status == "error" and "working day" in result.reason.lower()


def test_an_optional_holiday_does_not_reduce_the_working_days(tools_db):
    assert validate(tools_db, "E1", "annual", "2026-11-12", "2026-11-12").working_days == 1


def test_us_requesters_use_the_us_calendar(tools_db):
    # Thanksgiving-like 2026-11-26 (Thu) is public for US only
    assert validate(tools_db, "E3", "annual", "2026-11-23", "2026-11-27").working_days == 4
    assert validate(tools_db, "E1", "annual", "2026-11-23", "2026-11-27").working_days == 5


def test_the_calendar_comes_from_the_requesters_region_and_the_extracted_tables(tools_db):
    run_sql(tools_db, "DELETE FROM holidays WHERE date = '2026-11-11'")
    assert validate(tools_db, "E1", "annual", "2026-11-09", "2026-11-13").working_days == 5
    run_sql(tools_db, "INSERT INTO holidays (region, date, name, doc_id, kind, effective_date, regions) "
                      "VALUES ('IN', '2026-11-10', 'x', 'CAL-IN', 'public', '2026-01-01', '[\"IN\"]')")
    assert validate(tools_db, "E1", "annual", "2026-11-09", "2026-11-13").working_days == 4


# --- no usable calendar ----------------------------------------------------------------------------------------


def test_no_applicable_calendar_rejects_with_the_lookups_reason(tools_db):
    # E4 is in DE and there is no DE calendar
    from app.retrieval.holiday_lookup import NoApplicableCalendar, public_holidays_for

    with tools_db.read() as conn:
        lookup = public_holidays_for(conn, "DE", TODAY, allowed_access_levels=("all",))
    assert isinstance(lookup, NoApplicableCalendar)
    before = snapshot(tools_db)
    result = apply(tools_db, "E4", "annual", "2026-10-19", "2026-10-23")
    assert result.status == "error"
    assert result.reason == lookup.message
    assert snapshot(tools_db) == before


def test_an_unreadable_calendar_rejects_and_never_counts_every_weekday(tools_db):
    run_sql(
        tools_db,
        "INSERT INTO holiday_calendars (doc_id, effective_date, regions, access_level, status, public_count, "
        "optional_count, unknown_count, skipped_rows) VALUES ('CAL-DE', '2026-01-01', '[\"DE\"]', 'all', "
        "'unavailable', 0, 0, 0, 0)",
    )
    before = snapshot(tools_db)
    result = apply(tools_db, "E4", "annual", "2026-10-19", "2026-10-23")
    assert result.status == "error" and "calendar" in result.reason.lower()
    assert snapshot(tools_db) == before


def test_a_calendar_the_requester_may_not_read_is_not_used(tools_db):
    run_sql(tools_db, "UPDATE holiday_calendars SET access_level = 'hr' WHERE doc_id = 'CAL-IN'")
    employee = apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-23")
    assert employee.status == "error" and "calendar" in employee.reason.lower()
    hr = apply(tools_db, "H1", "annual", "2026-10-19", "2026-10-23")  # hr may read it
    assert hr.status == "success"


def test_a_calendar_not_yet_in_force_today_is_not_used(tools_db):
    run_sql(tools_db, "UPDATE holiday_calendars SET effective_date = '2027-01-01' WHERE doc_id = 'CAL-IN'")
    result = apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-23")
    assert result.status == "error" and "calendar" in result.reason.lower()


# --- balance -------------------------------------------------------------------------------------------------------


def test_an_exact_balance_is_allowed_and_leaves_zero(tools_db):
    result = apply(tools_db, "E1", "sick", "2026-10-19", "2026-10-23")  # 5 days against a balance of 5
    assert result.status == "success"
    assert balance_of(tools_db, "E1", "sick") == 0.0


def test_one_day_over_the_balance_is_rejected(tools_db):
    before = snapshot(tools_db)
    result = apply(tools_db, "E1", "sick", "2026-10-19", "2026-10-26")  # 6 working days
    assert result.status == "error" and "balance" in result.reason.lower()
    assert snapshot(tools_db) == before


def test_a_balance_with_a_half_day_fraction(tools_db):
    # E2 has 3.5 annual: 3 days is fine and leaves 0.5; 4 days is half a day too many
    over = apply(tools_db, "E2", "annual", "2026-10-19", "2026-10-22")
    assert over.status == "error" and "balance" in over.reason.lower()
    ok = apply(tools_db, "E2", "annual", "2026-10-19", "2026-10-21")
    assert ok.status == "success" and balance_of(tools_db, "E2", "annual") == 0.5


def test_a_request_of_balance_plus_half_a_day_is_rejected(tools_db):
    run_sql(tools_db, "UPDATE leave_balances SET balance_days = 3.5 WHERE employee_id = 'E1' AND leave_type = 'annual'")
    assert apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-22").status == "error"  # 4 > 3.5
    run_sql(tools_db, "UPDATE leave_balances SET balance_days = 4.0 WHERE employee_id = 'E1' AND leave_type = 'annual'")
    assert apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-22").status == "success"  # 4 == 4.0


def test_the_balance_checked_is_the_one_for_the_requested_leave_type(tools_db):
    run_sql(tools_db, "UPDATE leave_balances SET balance_days = 0 WHERE employee_id = 'E1' AND leave_type = 'sick'")
    assert apply(tools_db, "E1", "sick", "2026-10-19", "2026-10-19").status == "error"
    assert apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-19").status == "success"


def test_no_balance_row_for_the_leave_type_is_a_rejection(tools_db):
    before = snapshot(tools_db)
    assert apply(tools_db, "N1", "annual", "2026-10-19", "2026-10-19").status == "error"  # no rows at all
    assert apply(tools_db, "E2", "sick", "2026-10-19", "2026-10-19").status == "error"  # no sick row
    assert snapshot(tools_db) == before


# --- overlap with the requester's own requests ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end",
    [
        ("2026-11-03", "2026-11-03"),  # inside E1's pending 11-02..11-04
        ("2026-10-30", "2026-11-02"),  # shares the first day only (Fri..Mon)
        ("2026-11-04", "2026-11-05"),  # shares the last day only
        ("2026-11-02", "2026-11-04"),  # identical
        ("2026-11-01", "2026-11-05"),  # swallows it (Sunday to Thursday)
        ("2026-12-23", "2026-12-28"),  # overlaps the approved 12-21..12-24
        ("2026-12-24", "2026-12-24"),
    ],
)
def test_sharing_any_date_with_a_pending_or_approved_request_is_rejected(tools_db, start, end):
    before = snapshot(tools_db)
    result = apply(tools_db, "E1", "annual", start, end)
    assert result.status == "error" and "overlap" in result.reason.lower()
    assert snapshot(tools_db) == before


@pytest.mark.parametrize(
    "start, end",
    [("2026-10-30", "2026-11-01"), ("2026-11-05", "2026-11-06"), ("2026-12-17", "2026-12-18"),
     ("2026-12-25", "2026-12-31")],
)
def test_ranges_next_to_an_existing_request_are_not_overlaps(tools_db, start, end):
    assert validate(tools_db, "E1", "annual", start, end).ok


def test_a_weekend_only_overlap_still_counts_as_a_shared_date(tools_db):
    # an approved request over Sat 10-31 and Sun 11-01; the new one (Fri 10-30 to Sat 10-31) shares Saturday
    run_sql(tools_db, "INSERT INTO leave_requests VALUES ('RW','E1','annual','2026-10-31','2026-11-01',0,'approved','2026-10-10T09:00:00')")
    result = apply(tools_db, "E1", "annual", "2026-10-30", "2026-10-31")
    assert result.status == "error" and "overlap" in result.reason.lower()


def test_a_cancelled_request_does_not_block_the_same_dates(tools_db):
    # R3 is a cancelled sick request on 11-10..11-11
    result = apply(tools_db, "E1", "sick", "2026-11-10", "2026-11-11")
    assert result.status == "success"
    assert result.data["working_days"] == 1  # 11-11 is a public holiday


def test_someone_elses_requests_do_not_block_the_requester(tools_db):
    # R1 is E1's pending 11-02..11-04; E2 may take the same days (E2 has R4 on 11-04..11-06)
    assert apply(tools_db, "E2", "annual", "2026-11-02", "2026-11-03").status == "success"


def test_overlap_with_a_pending_request_of_the_same_dates_for_another_type_is_still_an_overlap(tools_db):
    result = apply(tools_db, "E1", "sick", "2026-11-03", "2026-11-03")  # R1 is annual
    assert result.status == "error" and "overlap" in result.reason.lower()


# --- the date and type rules -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end",
    [("2027-01-04", "2027-01-05"), ("2026-12-30", "2027-01-02"), ("2025-12-29", "2025-12-30"),
     ("2025-12-31", "2026-10-20"), ("2028-02-29", "2028-02-29")],
)
def test_both_dates_must_be_in_2026(tools_db, start, end):
    before = snapshot(tools_db)
    result = apply(tools_db, "E1", "annual", start, end)
    assert result.status == "error" and "2026" in result.reason
    assert snapshot(tools_db) == before


def test_the_last_day_of_2026_is_allowed(tools_db):
    assert validate(tools_db, "E1", "annual", "2026-12-31", "2026-12-31").working_days == 1


@pytest.mark.parametrize("start, end", [("2026-10-14", "2026-10-16"), ("2026-01-05", "2026-01-09"),
                                         ("2026-10-14", "2026-10-14")])
def test_a_start_date_in_the_past_is_rejected(tools_db, start, end):
    result = apply(tools_db, "E1", "annual", start, end)
    assert result.status == "error" and "past" in result.reason.lower()


def test_an_end_date_before_the_start_date_is_rejected(tools_db):
    result = apply(tools_db, "E1", "annual", "2026-10-23", "2026-10-19")
    assert result.status == "error" and "end" in result.reason.lower()


def test_a_single_day_request_is_allowed(tools_db):
    assert apply(tools_db, "E1", "annual", "2026-10-20", "2026-10-20").status == "success"


@pytest.mark.parametrize("bad", ["vacation", "Annual", "SICK", "", " annual", "annual ", None, 5, ["annual"], True])
def test_the_leave_type_must_be_annual_or_sick(tools_db, bad):
    before = snapshot(tools_db)
    result = apply(tools_db, "E1", bad, "2026-10-19", "2026-10-19")
    assert result.status == "error" and "annual" in result.reason and "sick" in result.reason
    assert snapshot(tools_db) == before


@pytest.mark.parametrize(
    "bad",
    ["15/10/2026", "2026-13-01", "2026-02-30", "2026-10-1", "26-10-19", "tomorrow", "", " ", None, 20261020,
     "2026-10-19T00:00:00", "2026-10-19 ", "２０２６-10-19", "next friday"],
)
@pytest.mark.parametrize("position", ["start", "end"])
def test_dates_must_be_valid_iso_dates(tools_db, bad, position):
    start, end = ("2026-10-19", bad) if position == "end" else (bad, "2026-10-20")
    before = snapshot(tools_db)
    result = apply(tools_db, "E1", "annual", start, end)
    assert result.status == "error" and result.reason
    assert snapshot(tools_db) == before


def test_every_rejection_has_a_distinct_specific_reason(tools_db):
    reasons = {
        apply(tools_db, "E1", "x", "2026-10-19", "2026-10-19").reason,
        apply(tools_db, "E1", "annual", "nope", "2026-10-19").reason,
        apply(tools_db, "E1", "annual", "2027-01-04", "2027-01-04").reason,
        apply(tools_db, "E1", "annual", "2026-10-14", "2026-10-14").reason,
        apply(tools_db, "E1", "annual", "2026-10-23", "2026-10-19").reason,
        apply(tools_db, "E1", "annual", "2026-10-17", "2026-10-18").reason,
        apply(tools_db, "E1", "sick", "2026-10-19", "2026-10-26").reason,
        apply(tools_db, "E1", "annual", "2026-11-03", "2026-11-03").reason,
        apply(tools_db, "E4", "annual", "2026-10-19", "2026-10-19").reason,
    }
    assert len(reasons) == 9


# --- success -----------------------------------------------------------------------------------------------------------------


def test_success_creates_exactly_one_pending_request_and_subtracts_exactly_the_working_days(tools_db):
    before_rows = request_rows(tools_db)
    before_balance = balance_of(tools_db, "E1", "annual")
    result = apply(tools_db, "E1", "annual", "2026-11-09", "2026-11-13")  # 4 working days (holiday inside)
    assert isinstance(result, ToolResult) and result.name == "apply_leave"
    assert result.status == "success" and result.reason is None

    new_rows = [r for r in request_rows(tools_db) if r not in before_rows]
    assert len(new_rows) == 1 and len(request_rows(tools_db)) == len(before_rows) + 1
    request_id, employee_id, leave_type, start, end, working_days, status, created_at = new_rows[0]
    assert uuid.UUID(request_id).version == 4 and str(uuid.UUID(request_id)) == request_id
    assert (employee_id, leave_type, start, end, working_days, status) == (
        "E1", "annual", "2026-11-09", "2026-11-13", 4, "pending")
    assert isinstance(datetime.fromisoformat(created_at), datetime)
    assert balance_of(tools_db, "E1", "annual") == before_balance - 4

    assert result.data == {
        "request_id": request_id, "leave_type": "annual", "start_date": "2026-11-09",
        "end_date": "2026-11-13", "working_days": 4, "status": "pending", "balance_days": 6.0,
    }
    assert result.args == {"leave_type": "annual", "start_date": "2026-11-09", "end_date": "2026-11-13"}


def test_success_touches_nothing_else(tools_db):
    before = snapshot(tools_db)
    apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-19")
    after = snapshot(tools_db)
    assert after["employees"] == before["employees"]
    other_balances = [r for r in before["leave_balances"] if r[:2] != ("E1", "annual")]
    assert [r for r in after["leave_balances"] if r[:2] != ("E1", "annual")] == other_balances
    assert set(before["leave_requests"]) <= set(after["leave_requests"])


def test_each_request_gets_its_own_uuid4(tools_db):
    ids = [apply(tools_db, "E1", "annual", d, d).data["request_id"]
           for d in ("2026-10-19", "2026-10-20", "2026-10-21")]
    assert len(set(ids)) == 3 and all(uuid.UUID(i).version == 4 for i in ids)


def test_created_at_is_iso_8601_from_the_given_clock_and_the_tool_reads_no_clock(tools_db):
    apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-19", now=datetime(2026, 10, 15, 9, 30, 5))
    created = [r[7] for r in request_rows(tools_db, "E1") if r[3] == "2026-10-19"]
    assert created == ["2026-10-15T09:30:05"]
    apply(tools_db, "E1", "annual", "2026-10-20", "2026-10-20")  # no clock given: derived from `today`
    created = [r[7] for r in request_rows(tools_db, "E1") if r[3] == "2026-10-20"]
    assert created[0].startswith("2026-10-15T")
    datetime.fromisoformat(created[0])


@pytest.mark.parametrize("requester_id", ["E1", "M1", "H1", "E3", "M2"])
def test_every_role_can_apply_for_themselves(tools_db, requester_id):
    result = apply(tools_db, requester_id, "annual", "2026-10-20", "2026-10-20")
    assert result.status == "success"
    assert [r[1] for r in request_rows(tools_db) if r[0] == result.data["request_id"]] == [requester_id]


def test_the_request_is_always_for_the_requester_and_the_signature_has_no_employee_argument(tools_db):
    params = [n for n, p in inspect.signature(apply_leave).parameters.items()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert params == ["requester", "leave_type", "start_date", "end_date"]
    for banned in ("employee_id", "target", "on_behalf_of"):
        with pytest.raises(TypeError):
            apply_leave(who(tools_db, "M1"), "annual", "2026-10-20", "2026-10-20", db=tools_db, today=TODAY,
                        **{banned: "E1"})  # type: ignore[arg-type]


def test_two_sequential_applies_cannot_overspend(tools_db):
    first = apply(tools_db, "E1", "sick", "2026-10-19", "2026-10-21")  # 3 of 5
    second = apply(tools_db, "E1", "sick", "2026-10-22", "2026-10-26")  # Thu Fri Mon = 3 > 2 left
    assert first.status == "success"
    assert second.status == "error" and "balance" in second.reason.lower()
    assert balance_of(tools_db, "E1", "sick") == 2.0
    third = apply(tools_db, "E1", "sick", "2026-10-22", "2026-10-23")  # 2 == 2
    assert third.status == "success" and balance_of(tools_db, "E1", "sick") == 0.0
    assert apply(tools_db, "E1", "sick", "2026-10-27", "2026-10-27").status == "error"  # nothing left


def test_applying_twice_for_the_same_dates_is_an_overlap_the_second_time(tools_db):
    assert apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-21").status == "success"
    again = apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-21")
    assert again.status == "error" and "overlap" in again.reason.lower()
    assert balance_of(tools_db, "E1", "annual") == 7.0


def test_two_concurrent_applies_cannot_both_spend_the_same_balance(tools_db):
    barrier = threading.Barrier(2)

    def attempt(dates):
        barrier.wait()
        return apply(tools_db, "E1", "sick", *dates)

    # 4 days each, on different weeks (no overlap), against a balance of 5
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, [("2026-10-19", "2026-10-22"), ("2026-10-26", "2026-10-29")]))
    assert sorted(r.status for r in results) == ["error", "success"]
    assert balance_of(tools_db, "E1", "sick") == 1.0
    assert len([r for r in request_rows(tools_db, "E1") if r[2] == "sick" and r[6] == "pending"]) == 1


def test_two_concurrent_identical_applies_create_one_request(tools_db):
    barrier = threading.Barrier(2)

    def attempt(_):
        barrier.wait()
        return apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-20")

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sorted(r.status for r in results) == ["error", "success"]
    assert balance_of(tools_db, "E1", "annual") == 8.0


# --- the checks are repeated inside the transaction ---------------------------------------------------------------------


class RacyDatabase(Database):
    """Runs ``hook`` right after the first read connection closes: a competing writer that
    gets in between the pre-check and the transaction."""

    def __init__(self, path, hook):
        super().__init__(path)
        self.hook = hook
        self.reads = 0

    @contextmanager
    def read(self):
        with super().read() as conn:
            yield conn
        self.reads += 1
        if self.reads == 1:
            self.hook()


def test_the_balance_is_checked_again_inside_the_transaction(tools_db):
    racy = RacyDatabase(
        tools_db.path,
        lambda: run_sql(tools_db, "UPDATE leave_balances SET balance_days = 1 WHERE employee_id = 'E1' AND leave_type = 'annual'"),
    )
    before = request_rows(tools_db)
    result = apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-21", db=racy, today=TODAY)
    assert result.status == "error" and "balance" in result.reason.lower()
    assert request_rows(tools_db) == before
    assert balance_of(tools_db, "E1", "annual") == 1.0


def test_overlap_is_checked_again_inside_the_transaction(tools_db):
    racy = RacyDatabase(
        tools_db.path,
        lambda: run_sql(tools_db, "INSERT INTO leave_requests VALUES ('RACE','E1','annual','2026-10-20','2026-10-20',1,'pending','2026-10-15T08:00:00')"),
    )
    result = apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-21", db=racy, today=TODAY)
    assert result.status == "error" and "overlap" in result.reason.lower()
    assert [r[0] for r in request_rows(tools_db, "E1") if r[0] == "RACE"] == ["RACE"]
    assert len(request_rows(tools_db, "E1")) == 4  # R1, R2, R3 and the competing one; nothing of ours
    assert balance_of(tools_db, "E1", "annual") == 10.0


# --- rollback ----------------------------------------------------------------------------------------------------------------------

BALANCE_TRIGGER = ("CREATE TRIGGER fail_balance BEFORE UPDATE ON leave_balances "
                   "BEGIN SELECT RAISE(ABORT, 'forced failure'); END")
INSERT_TRIGGER = ("CREATE TRIGGER fail_insert BEFORE INSERT ON leave_requests "
                  "BEGIN SELECT RAISE(ABORT, 'forced failure'); END")


@pytest.mark.parametrize("trigger", [BALANCE_TRIGGER, INSERT_TRIGGER], ids=["fails-after-insert", "fails-on-insert"])
def test_a_database_failure_rolls_everything_back(tools_db, trigger):
    run_sql(tools_db, trigger)
    before = snapshot(tools_db)
    result = apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-21")
    assert result.status == "error" and result.data == {}
    assert "forced" not in result.reason  # no database detail reaches the requester
    assert "nothing was changed" in result.reason.lower()
    assert snapshot(tools_db) == before


def test_an_unexpected_exception_after_the_insert_rolls_back_and_propagates(tools_db, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(leave_apply, "_debit_balance", boom)
    before = snapshot(tools_db)
    with pytest.raises(RuntimeError, match="boom"):
        apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-21")
    assert snapshot(tools_db) == before


def test_the_write_lock_is_released_after_a_failure(tools_db):
    run_sql(tools_db, BALANCE_TRIGGER)
    apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-21")
    run_sql(tools_db, "DROP TRIGGER fail_balance")
    assert apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-21").status == "success"


def test_the_write_uses_begin_immediate(tools_db, sql_log):
    sql_log.clear()
    apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-19")
    begins = [s.upper() for s in sql_log if s.upper().startswith("BEGIN")]
    assert begins == ["BEGIN IMMEDIATE"]
    assert any(s.upper().startswith("COMMIT") for s in sql_log)


# --- validate_apply_leave --------------------------------------------------------------------------------------------------------


def test_validate_returns_the_working_days_and_writes_nothing(tools_db):
    before = snapshot(tools_db)
    result = validate(tools_db, "E1", "annual", "2026-11-09", "2026-11-13")
    assert isinstance(result, LeaveValidation)
    assert result.ok and result.working_days == 4 and result.reason is None
    assert snapshot(tools_db) == before


REJECTIONS = [
    ("annual", "2026-10-17", "2026-10-18"),
    ("sick", "2026-10-19", "2026-10-26"),
    ("annual", "2026-11-03", "2026-11-03"),
    ("annual", "2027-01-04", "2027-01-04"),
    ("annual", "2026-10-14", "2026-10-14"),
    ("annual", "2026-10-23", "2026-10-19"),
    ("bogus", "2026-10-19", "2026-10-19"),
    ("annual", "19/10/2026", "2026-10-19"),
]


@pytest.mark.parametrize("leave_type, start, end", REJECTIONS)
def test_validate_gives_the_same_reason_as_apply_and_writes_nothing(tools_db, leave_type, start, end):
    before = snapshot(tools_db)
    checked = validate(tools_db, "E1", leave_type, start, end)
    assert snapshot(tools_db) == before
    assert not checked.ok and checked.working_days is None and checked.reason
    applied = apply(tools_db, "E1", leave_type, start, end)
    assert applied.status == "error" and applied.reason == checked.reason


def test_validate_has_the_calendar_rejections_too(tools_db):
    checked = validate(tools_db, "E4", "annual", "2026-10-19", "2026-10-23")
    applied = apply(tools_db, "E4", "annual", "2026-10-19", "2026-10-23")
    assert not checked.ok and checked.reason == applied.reason


def test_validate_and_apply_agree_on_working_days(tools_db):
    run_sql(tools_db, "UPDATE leave_balances SET balance_days = 40 WHERE employee_id = 'E1' AND leave_type = 'annual'")
    for start, end in [("2026-10-19", "2026-10-23"), ("2026-11-09", "2026-11-13"), ("2026-12-28", "2026-12-31")]:
        checked = validate(tools_db, "E1", "annual", start, end)
        applied = apply(tools_db, "E1", "annual", start, end)
        assert checked.ok and applied.status == "success"
        assert checked.working_days == applied.data["working_days"]


def test_validate_runs_on_a_read_only_connection(tools_db, sql_log):
    sql_log.clear()
    validate(tools_db, "E1", "annual", "2026-10-19", "2026-10-23")
    assert not [s for s in sql_log if re.match(r"\s*(INSERT|UPDATE|DELETE|BEGIN|COMMIT)", s, re.I)]


def test_validate_never_takes_a_reference_to_other_employees(tools_db):
    names = [n for n in inspect.signature(validate_apply_leave).parameters
             if n not in ("db", "today")]
    assert names == ["requester", "leave_type", "start_date", "end_date"]


# --- identity, today and personal data -------------------------------------------------------------------------------------------


def test_today_is_a_parameter_not_the_clock(tools_db):
    later = apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-19", db=tools_db,
                        today=datetime(2026, 10, 20).date())
    assert later.status == "error" and "past" in later.reason.lower()  # 10-19 is before the given today
    with pytest.raises(TypeError):
        apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-19", db=tools_db, today="2026-10-15")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-19", db=tools_db)  # type: ignore[call-arg]


def test_the_holiday_lookup_uses_today_as_its_reference_date(tools_db):
    run_sql(tools_db, "UPDATE holiday_calendars SET effective_date = '2026-10-16' WHERE doc_id = 'CAL-IN'")
    assert apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-19").status == "error"  # not yet in force on 10-15
    later = apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-19", db=tools_db,
                        today=datetime(2026, 10, 16).date())
    assert later.status == "success"


def test_no_name_email_or_phone_in_results_or_args_for_success_or_error(tools_db):
    names, emails, phones = pii_values(tools_db)
    results = [
        apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-19"),
        apply(tools_db, "E1", "annual", "2026-10-19", "2026-10-19"),  # overlap now
        apply(tools_db, "E1", "Priya Raman", "ravi.kumar@veltrix.example", "+91 98221 51875"),
        apply(tools_db, "E4", "annual", "2026-10-19", "2026-10-23"),
    ]
    for result in results:
        blob = json.dumps(dataclasses.asdict(result))
        for value in (*names, *emails, *phones, "Priya", "ravi.kumar", "98221"):
            assert value not in blob, value
    assert set(results[2].args) <= {"leave_type", "start_date", "end_date"}
    assert set(results[2].args.values()) == {"[invalid]"}
