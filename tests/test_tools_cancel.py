"""cancel_leave and validate_cancel_leave (README B6): own requests only, one transaction."""
from __future__ import annotations

import dataclasses
import inspect
import json
import logging
from datetime import date

import pytest

import app.tools.leave_cancel as leave_cancel
from app.tools import ToolResult, cancel_leave, validate_cancel_leave
from app.tools.leave_cancel import CancelValidation
from tests.db_helpers import dump_tables
from tests.tool_helpers import CORE_TABLES, TODAY, balance_of, pii_values, request_rows, run_sql, who

# E1: R1 annual pending 11-02..11-04 (3 days), R2 annual approved 12-21..12-24 (4 days),
# R3 sick cancelled 11-10..11-11 (2 days). Balances: annual 10, sick 5.
# Extra rows used here are inserted by `add_requests`.


def cancel(db, requester_id, request_id, **kw):
    return cancel_leave(who(db, requester_id), request_id, db=db, today=TODAY, **kw)


def validate(db, requester_id, request_id):
    return validate_cancel_leave(who(db, requester_id), request_id, db=db, today=TODAY)


def snapshot(db):
    return dump_tables(db.path, CORE_TABLES)


def add_requests(db):
    run_sql(
        db,
        "INSERT INTO leave_requests VALUES ('R-TODAY','E1','annual','2026-10-15','2026-10-16',2,'approved','2026-10-01T09:00:00')",
        "INSERT INTO leave_requests VALUES ('R-STARTED','E1','annual','2026-10-12','2026-10-20',7,'approved','2026-10-01T09:00:00')",
        "INSERT INTO leave_requests VALUES ('R-PAST','E1','sick','2026-09-01','2026-09-02',2,'approved','2026-08-20T09:00:00')",
        "INSERT INTO leave_requests VALUES ('R-TOMORROW','E1','sick','2026-10-16','2026-10-16',1,'pending','2026-10-14T09:00:00')",
    )


# --- success ---------------------------------------------------------------------------------------------------


def test_a_pending_request_is_cancelled_and_its_days_come_back_exactly(tools_db):
    before_balance = balance_of(tools_db, "E1", "annual")
    result = cancel(tools_db, "E1", "R1")
    assert isinstance(result, ToolResult) and result.name == "cancel_leave"
    assert result.status == "success" and result.reason is None
    assert balance_of(tools_db, "E1", "annual") == before_balance + 3
    assert [r[6] for r in request_rows(tools_db, "E1") if r[0] == "R1"] == ["cancelled"]
    assert result.data == {
        "request_id": "R1", "status": "cancelled", "leave_type": "annual", "start_date": "2026-11-02",
        "end_date": "2026-11-04", "working_days": 3, "balance_days": 13.0,
    }
    assert result.args == {"request_id": "R1"}


def test_an_approved_request_can_be_cancelled_too(tools_db):
    result = cancel(tools_db, "E1", "R2")
    assert result.status == "success"
    assert balance_of(tools_db, "E1", "annual") == 14.0  # 10 + 4
    assert [r[6] for r in request_rows(tools_db, "E1") if r[0] == "R2"] == ["cancelled"]


def test_only_the_requests_leave_type_balance_changes(tools_db):
    cancel(tools_db, "E1", "R1")
    assert balance_of(tools_db, "E1", "sick") == 5.0
    assert balance_of(tools_db, "E2", "annual") == 3.5


def test_cancel_changes_nothing_else(tools_db):
    before = snapshot(tools_db)
    cancel(tools_db, "E1", "R1")
    after = snapshot(tools_db)
    assert after["employees"] == before["employees"]
    assert [r for r in after["leave_requests"] if r[0] != "R1"] == [r for r in before["leave_requests"] if r[0] != "R1"]
    assert [r for r in after["leave_balances"] if r[:2] != ("E1", "annual")] == [
        r for r in before["leave_balances"] if r[:2] != ("E1", "annual")]


def test_a_request_starting_tomorrow_can_be_cancelled(tools_db):
    add_requests(tools_db)
    assert cancel(tools_db, "E1", "R-TOMORROW").status == "success"
    assert balance_of(tools_db, "E1", "sick") == 6.0


def test_the_days_restored_can_be_spent_again(tools_db):
    from app.tools import apply_leave

    assert apply_leave(who(tools_db, "E1"), "sick", "2026-10-19", "2026-10-23", db=tools_db, today=TODAY).status == "success"
    assert balance_of(tools_db, "E1", "sick") == 0.0
    request_id = [r[0] for r in request_rows(tools_db, "E1") if r[3] == "2026-10-19"][0]
    assert cancel(tools_db, "E1", request_id).status == "success"
    assert balance_of(tools_db, "E1", "sick") == 5.0
    assert apply_leave(who(tools_db, "E1"), "sick", "2026-10-19", "2026-10-23", db=tools_db, today=TODAY).status == "success"


def test_a_fractional_balance_gets_the_whole_days_added(tools_db):
    run_sql(tools_db, "UPDATE leave_balances SET balance_days = 0.5 WHERE employee_id = 'E1' AND leave_type = 'annual'")
    cancel(tools_db, "E1", "R1")
    assert balance_of(tools_db, "E1", "annual") == 3.5


@pytest.mark.parametrize("requester_id, request_id", [("E1", "R1"), ("M1", "R6"), ("H1", None), ("E3", "R5"), ("M3", "R7")])
def test_every_role_can_cancel_their_own_requests(tools_db, requester_id, request_id):
    if request_id is None:  # hr has no request in the data: make one
        run_sql(tools_db, "INSERT INTO leave_requests VALUES ('RH','H1','annual','2026-11-20','2026-11-20',1,'pending','2026-10-10T09:00:00')")
        request_id = "RH"
    assert cancel(tools_db, requester_id, request_id).status == "success"


# --- what cannot be cancelled -----------------------------------------------------------------------------------------


def test_a_cancelled_request_cannot_be_cancelled_again_and_days_are_not_added_twice(tools_db):
    assert cancel(tools_db, "E1", "R1").status == "success"
    again = cancel(tools_db, "E1", "R1")
    assert again.status == "error" and "cancelled" in again.reason.lower()
    assert balance_of(tools_db, "E1", "annual") == 13.0


def test_an_already_cancelled_request_in_the_data_is_refused(tools_db):
    before = snapshot(tools_db)
    result = cancel(tools_db, "E1", "R3")
    assert result.status == "error" and "cancelled" in result.reason.lower()
    assert snapshot(tools_db) == before


@pytest.mark.parametrize("request_id", ["R-TODAY", "R-STARTED", "R-PAST"])
def test_a_request_that_has_started_or_passed_cannot_be_cancelled(tools_db, request_id):
    add_requests(tools_db)
    before = snapshot(tools_db)
    result = cancel(tools_db, "E1", request_id)
    assert result.status == "error"
    assert "start" in result.reason.lower()
    assert snapshot(tools_db) == before


def test_start_date_must_be_after_today_not_equal(tools_db):
    add_requests(tools_db)
    assert cancel(tools_db, "E1", "R-TODAY").status == "error"  # starts today
    later = cancel_leave(who(tools_db, "E1"), "R-TODAY", db=tools_db, today=date(2026, 10, 14))
    assert later.status == "success"  # the day before: it starts after "today"


def test_a_missing_balance_row_means_the_request_stays_as_it_is(tools_db):
    run_sql(tools_db, "DELETE FROM leave_balances WHERE employee_id = 'E1' AND leave_type = 'annual'")
    before = snapshot(tools_db)
    result = cancel(tools_db, "E1", "R1")
    assert result.status == "error" and "balance" in result.reason.lower()
    assert snapshot(tools_db) == before


# --- someone else's request and a nonexistent one look the same --------------------------------------------------------------


NOT_OWN = [("E1", "R4"), ("E1", "R5"), ("M1", "R1"), ("M1", "R4"), ("H1", "R1"), ("H1", "R6"), ("E2", "R1"), ("M3", "R8")]
GHOSTS = ["NOPE", "R999", "r1", "00000000-0000-4000-8000-000000000000", "R1x"]


@pytest.mark.parametrize("requester_id, request_id", NOT_OWN)
@pytest.mark.parametrize("ghost", GHOSTS)
def test_someone_elses_request_and_a_nonexistent_request_get_the_same_error(tools_db, requester_id, request_id, ghost):
    before = snapshot(tools_db)
    other = cancel(tools_db, requester_id, request_id)
    missing = cancel(tools_db, requester_id, ghost)
    assert other.status == "error"
    assert dataclasses.replace(other, args={}) == dataclasses.replace(missing, args={})
    assert repr(dataclasses.replace(other, args={})) == repr(dataclasses.replace(missing, args={}))
    assert other.reason == missing.reason
    assert snapshot(tools_db) == before  # neither touched anything


@pytest.mark.parametrize("requester_id, request_id", NOT_OWN)
def test_the_error_does_not_say_whose_request_it_is_or_whether_it_exists(tools_db, requester_id, request_id):
    result = cancel(tools_db, requester_id, request_id)
    text = json.dumps(dataclasses.asdict(dataclasses.replace(result, args={}))).lower()
    for word in ("belongs", "another", "someone", "other employee", "not yours", "exist", "permission", "allowed"):
        assert word not in text, word


def test_the_same_sql_runs_for_someone_elses_and_a_nonexistent_request(tools_db, sql_log):
    sql_log.clear()
    cancel(tools_db, "E1", "R4")
    first = [s.replace("'R4'", "'<ID>'") for s in sql_log]
    sql_log.clear()
    cancel(tools_db, "E1", "ZZZ")
    second = [s.replace("'ZZZ'", "'<ID>'") for s in sql_log]
    assert first == second


def test_ownership_is_part_of_the_lookup_itself(tools_db, sql_log):
    sql_log.clear()
    cancel(tools_db, "E1", "R4")
    lookups = [s for s in sql_log if "'R4'" in s]
    assert lookups and all("employee_id" in s and "'E1'" in s for s in lookups)


@pytest.mark.parametrize("bad", ["", " ", None, 5, ["R1"], {"id": "R1"}, "R1; DROP TABLE leave_requests", "a" * 300, True])
def test_a_malformed_request_id_gets_the_same_not_found_error(tools_db, bad):
    reference = cancel(tools_db, "E1", "R4")
    result = cancel(tools_db, "E1", bad)
    assert dataclasses.replace(result, args={}) == dataclasses.replace(reference, args={})
    assert len(request_rows(tools_db)) == 8


def test_the_args_never_echo_something_that_is_not_an_id(tools_db):
    result = cancel(tools_db, "E1", "Priya Raman cancel my leave")
    assert result.args == {"request_id": "[invalid]"}


# --- rollback -------------------------------------------------------------------------------------------------------------------------

BALANCE_TRIGGER = ("CREATE TRIGGER fail_balance BEFORE UPDATE ON leave_balances "
                   "BEGIN SELECT RAISE(ABORT, 'forced failure'); END")
STATUS_TRIGGER = ("CREATE TRIGGER fail_status BEFORE UPDATE ON leave_requests "
                  "BEGIN SELECT RAISE(ABORT, 'forced failure'); END")


@pytest.mark.parametrize("trigger", [BALANCE_TRIGGER, STATUS_TRIGGER], ids=["fails-on-balance", "fails-on-status"])
def test_a_database_failure_rolls_everything_back(tools_db, trigger):
    run_sql(tools_db, trigger)
    before = snapshot(tools_db)
    result = cancel(tools_db, "E1", "R1")
    assert result.status == "error" and result.data == {}
    assert "forced" not in result.reason and "nothing was changed" in result.reason.lower()
    assert snapshot(tools_db) == before


def test_an_unexpected_exception_after_the_status_change_rolls_back(tools_db, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(leave_cancel, "_credit_balance", boom)
    before = snapshot(tools_db)
    with pytest.raises(RuntimeError, match="boom"):
        cancel(tools_db, "E1", "R1")
    assert snapshot(tools_db) == before


def test_the_write_uses_begin_immediate(tools_db, sql_log):
    sql_log.clear()
    cancel(tools_db, "E1", "R1")
    assert [s.upper() for s in sql_log if s.upper().startswith("BEGIN")] == ["BEGIN IMMEDIATE"]


def test_the_status_is_checked_again_inside_the_transaction(tools_db):
    """A competing cancel between the pre-check and the transaction must not add days twice."""
    from app.db import Database
    from contextlib import contextmanager

    class Racy(Database):
        reads = 0

        @contextmanager
        def read(self):
            with super().read() as conn:
                yield conn
            Racy.reads += 1
            if Racy.reads == 1:
                run_sql(tools_db, "UPDATE leave_requests SET status = 'cancelled' WHERE request_id = 'R1'")

    result = cancel_leave(who(tools_db, "E1"), "R1", db=Racy(tools_db.path), today=TODAY)
    assert result.status == "error"
    assert balance_of(tools_db, "E1", "annual") == 10.0  # not 13


def test_two_concurrent_cancels_add_the_days_once(tools_db):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2)

    def attempt(_):
        barrier.wait()
        return cancel(tools_db, "E1", "R1")

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sorted(r.status for r in results) == ["error", "success"]
    assert balance_of(tools_db, "E1", "annual") == 13.0


# --- validate_cancel_leave -----------------------------------------------------------------------------------------------------------------


def test_validate_returns_the_request_summary_and_writes_nothing(tools_db):
    before = snapshot(tools_db)
    result = validate(tools_db, "E1", "R1")
    assert isinstance(result, CancelValidation)
    assert result.ok and result.reason is None
    assert (result.request_id, result.leave_type, result.start_date, result.end_date, result.working_days) == (
        "R1", "annual", "2026-11-02", "2026-11-04", 3)
    assert snapshot(tools_db) == before


@pytest.mark.parametrize("request_id", ["R3", "R-TODAY", "R-STARTED", "R-PAST", "R4", "ZZZ", ""])
def test_validate_gives_the_same_reason_as_cancel_and_writes_nothing(tools_db, request_id):
    add_requests(tools_db)
    before = snapshot(tools_db)
    checked = validate(tools_db, "E1", request_id)
    assert snapshot(tools_db) == before
    assert not checked.ok and checked.reason and checked.working_days is None
    applied = cancel(tools_db, "E1", request_id)
    assert applied.status == "error" and applied.reason == checked.reason


def test_validate_for_someone_elses_request_matches_a_nonexistent_one(tools_db):
    assert validate(tools_db, "E1", "R4") == validate(tools_db, "E1", "ZZZ")


def test_validate_runs_no_write_statements(tools_db, sql_log):
    import re

    sql_log.clear()
    validate(tools_db, "E1", "R1")
    assert not [s for s in sql_log if re.match(r"\s*(INSERT|UPDATE|DELETE|BEGIN|COMMIT)", s, re.I)]


# --- signature, today, personal data -----------------------------------------------------------------------------------------------------------


def test_the_only_argument_is_the_request_id(tools_db):
    positional = [n for n, p in inspect.signature(cancel_leave).parameters.items()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert positional == ["requester", "request_id"]
    with pytest.raises(TypeError):
        cancel_leave(who(tools_db, "H1"), "R1", employee_id="E1", db=tools_db, today=TODAY)  # type: ignore[call-arg]


def test_today_is_required_and_must_be_a_date(tools_db):
    with pytest.raises(TypeError):
        cancel_leave(who(tools_db, "E1"), "R1", db=tools_db)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        cancel_leave(who(tools_db, "E1"), "R1", db=tools_db, today="2026-10-15")  # type: ignore[arg-type]


def test_no_name_email_or_phone_in_results_or_args(tools_db):
    names, emails, phones = pii_values(tools_db)
    add_requests(tools_db)
    results = [cancel(tools_db, "E1", rid) for rid in ("R1", "R1", "R3", "R-TODAY", "R4", "ZZ", "Priya Raman")]
    for result in results:
        blob = json.dumps(dataclasses.asdict(result))
        for value in (*names, *emails, *phones, "Priya", "@"):
            assert value not in blob, value


def test_logging_carries_no_personal_data(tools_db, caplog):
    names, emails, phones = pii_values(tools_db)
    with caplog.at_level(logging.DEBUG):
        cancel(tools_db, "E1", "R1")
        cancel(tools_db, "E1", "R4")
    for value in (*names, *emails, *phones):
        assert value not in caplog.text
