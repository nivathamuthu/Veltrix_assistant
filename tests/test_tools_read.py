"""get_leave_balance, list_leave_requests, get_team_leave: who may see what (README B3, B6)."""
from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

import app.tools.leave_balance as balance_module
import app.tools.leave_requests as requests_module
from app.tools import (
    REFUSAL_MESSAGE,
    ToolResult,
    get_leave_balance,
    get_team_leave,
    list_leave_requests,
)
from tests.tool_helpers import TODAY, pii_values, who

TOOLS = {"get_leave_balance": get_leave_balance, "list_leave_requests": list_leave_requests}


def call(tool, db, requester_id, employee_id=None, **kw):
    return tool(who(db, requester_id), employee_id, db=db, **kw)


# (requester, target) pairs. Every one of these targets has balance rows and requests data.
ALLOWED = [
    ("E1", None), ("E1", "E1"), ("E3", "E3"), ("E4", "E4"),
    ("M1", "E1"), ("M1", "E2"), ("M1", "M3"), ("M1", "M1"), ("M1", None),
    ("M2", "E3"), ("M3", "E5"), ("M3", "M3"),
    ("H1", "E1"), ("H1", "E3"), ("H1", "E4"), ("H1", "M2"), ("H1", "M3"), ("H1", "H1"), ("H1", "E5"),
]
DENIED = [
    ("E1", "E2"),  # a colleague in the same team
    ("E1", "M1"),  # their own manager
    ("E1", "H1"), ("E3", "E1"), ("E4", "E1"), ("E5", "M3"),
    ("M1", "E3"),  # someone else's report
    ("M1", "E5"),  # a report's report is not a direct report
    ("M1", "M2"), ("M1", "H1"), ("M1", "E4"),
    ("M3", "M1"),  # their own manager
    ("M3", "E1"), ("M2", "E1"), ("M2", "M1"),
]
NONEXISTENT = ["ZZ99", "E999", "e1", "E1x", "M0"]


def target_of(requester_id, employee_id):
    return requester_id if employee_id is None else employee_id


def without_args(result: ToolResult) -> ToolResult:
    """Everything but the echo of the caller's own arguments."""
    return dataclasses.replace(result, args={})


# --- who may see what: every role against both per-employee tools ----------------------------------


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("requester_id, employee_id", ALLOWED)
def test_allowed_combinations_return_the_targets_data(tools_db, name, requester_id, employee_id):
    result = call(TOOLS[name], tools_db, requester_id, employee_id)
    assert isinstance(result, ToolResult) and result.name == name
    assert result.status == "success", result
    assert result.reason is None
    assert result.data["employee_id"] == target_of(requester_id, employee_id)


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("requester_id, employee_id", DENIED)
def test_denied_combinations_get_the_fixed_refusal_and_no_data(tools_db, name, requester_id, employee_id):
    result = call(TOOLS[name], tools_db, requester_id, employee_id)
    assert result.status == "error"
    assert result.reason == REFUSAL_MESSAGE
    assert result.data == {}


def test_the_refusal_message_is_the_agreed_text():
    assert REFUSAL_MESSAGE == "I can't share information about that employee."


@pytest.mark.parametrize("name", TOOLS)
def test_an_employee_may_see_only_themselves_manager_sees_reports_only_hr_sees_any_one(tools_db, name):
    tool = TOOLS[name]
    for requester_id in ("E1", "M1", "H1"):
        results = {target: call(tool, tools_db, requester_id, target).status
                   for target in ("E1", "E2", "E3", "M1", "M2", "M3", "H1", "E4", "E5")}
        visible = {t for t, s in results.items() if s == "success"}
        if requester_id == "E1":
            assert visible == {"E1"}
        elif requester_id == "M1":
            assert visible == {"M1", "E1", "E2", "M3"}
        else:
            assert visible == set(results)  # HR: any one at a time


def test_an_unknown_role_sees_only_themselves(tools_db):
    odd = dataclasses.replace(who(tools_db, "M1"), role="admin")
    assert get_leave_balance(odd, "E1", db=tools_db).reason == REFUSAL_MESSAGE
    assert get_leave_balance(odd, None, db=tools_db).status == "success"
    assert get_team_leave(odd, "2026-11-01", "2026-11-30", db=tools_db).status == "error"


# --- bulk requests are refused for every role -----------------------------------------------------------

BULK_IDS = ["*", "all", "ALL", "everyone", "E1,E2", "E1 E2", "E1;E2", "%", "", " ", "   ", ["E1", "E2"],
            ("E1", "E2"), {"E1": 1}, 5, 1.5, True, "E1\nE2", "E1' OR '1'='1"]


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("requester_id", ["E1", "M1", "H1"])
@pytest.mark.parametrize("bulk", BULK_IDS, ids=repr)
def test_bulk_requests_never_return_data_for_any_role(tools_db, name, requester_id, bulk):
    result = call(TOOLS[name], tools_db, requester_id, bulk)
    assert result.status == "error"
    assert result.data == {}


@pytest.mark.parametrize("name", TOOLS)
def test_hr_gets_one_employee_per_call_and_nothing_in_bulk(tools_db, name):
    one = call(TOOLS[name], tools_db, "H1", "E1")
    assert one.status == "success" and one.data["employee_id"] == "E1"
    blob = json.dumps(one.data)
    for other in ("E2", "E3", "M1", "M2"):
        assert f'"{other}"' not in blob


def test_an_empty_employee_id_is_not_treated_as_self(tools_db):
    for requester_id in ("E1", "H1"):
        result = call(get_leave_balance, tools_db, requester_id, "")
        assert result.status == "error" and result.data == {}


# --- the same answer for a forbidden employee and one who does not exist -----------------------------------


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("requester_id, forbidden", DENIED)
@pytest.mark.parametrize("ghost", NONEXISTENT)
def test_a_nonexistent_id_and_a_forbidden_existing_id_get_identical_results(
    tools_db, name, requester_id, forbidden, ghost
):
    real = call(TOOLS[name], tools_db, requester_id, forbidden)
    missing = call(TOOLS[name], tools_db, requester_id, ghost)
    assert real.reason == REFUSAL_MESSAGE
    assert without_args(real) == without_args(missing)
    assert repr(without_args(real)) == repr(without_args(missing))
    assert json.dumps(dataclasses.asdict(without_args(real)), sort_keys=True) == json.dumps(
        dataclasses.asdict(without_args(missing)), sort_keys=True
    )


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("requester_id, forbidden", [("E1", "E2"), ("M1", "E3"), ("M3", "M1")])
def test_the_same_sql_runs_for_a_nonexistent_and_a_forbidden_id(tools_db, sql_log, name, requester_id, forbidden):
    """The decision is the same code path: only the id inside the statements differs."""
    sql_log.clear()
    call(TOOLS[name], tools_db, requester_id, forbidden)
    for_forbidden = [s.replace(f"'{forbidden}'", "'<ID>'") for s in sql_log]
    sql_log.clear()
    call(TOOLS[name], tools_db, requester_id, "ZZ99")
    for_missing = [s.replace("'ZZ99'", "'<ID>'") for s in sql_log]
    assert for_forbidden == for_missing


# --- the permission check runs before the target is looked up ----------------------------------------------


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("requester_id, forbidden", [("M1", "E3"), ("M1", "E5"), ("M3", "M1"), ("M1", "ZZ99")])
def test_for_a_forbidden_manager_target_no_row_of_it_is_read(tools_db, sql_log, name, requester_id, forbidden):
    sql_log.clear()
    result = call(TOOLS[name], tools_db, requester_id, forbidden)
    assert result.reason == REFUSAL_MESSAGE
    touching = [s for s in sql_log if f"'{forbidden}'" in s]
    # the only statement that may mention the target is the relationship check on the
    # requester's own id: it returns no columns of the target, only whether the link exists
    assert touching, "the check must run"
    for statement in touching:
        assert "leave_balances" not in statement and "leave_requests" not in statement
        assert "manager_id" in statement and f"'{requester_id}'" in statement
        assert statement.strip().upper().startswith("SELECT 1 ")


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize("requester_id, forbidden", [("E1", "E2"), ("E1", "ZZ99"), ("E4", "E1")])
def test_for_an_employee_the_database_is_not_touched_for_another_id(tools_db, sql_log, name, requester_id, forbidden):
    sql_log.clear()
    call(TOOLS[name], tools_db, requester_id, forbidden)
    assert not [s for s in sql_log if f"'{forbidden}'" in s]


@pytest.mark.parametrize(
    "module, reader", [(balance_module, "get_balance"), (requests_module, "list_requests")]
)
def test_the_data_readers_are_never_called_for_a_forbidden_target(tools_db, monkeypatch, module, reader):
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[1:])
        raise AssertionError("target data was read before the permission check passed")

    monkeypatch.setattr(module, reader, spy)
    tool = get_leave_balance if module is balance_module else list_leave_requests
    for requester_id, target in [("E1", "E2"), ("M1", "E3"), ("M1", "ZZ99"), ("E3", "E1")]:
        result = call(tool, tools_db, requester_id, target)
        assert result.reason == REFUSAL_MESSAGE
    assert calls == []


def test_a_forbidden_request_with_a_bad_filter_still_gets_only_the_refusal(tools_db):
    assert call(get_leave_balance, tools_db, "E1", "E2", leave_type="nonsense").reason == REFUSAL_MESSAGE
    assert call(list_leave_requests, tools_db, "E1", "E2", status="nonsense").reason == REFUSAL_MESSAGE


# --- get_leave_balance ----------------------------------------------------------------------------------------------


def test_balance_is_returned_per_leave_type(tools_db):
    result = call(get_leave_balance, tools_db, "E1")
    assert result.status == "success"
    assert result.data == {
        "employee_id": "E1",
        "balances": [
            {"leave_type": "annual", "balance_days": 10.0},
            {"leave_type": "sick", "balance_days": 5.0},
        ],
    }


def test_balance_can_be_limited_to_one_leave_type(tools_db):
    result = call(get_leave_balance, tools_db, "E1", leave_type="sick")
    assert result.data["balances"] == [{"leave_type": "sick", "balance_days": 5.0}]
    assert call(get_leave_balance, tools_db, "E2", leave_type="annual").data["balances"] == [
        {"leave_type": "annual", "balance_days": 3.5}
    ]


@pytest.mark.parametrize("bad", ["vacation", "Annual", "ANNUAL", "", " annual", 5, ["annual"], True])
def test_an_unknown_leave_type_is_an_error_with_a_clear_reason(tools_db, bad):
    result = call(get_leave_balance, tools_db, "E1", leave_type=bad)
    assert result.status == "error" and result.data == {}
    assert "annual" in result.reason and "sick" in result.reason


def test_a_missing_balance_is_an_error_without_data(tools_db):
    result = call(get_leave_balance, tools_db, "N1")  # an employee with no balance rows
    assert result.status == "error" and result.data == {}
    assert result.reason and result.reason != REFUSAL_MESSAGE
    assert call(get_leave_balance, tools_db, "E2", leave_type="sick").status == "error"  # no sick row


def test_hr_asking_for_an_unknown_employee_gets_an_error_without_data(tools_db):
    result = call(get_leave_balance, tools_db, "H1", "ZZ99")
    assert result.status == "error" and result.data == {}


# --- list_leave_requests ----------------------------------------------------------------------------------------------


def test_requests_are_listed_oldest_start_first_with_the_stored_values(tools_db):
    result = call(list_leave_requests, tools_db, "E1")
    assert result.status == "success"
    assert result.data["employee_id"] == "E1"
    rows = result.data["requests"]
    assert [r["request_id"] for r in rows] == ["R1", "R3", "R2"]
    assert rows[0] == {
        "request_id": "R1", "leave_type": "annual", "start_date": "2026-11-02",
        "end_date": "2026-11-04", "working_days": 3, "status": "pending",
        "created_at": "2026-10-01T09:00:00",
    }


@pytest.mark.parametrize(
    "status, expected",
    [("pending", ["R1"]), ("approved", ["R2"]), ("cancelled", ["R3"])],
)
def test_requests_can_be_filtered_by_status(tools_db, status, expected):
    result = call(list_leave_requests, tools_db, "E1", status=status)
    assert [r["request_id"] for r in result.data["requests"]] == expected


def test_an_employee_without_requests_gets_an_empty_list_not_an_error(tools_db):
    result = call(list_leave_requests, tools_db, "N1")
    assert result.status == "success" and result.data == {"employee_id": "N1", "requests": []}
    assert call(list_leave_requests, tools_db, "E2", status="pending").data["requests"] == []


@pytest.mark.parametrize("bad", ["rejected", "Pending", "", "all", 1, ["pending"], True])
def test_an_unknown_status_is_an_error_with_a_clear_reason(tools_db, bad):
    result = call(list_leave_requests, tools_db, "E1", status=bad)
    assert result.status == "error" and result.data == {}
    assert "pending" in result.reason and "approved" in result.reason and "cancelled" in result.reason


def test_a_manager_sees_a_reports_requests_but_hr_sees_one_employee_at_a_time(tools_db):
    assert [r["request_id"] for r in call(list_leave_requests, tools_db, "M1", "E2").data["requests"]] == ["R4"]
    assert [r["request_id"] for r in call(list_leave_requests, tools_db, "H1", "E3").data["requests"]] == ["R5"]


def test_hr_asking_for_requests_of_an_unknown_employee_gets_an_error(tools_db):
    result = call(list_leave_requests, tools_db, "H1", "ZZ99")
    assert result.status == "error" and result.data == {}


# --- get_team_leave -------------------------------------------------------------------------------------------------------


def team(db, requester_id, start="2026-11-01", end="2026-11-30"):
    return get_team_leave(who(db, requester_id), start, end, db=db)


def request_ids(result):
    return [r["request_id"] for r in result.data["requests"]]


def test_a_manager_gets_pending_and_approved_leave_of_direct_reports_only(tools_db):
    result = team(tools_db, "M1")
    assert result.status == "success"
    # E1 pending, E2 approved, M3 pending. Not: R3 (cancelled), R6 (M1's own), R5 (another
    # team), R8 (E5 reports to M3, not to M1), R2 (outside the range)
    assert sorted(request_ids(result)) == ["R1", "R4", "R7"]
    assert {r["employee_id"] for r in result.data["requests"]} == {"E1", "E2", "M3"}
    by_id = {r["request_id"]: r for r in result.data["requests"]}
    assert by_id["R1"] == {
        "request_id": "R1", "employee_id": "E1", "full_name": "Eli Fenn", "leave_type": "annual",
        "start_date": "2026-11-02", "end_date": "2026-11-04", "working_days": 3, "status": "pending",
    }
    assert by_id["R4"]["full_name"] == "Esme Rook" and by_id["R7"]["full_name"] == "Milo Ashby"


def test_each_manager_sees_only_their_own_team(tools_db):
    assert request_ids(team(tools_db, "M2")) == ["R5"]
    assert request_ids(team(tools_db, "M3")) == ["R8"]  # M3's report E5, not M3's own R7


def test_a_manager_with_no_reports_gets_an_empty_list(tools_db):
    result = team(tools_db, "M3", "2026-01-01", "2026-10-31")
    assert result.status == "success" and result.data["requests"] == []


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2026-11-04", "2026-11-04", ["R1", "R4"]),  # a single day: R1 ends on it, R4 starts on it
        ("2026-11-01", "2026-11-02", ["R1"]),  # the range ends on R1's first day
        ("2026-10-30", "2026-11-01", []),  # ends the day before R1 starts
        ("2026-11-06", "2026-11-09", ["R4"]),  # starts on R4's last day
        ("2026-11-07", "2026-11-09", []),  # starts the day after R4 ends
        ("2026-11-05", "2026-11-05", ["R4", "R7"]),  # R4 spans it, R7 is on it
        ("2026-12-21", "2026-12-24", ["R2"]),  # exactly E1's approved request
        ("2026-12-24", "2026-12-24", ["R2"]),  # its last day
        ("2026-12-25", "2026-12-31", []),  # the day after
    ],
)
def test_overlap_boundaries_are_inclusive(tools_db, start, end, expected):
    assert sorted(request_ids(team(tools_db, "M1", start, end))) == expected


def test_cancelled_requests_are_excluded(tools_db):
    result = team(tools_db, "M1", "2026-11-10", "2026-11-11")  # R3 (cancelled) is on these days
    assert request_ids(result) == []


@pytest.mark.parametrize("requester_id", ["E1", "E3", "N1", "H1"])
def test_only_managers_may_use_get_team_leave(tools_db, requester_id):
    result = team(tools_db, requester_id)
    assert result.status == "error" and result.data == {}
    assert "manager" in result.reason.lower()
    assert result.reason != REFUSAL_MESSAGE


def test_hr_is_not_a_manager_for_team_leave_even_with_reports_in_the_data(tools_db):
    assert team(tools_db, "H1").status == "error"  # M1 and M2 report to H1; still refused


@pytest.mark.parametrize(
    "start, end",
    [("2026-11-30", "2026-11-01"), ("11/01/2026", "2026-11-30"), ("2026-11-01", "tomorrow"),
     ("2026-13-01", "2026-11-30"), ("2026-11-31", "2026-12-01"), ("2026-11-1", "2026-11-30"),
     ("", ""), (None, "2026-11-30"), ("2026-11-01", None), (20261101, "2026-11-30"),
     ("2026-11-01T00:00:00", "2026-11-30")],
)
def test_dates_must_be_valid_iso_dates_with_start_not_after_end(tools_db, start, end):
    result = team(tools_db, "M1", start, end)
    assert result.status == "error" and result.data == {}
    assert result.reason


def test_a_single_day_range_is_allowed(tools_db):
    assert team(tools_db, "M1", "2026-11-03", "2026-11-03").status == "success"


def test_get_team_leave_cannot_be_given_employees(tools_db):
    names = list(inspect.signature(get_team_leave).parameters)
    assert names[:3] == ["requester", "start_date", "end_date"]
    with pytest.raises(TypeError):
        get_team_leave(who(tools_db, "M1"), "2026-11-01", "2026-11-30", ["E3"], db=tools_db)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        get_team_leave(who(tools_db, "M1"), "2026-11-01", "2026-11-30", employee_ids=["E3"], db=tools_db)  # type: ignore[call-arg]


def test_team_leave_returns_names_of_reports_only_and_never_an_email_or_phone(tools_db):
    names, emails, phones = pii_values(tools_db)
    result = team(tools_db, "M1")
    blob = json.dumps(dataclasses.asdict(result))
    for email in emails:
        assert email not in blob
    for phone in phones:
        assert phone not in blob
    assert "@" not in blob
    reports = {"Eli Fenn", "Esme Rook", "Milo Ashby"}
    for name in names:
        assert (name in blob) == (name in reports), name


def test_team_leave_args_contain_only_the_dates(tools_db):
    result = team(tools_db, "M1")
    assert result.args == {"start_date": "2026-11-01", "end_date": "2026-11-30"}


# --- shapes shared by all three ----------------------------------------------------------------------------------------------


def test_the_b6_argument_names_are_the_only_arguments(tools_db):
    def positional(fn):
        return [n for n, p in inspect.signature(fn).parameters.items()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

    assert positional(get_leave_balance) == ["requester", "employee_id", "leave_type"]
    assert positional(list_leave_requests) == ["requester", "employee_id", "status"]
    assert positional(get_team_leave) == ["requester", "start_date", "end_date"]


def test_results_echo_only_b6_argument_names(tools_db):
    cases = [
        (get_leave_balance, call(get_leave_balance, tools_db, "E1", "E1", leave_type="sick"),
         {"employee_id", "leave_type"}),
        (list_leave_requests, call(list_leave_requests, tools_db, "M1", "E1", status="pending"),
         {"employee_id", "status"}),
    ]
    for _, result, allowed in cases:
        assert set(result.args) <= allowed and result.args  # something is echoed, nothing else
    assert call(get_leave_balance, tools_db, "E1").args == {}  # nothing was passed


def test_an_id_that_looks_like_a_person_is_not_echoed_into_the_args(tools_db):
    for bad in ("Priya Raman", "priya.raman@veltrix.example", "+91 98221 51875"):
        result = call(get_leave_balance, tools_db, "H1", bad)
        assert result.status == "error"
        blob = json.dumps(dataclasses.asdict(result))
        assert "Priya" not in blob and "@" not in blob and "98221" not in blob


def test_results_cannot_be_mutated(tools_db):
    result = call(get_leave_balance, tools_db, "E1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "error"  # type: ignore[misc]


@pytest.mark.parametrize("requester", [None, "E1", {"employee_id": "E1", "role": "hr"}])
def test_the_requester_must_be_an_employee_record(tools_db, requester):
    with pytest.raises(TypeError):
        get_leave_balance(requester, "E1", db=tools_db)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        get_team_leave(requester, "2026-11-01", "2026-11-02", db=tools_db)  # type: ignore[arg-type]


def test_every_role_is_allowed_to_read_its_own_data_and_get_an_answer_in_the_same_shape(tools_db):
    shapes = {
        tuple(call(get_leave_balance, tools_db, r).data) for r in ("E1", "M1", "H1")
    }
    assert shapes == {("employee_id", "balances")}
