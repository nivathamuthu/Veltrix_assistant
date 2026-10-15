"""Properties shared by all five tools: ToolResult, no clock, no personal data, no tables touched."""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path

import pytest

from app.tools import (
    REFUSAL_MESSAGE,
    STATUS_ERROR,
    STATUS_PENDING_CONFIRMATION,
    STATUS_SUCCESS,
    ToolResult,
    apply_leave,
    cancel_leave,
    get_leave_balance,
    get_team_leave,
    list_leave_requests,
    validate_apply_leave,
    validate_cancel_leave,
)
from tests.tool_helpers import TODAY, pii_values, who

TOOLS_DIR = Path(__file__).resolve().parent.parent / "app" / "tools"
B6_ARGUMENTS = {"employee_id", "leave_type", "status", "start_date", "end_date", "request_id"}


# --- ToolResult ---------------------------------------------------------------------------------------------


def test_tool_result_has_the_agreed_fields_and_statuses():
    assert [f.name for f in dataclasses.fields(ToolResult)] == ["name", "args", "status", "data", "reason"]
    assert (STATUS_SUCCESS, STATUS_ERROR) == ("success", "error")
    assert STATUS_PENDING_CONFIRMATION == "pending_confirmation"
    ok = ToolResult("get_leave_balance", {"employee_id": "E1"}, "success", {"x": 1}, None)
    assert ok.status == "success" and ok.reason is None


@pytest.mark.parametrize("status", ["ok", "pending", "", None, "SUCCESS"])
def test_only_the_three_known_statuses_are_valid(status):
    with pytest.raises(ValueError):
        ToolResult("get_leave_balance", {}, status, {}, None)


def test_pending_confirmation_is_a_valid_status_without_a_reason():
    pending = ToolResult("apply_leave", {"leave_type": "annual"}, "pending_confirmation", {"working_days": 3}, None)
    assert pending.status == STATUS_PENDING_CONFIRMATION and pending.reason is None
    with pytest.raises(ValueError):
        ToolResult("apply_leave", {}, "pending_confirmation", {}, "a reason")


SRC = Path(__file__).resolve().parent.parent / "app"


def test_only_the_confirmation_module_produces_pending_confirmation():
    """The tools never return it; the constant appears only where it is defined, produced
    (confirmation) or read (response)."""
    allowed = {
        SRC / "tools" / "common.py",
        SRC / "tools" / "__init__.py",  # re-export only
        SRC / "orchestrator" / "confirmation.py",
        SRC / "orchestrator" / "response.py",
        SRC / "orchestrator" / "service.py",  # strips it from a responder's output; never creates one
    }
    users = {path for path in SRC.rglob("*.py") if "STATUS_PENDING_CONFIRMATION" in path.read_text("utf-8")}
    assert users <= allowed, sorted(str(p) for p in users - allowed)
    for tool_file in (SRC / "tools").glob("*.py"):
        if tool_file.name not in ("common.py", "__init__.py"):
            assert "pending_confirmation" not in tool_file.read_text("utf-8").lower()


def test_an_error_needs_a_reason_and_a_success_does_not_have_one():
    with pytest.raises(ValueError):
        ToolResult("get_leave_balance", {}, "error", {}, None)
    with pytest.raises(ValueError):
        ToolResult("get_leave_balance", {}, "success", {}, "a reason")


def test_the_args_of_a_result_may_only_use_b6_argument_names():
    with pytest.raises(ValueError):
        ToolResult("get_leave_balance", {"full_name": "x"}, "success", {}, None)
    with pytest.raises(ValueError):
        ToolResult("get_leave_balance", {"email": "x"}, "success", {}, None)


def test_the_refusal_text_is_exact_and_free_of_any_identity():
    assert REFUSAL_MESSAGE == "I can't share information about that employee."


# --- nothing in the tools reads a clock --------------------------------------------------------------------------


def test_tool_code_never_calls_the_clock():
    forbidden = ["datetime.now", "date.today", "datetime.today", "time.time(", "utcnow", "time.monotonic",
                 "perf_counter", "import time", "from time import"]
    sources = sorted(TOOLS_DIR.glob("*.py"))
    assert len(sources) >= 7, "expected the tools package"
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text, f"{path.name} uses {pattern}"


def test_tool_code_has_no_llm_provider_or_orchestrator_imports():
    for path in TOOLS_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+(openai|anthropic|google|requests|httpx|app\.llm|app\.orchestrator)", text, re.M), path.name


def test_tool_sql_is_parameterised():
    for path in TOOLS_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"execute\(\s*f[\"']", text), path.name
        assert not re.search(r"execute\([^)]*(\.format\(|%\s*\()", text), path.name


# --- no personal data except the team view names -----------------------------------------------------------------


def run_everything(db):
    """One call to every tool across the roles, successes and errors alike."""
    e1, m1, h1, e3, e4 = (who(db, i) for i in ("E1", "M1", "H1", "E3", "E4"))
    results = [
        get_leave_balance(e1, db=db), get_leave_balance(m1, "E1", db=db), get_leave_balance(h1, "E3", db=db),
        get_leave_balance(e1, "E2", db=db), get_leave_balance(e1, "ZZ99", db=db),
        get_leave_balance(h1, "Priya Raman", db=db), get_leave_balance(e1, "E1", "nonsense", db=db),
        list_leave_requests(e1, db=db), list_leave_requests(m1, "E2", db=db), list_leave_requests(h1, "E4", db=db),
        list_leave_requests(e1, "M1", db=db), list_leave_requests(e1, status="bogus", db=db),
        get_team_leave(m1, "2026-11-01", "2026-11-30", db=db), get_team_leave(e1, "2026-11-01", "2026-11-30", db=db),
        get_team_leave(h1, "2026-11-01", "2026-11-30", db=db), get_team_leave(m1, "bad", "2026-11-30", db=db),
        apply_leave(e1, "annual", "2026-10-19", "2026-10-20", db=db, today=TODAY),
        apply_leave(e1, "annual", "2026-10-19", "2026-10-20", db=db, today=TODAY),
        apply_leave(e4, "annual", "2026-10-19", "2026-10-20", db=db, today=TODAY),
        apply_leave(e3, "x", "2026-10-19", "2026-10-20", db=db, today=TODAY),
        cancel_leave(e1, "R1", db=db, today=TODAY), cancel_leave(e1, "R1", db=db, today=TODAY),
        cancel_leave(e1, "R4", db=db, today=TODAY), cancel_leave(e3, "Priya Raman", db=db, today=TODAY),
    ]
    return results


def test_no_tool_ever_returns_an_email_or_phone_and_names_only_in_the_team_view(tools_db):
    names, emails, phones = pii_values(tools_db)
    for result in run_everything(tools_db):
        assert isinstance(result, ToolResult) and result.status in ("success", "error")
        assert set(result.args) <= B6_ARGUMENTS
        blob = json.dumps(dataclasses.asdict(result))
        for value in (*emails, *phones, "@", "veltrix.example"):
            assert value not in blob, (result.name, value)
        if result.name != "get_team_leave":
            for name in names:
                assert name not in blob, (result.name, name)
            assert "full_name" not in blob


def test_validators_return_no_personal_data_either(tools_db):
    names, emails, phones = pii_values(tools_db)
    checks = [
        validate_apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-20", db=tools_db, today=TODAY),
        validate_apply_leave(who(tools_db, "E4"), "annual", "2026-10-19", "2026-10-20", db=tools_db, today=TODAY),
        validate_cancel_leave(who(tools_db, "E1"), "R1", db=tools_db, today=TODAY),
        validate_cancel_leave(who(tools_db, "E1"), "R4", db=tools_db, today=TODAY),
    ]
    for check in checks:
        blob = json.dumps(dataclasses.asdict(check), default=str)
        for value in (*names, *emails, *phones):
            assert value not in blob


def test_logs_from_every_tool_contain_no_names_emails_or_phones(tools_db, caplog):
    names, emails, phones = pii_values(tools_db)
    with caplog.at_level(logging.DEBUG):
        run_everything(tools_db)
    for value in (*names, *emails, *phones, "Priya"):
        assert value not in caplog.text, value
    assert "@" not in caplog.text


def test_a_tool_call_logs_one_line_with_ids_and_status_only(tools_db, caplog):
    with caplog.at_level(logging.INFO, logger="app.tools"):
        get_leave_balance(who(tools_db, "M1"), "E3", db=tools_db)
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("app.tools")]
    assert len(lines) == 1
    assert "get_leave_balance" in lines[0] and "requester=M1" in lines[0] and "status=error" in lines[0]
    assert REFUSAL_MESSAGE not in lines[0]


# --- the tools write only where they should -------------------------------------------------------------------------------


def test_read_tools_never_write(tools_db, sql_log):
    sql_log.clear()
    for requester_id, call in [
        ("E1", lambda r: get_leave_balance(r, db=tools_db)),
        ("M1", lambda r: list_leave_requests(r, "E1", db=tools_db)),
        ("M1", lambda r: get_team_leave(r, "2026-11-01", "2026-11-30", db=tools_db)),
        ("H1", lambda r: get_leave_balance(r, "E3", db=tools_db)),
    ]:
        call(who(tools_db, requester_id))
    assert not [s for s in sql_log if re.match(r"\s*(INSERT|UPDATE|DELETE|BEGIN|COMMIT|CREATE|DROP|ALTER)", s, re.I)]


def test_the_write_tools_only_touch_leave_tables(tools_db, sql_log):
    sql_log.clear()
    apply_leave(who(tools_db, "E1"), "annual", "2026-10-19", "2026-10-19", db=tools_db, today=TODAY)
    cancel_leave(who(tools_db, "E1"), "R1", db=tools_db, today=TODAY)
    writes = [s for s in sql_log if re.match(r"\s*(INSERT|UPDATE|DELETE)", s, re.I)]
    assert writes
    for statement in writes:
        assert re.search(r"\b(leave_requests|leave_balances)\b", statement)
        assert not re.search(r"\b(employees|holidays|holiday_calendars|chunks|sessions)\b", statement)


def test_existing_tables_keep_their_schema(tools_db):
    from tests.db_helpers import schema_of

    before = schema_of(tools_db.path, ("employees", "leave_balances", "leave_requests"))
    run_everything(tools_db)
    assert schema_of(tools_db.path, ("employees", "leave_balances", "leave_requests")) == before
