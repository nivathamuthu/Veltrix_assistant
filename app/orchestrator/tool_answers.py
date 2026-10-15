"""A deterministic rendering of a tool result as a ``ChatResult`` (route ``tool``).

Names appear only where the tool returned them (a manager's own direct reports); emails and
phones are never in tool data. Used by the service for leave-data intents and by the hybrid
policy flow when the leave tool fails.
"""
from __future__ import annotations

from typing import Any

from app.orchestrator.response import NO_ANSWER_MESSAGE, ROUTE_TOOL, ChatResult, tool_call_of
from app.tools import STATUS_ERROR, ToolResult


def _days(n: Any) -> str:
    return f"{n:g}"


def tool_answer(result: ToolResult) -> ChatResult:
    call = tool_call_of(result)
    if result.status == STATUS_ERROR:
        return ChatResult(answer=result.reason or NO_ANSWER_MESSAGE, route=ROUTE_TOOL, tool_calls=(call,))
    data = result.data
    if result.name == "get_leave_balance":
        who = result.args.get("employee_id")
        head = "Your leave balance" if who is None else f"Leave balance for {data['employee_id']}"
        parts = [f"{b['leave_type']}: {_days(b['balance_days'])} days" for b in data["balances"]]
        text = f"{head}: " + "; ".join(parts) + "."
    elif result.name == "list_leave_requests":
        head = "Your leave requests" if "employee_id" not in result.args else f"Leave requests for {data['employee_id']}"
        rows = [
            f"- {r['request_id']}: {r['leave_type']}, {r['start_date']} to {r['end_date']} "
            f"({r['working_days']} working days), {r['status']}"
            for r in data["requests"]
        ]
        text = f"{head}:\n" + "\n".join(rows) if rows else "No leave requests found."
    else:  # get_team_leave
        span = f"between {data['start_date']} and {data['end_date']}"
        rows = [
            f"- {r['full_name']} ({r['employee_id']}): {r['leave_type']}, {r['start_date']} to "
            f"{r['end_date']} ({r['working_days']} working days), {r['status']}"
            for r in data["requests"]
        ]
        text = (
            f"Your direct reports on leave {span}:\n" + "\n".join(rows)
            if rows else f"None of your direct reports are on leave {span}."
        )
    return ChatResult(answer=text, route=ROUTE_TOOL, tool_calls=(call,))
