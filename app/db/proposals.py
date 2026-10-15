"""Pending confirmation proposals (README B8): at most ONE per session.

The table (``pending_proposals``, created by the idempotent migrations) has the session id
as its primary key, so a second proposal can only replace the first: there is no way to
hold two. ``claim_proposal`` reads and deletes in one step, so of two concurrent
confirmations only one can obtain the proposal and run it.

The payload holds ids, a leave type and dates: never free text from the user, never a name,
email or phone.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final

from app.db.connection import Database

ACTION_APPLY: Final = "apply_leave"
ACTION_CANCEL: Final = "cancel_leave"
ACTIONS: Final = (ACTION_APPLY, ACTION_CANCEL)


@dataclass(frozen=True)
class Proposal:
    session_id: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)  # {} when the stored JSON is not an object
    created_at: str = ""


def _decode(raw: object) -> dict[str, Any]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _from_row(row: sqlite3.Row) -> Proposal:
    return Proposal(row["session_id"], row["action"], _decode(row["payload_json"]), row["created_at"])


def save_proposal(
    conn: sqlite3.Connection,
    session_id: str,
    action: str,
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
) -> None:
    """Store the session's one proposal, replacing any earlier one (a single statement)."""
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    created_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR REPLACE INTO pending_proposals (session_id, action, payload_json, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, action, json.dumps(payload, sort_keys=True), created_at),
    )


def get_proposal(conn: sqlite3.Connection, session_id: str) -> Proposal | None:
    row = conn.execute(
        "SELECT session_id, action, payload_json, created_at FROM pending_proposals WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return None if row is None else _from_row(row)


def delete_proposal(conn: sqlite3.Connection, session_id: str) -> bool:
    return conn.execute("DELETE FROM pending_proposals WHERE session_id = ?", (session_id,)).rowcount == 1


def claim_proposal(conn: sqlite3.Connection, session_id: str) -> Proposal | None:
    """Take the proposal out of the table. Call inside ``Database.transaction()`` (BEGIN
    IMMEDIATE): the write lock makes the read and the delete one step, and only the caller
    whose DELETE removes the row gets the proposal."""
    proposal = get_proposal(conn, session_id)
    if proposal is None:
        return None
    return proposal if delete_proposal(conn, session_id) else None


def count_proposals(db: Database) -> int:
    with db.read() as conn:
        return conn.execute("SELECT COUNT(*) FROM pending_proposals").fetchone()[0]
