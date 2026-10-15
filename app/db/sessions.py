"""Session ownership: a session belongs to whoever sent its first message."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Final, Literal

SessionCheck = Literal["ok", "forbidden"]

SESSION_OK: Final = "ok"
SESSION_FORBIDDEN: Final = "forbidden"

# INSERT OR IGNORE is one atomic statement: when two employees race on a new
# session id, exactly one row is created and the other insert is a no-op.
_SQL_CLAIM = (
    "INSERT OR IGNORE INTO sessions (session_id, employee_id, created_at) VALUES (?, ?, ?)"
)
_SQL_OWNER = "SELECT employee_id FROM sessions WHERE session_id = ?"


def claim_or_check_session(
    conn: sqlite3.Connection,
    session_id: str,
    employee_id: str,
    *,
    now: datetime | None = None,
) -> SessionCheck:
    """Claim ``session_id`` for ``employee_id`` if it is new.

    Returns ``"ok"`` if the session is new (now theirs) or already theirs, and
    ``"forbidden"`` if it belongs to a different employee. ``employee_id`` must
    already be a verified employee (foreign key); the API rejects unknown ids
    with 401 before reaching this point. The ownership row is never changed or
    deleted once written.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(employee_id, str) or not employee_id:
        raise ValueError("employee_id must be a non-empty string")
    created_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    conn.execute(_SQL_CLAIM, (session_id, employee_id, created_at))
    owner = conn.execute(_SQL_OWNER, (session_id,)).fetchone()["employee_id"]
    return SESSION_OK if owner == employee_id else SESSION_FORBIDDEN
