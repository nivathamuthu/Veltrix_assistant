"""Incomplete apply_leave details remembered across one session's next message.

Not a write proposal: nothing is submitted until propose_apply stores a pending_proposals
row and the employee replies yes. This only keeps the leave type and the employee's own
date words so a follow-up like "annual" can finish "apply leave on Monday".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.db.connection import Database

_SQL_UPSERT = (
    "INSERT OR REPLACE INTO apply_drafts (session_id, leave_type, dates_text, created_at) "
    "VALUES (?, ?, ?, ?)"
)
_SQL_GET = "SELECT session_id, leave_type, dates_text FROM apply_drafts WHERE session_id = ?"
_SQL_DELETE = "DELETE FROM apply_drafts WHERE session_id = ?"


@dataclass(frozen=True)
class ApplyDraft:
    session_id: str
    leave_type: str | None
    dates_text: str | None


def save_apply_draft(
    db: Database,
    session_id: str,
    *,
    leave_type: str | None,
    dates_text: str | None,
    now: datetime | None = None,
) -> None:
    created_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    with db.write() as conn:
        conn.execute(_SQL_UPSERT, (session_id, leave_type or "", dates_text or "", created_at))


def get_apply_draft(db: Database, session_id: str) -> ApplyDraft | None:
    with db.read() as conn:
        row = conn.execute(_SQL_GET, (session_id,)).fetchone()
    if row is None:
        return None
    leave_type = row["leave_type"] or None
    dates_text = row["dates_text"] or None
    return ApplyDraft(row["session_id"], leave_type, dates_text)


def delete_apply_draft(db: Database, session_id: str) -> None:
    with db.write() as conn:
        conn.execute(_SQL_DELETE, (session_id,))
