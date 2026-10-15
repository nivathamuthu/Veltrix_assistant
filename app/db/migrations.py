"""Idempotent startup migrations: NEW tables and indexes only.

The existing ``employees``, ``leave_balances`` and ``leave_requests`` tables
are never altered, dropped or written by anything in this module. Every
statement is ``CREATE ... IF NOT EXISTS``, so running the migrations on every
start (or twice in a row) is safe. They run in one transaction.

Contents of these tables are produced at runtime (ingestion, sessions); no
corpus facts are stored here as seed data.
"""
from __future__ import annotations

from app.db.connection import Database

# Tables this module owns (the existing three are deliberately absent).
NEW_TABLES = ("holidays", "chunks", "sessions", "pending_proposals")

MIGRATIONS: tuple[str, ...] = (
    # Public holidays extracted from calendar documents at ingestion.
    """
    CREATE TABLE IF NOT EXISTS holidays (
        region  TEXT NOT NULL,
        date    TEXT NOT NULL,          -- YYYY-MM-DD
        name    TEXT NOT NULL,
        doc_id  TEXT NOT NULL,          -- calendar document it came from
        PRIMARY KEY (region, date, doc_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_holidays_doc ON holidays (doc_id)",
    # Chunk text plus the manifest metadata every chunk carries.
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id        TEXT PRIMARY KEY,
        doc_id          TEXT NOT NULL,
        locator         TEXT NOT NULL,   -- nearest heading, else 'p. N'
        regions         TEXT NOT NULL,   -- JSON array, e.g. ["GLOBAL"]
        effective_date  TEXT NOT NULL,   -- YYYY-MM-DD
        doc_type        TEXT NOT NULL,
        access_level    TEXT NOT NULL,
        text            TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_access ON chunks (access_level, effective_date)",
    # A session belongs to the employee who sent its first message.
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id   TEXT PRIMARY KEY,
        employee_id  TEXT NOT NULL REFERENCES employees (employee_id),
        created_at   TEXT NOT NULL       -- ISO 8601 UTC
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_employee ON sessions (employee_id)",
    # At most one pending write proposal per session (the primary key).
    """
    CREATE TABLE IF NOT EXISTS pending_proposals (
        session_id    TEXT PRIMARY KEY REFERENCES sessions (session_id) ON DELETE CASCADE,
        action        TEXT NOT NULL CHECK (action IN ('apply_leave', 'cancel_leave')),
        payload_json  TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )
    """,
)


def run_migrations(db: Database) -> None:
    """Create any missing new tables and indexes. Safe to call repeatedly."""
    with db.transaction() as conn:
        for statement in MIGRATIONS:
            conn.execute(statement)
