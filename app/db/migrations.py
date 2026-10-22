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
NEW_TABLES = (
    "holidays", "holiday_calendars", "chunks", "sessions", "pending_proposals", "apply_drafts",
)

MIGRATIONS: tuple[str, ...] = (
    # Holidays extracted from calendar documents at ingestion (rebuilt at startup).
    # Only rows with kind = 'public' may count as non-working days.
    """
    CREATE TABLE IF NOT EXISTS holidays (
        region          TEXT NOT NULL,
        date            TEXT NOT NULL,          -- YYYY-MM-DD
        name            TEXT NOT NULL,
        doc_id          TEXT NOT NULL,          -- calendar document it came from
        kind            TEXT NOT NULL DEFAULT 'unknown',  -- public | optional | unknown
        effective_date  TEXT NOT NULL DEFAULT '',          -- of the calendar document
        regions         TEXT NOT NULL DEFAULT '[]',        -- JSON, of the calendar document
        PRIMARY KEY (region, date, doc_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_holidays_doc ON holidays (doc_id)",
    # One row per calendar document in the manifest, whether or not extraction worked.
    """
    CREATE TABLE IF NOT EXISTS holiday_calendars (
        doc_id          TEXT PRIMARY KEY,
        effective_date  TEXT NOT NULL,
        regions         TEXT NOT NULL,          -- JSON array
        access_level    TEXT NOT NULL,
        status          TEXT NOT NULL,          -- ok | no_public_holidays | unavailable
        public_count    INTEGER NOT NULL DEFAULT 0,
        optional_count  INTEGER NOT NULL DEFAULT 0,
        unknown_count   INTEGER NOT NULL DEFAULT 0,
        skipped_rows    INTEGER NOT NULL DEFAULT 0
    )
    """,
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
        text            TEXT NOT NULL,
        token_count     INTEGER NOT NULL DEFAULT 0   -- cl100k_base tokens in text
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
    # Incomplete apply details (leave type / date words) until the request is complete.
    """
    CREATE TABLE IF NOT EXISTS apply_drafts (
        session_id   TEXT PRIMARY KEY REFERENCES sessions (session_id) ON DELETE CASCADE,
        leave_type   TEXT NOT NULL DEFAULT '',
        dates_text   TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL
    )
    """,
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


# A database first migrated in Phase 2 has a ``chunks`` table without
# token_count. Adding a column to OUR OWN table is allowed (rule 9 protects the
# given tables); it is guarded so repeated runs do nothing.
#
# The same applies to ``holidays`` (Phase 3c adds kind, effective_date, regions).
# A legacy row would default to kind 'unknown', which can never count as a
# public holiday.
_SQL_HAS_COLUMN = "SELECT 1 FROM pragma_table_info(?) WHERE name = ?"
_SQL_ADD_TOKEN_COUNT = "ALTER TABLE chunks ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0"
_SQL_ADD_HOLIDAY_KIND = "ALTER TABLE holidays ADD COLUMN kind TEXT NOT NULL DEFAULT 'unknown'"
_SQL_ADD_HOLIDAY_EFFECTIVE = "ALTER TABLE holidays ADD COLUMN effective_date TEXT NOT NULL DEFAULT ''"
_SQL_ADD_HOLIDAY_REGIONS = "ALTER TABLE holidays ADD COLUMN regions TEXT NOT NULL DEFAULT '[]'"

# (table, column, statement that adds it), applied only where the column is missing.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("chunks", "token_count", _SQL_ADD_TOKEN_COUNT),
    ("holidays", "kind", _SQL_ADD_HOLIDAY_KIND),
    ("holidays", "effective_date", _SQL_ADD_HOLIDAY_EFFECTIVE),
    ("holidays", "regions", _SQL_ADD_HOLIDAY_REGIONS),
)


def run_migrations(db: Database) -> None:
    """Create any missing new tables and indexes. Safe to call repeatedly."""
    with db.transaction() as conn:
        for statement in MIGRATIONS:
            conn.execute(statement)
        for table, column, add_statement in _ADDED_COLUMNS:
            if conn.execute(_SQL_HAS_COLUMN, (table, column)).fetchone() is None:
                conn.execute(add_statement)
