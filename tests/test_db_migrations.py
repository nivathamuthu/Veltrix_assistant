import sqlite3
from pathlib import Path

import pytest

from app.db import NEW_TABLES, run_migrations
from tests.db_helpers import dump_tables, schema_of

EXISTING_TABLES = ("employees", "leave_balances", "leave_requests")


def _names(db, kind):
    with db.read() as conn:
        return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def _columns(db, table):
    with db.read() as conn:
        return {r["name"]: r for r in conn.execute("SELECT * FROM pragma_table_info(?)", (table,))}


def test_migrations_create_the_new_tables_and_indexes(core_db):
    run_migrations(core_db)
    assert set(NEW_TABLES) <= _names(core_db, "table")
    indexes = _names(core_db, "index")
    assert {
        "idx_holidays_doc",
        "idx_chunks_doc",
        "idx_chunks_access",
        "idx_sessions_employee",
    } <= indexes


def test_migrations_run_twice_without_error_and_change_nothing(core_db):
    run_migrations(core_db)
    with core_db.read() as conn:
        first = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    run_migrations(core_db)
    run_migrations(core_db)
    with core_db.read() as conn:
        again = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    assert [tuple(r) for r in first] == [tuple(r) for r in again]


def test_migrations_keep_data_in_new_tables(synthetic_db):
    run_migrations(synthetic_db)
    with synthetic_db.transaction() as conn:
        conn.execute("INSERT INTO holidays VALUES ('IN', '2026-01-26', 'Republic Day', 'DOC-X')")
    run_migrations(synthetic_db)  # a restart must not wipe or fail on existing rows
    with synthetic_db.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM holidays").fetchone()[0] == 1


def test_existing_tables_are_unchanged(core_db):
    path = core_db.path
    schema_before = schema_of(path, EXISTING_TABLES)
    rows_before = dump_tables(path, EXISTING_TABLES)
    assert set(schema_before) == set(EXISTING_TABLES)

    run_migrations(core_db)
    run_migrations(core_db)

    assert schema_of(path, EXISTING_TABLES) == schema_before  # same DDL, same columns
    assert dump_tables(path, EXISTING_TABLES) == rows_before  # same rows


def test_migrations_never_reference_existing_tables_destructively():
    """Static guard: only CREATE ... IF NOT EXISTS statements, nothing else."""
    from app.db.migrations import MIGRATIONS

    for statement in MIGRATIONS:
        normalised = " ".join(statement.split()).upper()
        assert normalised.startswith(("CREATE TABLE IF NOT EXISTS ", "CREATE INDEX IF NOT EXISTS "))
        # "ON DELETE CASCADE" is a constraint on a new table, not a DELETE statement.
        checked = normalised.replace("ON DELETE CASCADE", "")
        assert not any(word in checked for word in ("DROP ", "ALTER ", "DELETE ", "UPDATE ", "INSERT "))
    created = {
        " ".join(s.split()).split("EXISTS ")[1].split(" ")[0]
        for s in MIGRATIONS
        if "CREATE TABLE" in s
    }
    assert created == set(NEW_TABLES)
    assert not created & set(EXISTING_TABLES)


def test_foreign_keys_still_consistent_after_migration(core_db):
    run_migrations(core_db)
    with core_db.read() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


# --- shape of the new tables ----------------------------------------------


@pytest.mark.parametrize(
    "table, expected",
    [
        ("holidays", ["region", "date", "name", "doc_id"]),
        (
            "chunks",
            ["chunk_id", "doc_id", "locator", "regions", "effective_date", "doc_type", "access_level",
             "text", "token_count"],
        ),
        ("sessions", ["session_id", "employee_id", "created_at"]),
        ("pending_proposals", ["session_id", "action", "payload_json", "created_at"]),
    ],
)
def test_new_table_columns(synthetic_db, table, expected):
    run_migrations(synthetic_db)
    cols = _columns(synthetic_db, table)
    assert sorted(cols, key=lambda c: cols[c]["cid"]) == expected
    assert all(c["notnull"] or c["pk"] for c in cols.values())


def test_pending_proposals_allows_one_row_per_session(synthetic_db):
    run_migrations(synthetic_db)
    with synthetic_db.transaction() as conn:
        conn.execute("INSERT INTO sessions VALUES ('S1', 'E1', '2026-10-15T00:00:00')")
        conn.execute("INSERT INTO pending_proposals VALUES ('S1', 'apply_leave', '{}', 'now')")
    with pytest.raises(sqlite3.IntegrityError):
        with synthetic_db.transaction() as conn:
            conn.execute("INSERT INTO pending_proposals VALUES ('S1', 'cancel_leave', '{}', 'now')")


def test_pending_proposals_needs_a_real_session_and_a_known_action(synthetic_db):
    run_migrations(synthetic_db)
    with pytest.raises(sqlite3.IntegrityError):  # no such session
        with synthetic_db.transaction() as conn:
            conn.execute("INSERT INTO pending_proposals VALUES ('GHOST', 'apply_leave', '{}', 'now')")
    with pytest.raises(sqlite3.IntegrityError):  # unknown action
        with synthetic_db.transaction() as conn:
            conn.execute("INSERT INTO sessions VALUES ('S2', 'E1', 'now')")
            conn.execute("INSERT INTO pending_proposals VALUES ('S2', 'delete_everything', '{}', 'now')")


def test_sessions_reference_real_employees(synthetic_db):
    run_migrations(synthetic_db)
    with pytest.raises(sqlite3.IntegrityError):
        with synthetic_db.transaction() as conn:
            conn.execute("INSERT INTO sessions VALUES ('S3', 'NO_SUCH_EMPLOYEE', 'now')")


def test_chunks_table_from_phase_2_gains_token_count_without_losing_rows(synthetic_db):
    """A database migrated before Phase 3b has no token_count; adding it is idempotent."""
    with synthetic_db.transaction() as conn:
        conn.execute(
            "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, locator TEXT NOT NULL,"
            " regions TEXT NOT NULL, effective_date TEXT NOT NULL, doc_type TEXT NOT NULL,"
            " access_level TEXT NOT NULL, text TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO chunks VALUES ('C1','D1','h','[\"GLOBAL\"]','2026-01-01','policy','all','t')")
    run_migrations(synthetic_db)
    run_migrations(synthetic_db)  # twice: no error, no second column
    cols = _columns(synthetic_db, "chunks")
    assert list(cols).count("token_count") == 1
    with synthetic_db.read() as conn:
        assert conn.execute("SELECT chunk_id, token_count FROM chunks").fetchall()[0]["token_count"] == 0


def test_startup_runs_migrations_and_never_creates_a_missing_db(
    tmp_path, data_dir, monkeypatch, fake_embedder
):
    """Through the real startup path: tables appear; a wrong DB_PATH stays a failure."""
    import time

    from fastapi.testclient import TestClient

    from app.config import load_settings
    from app.main import create_app

    monkeypatch.setenv("CORPUS_DIR", str(data_dir / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(data_dir / "manifest.json"))
    monkeypatch.setenv("DB_PATH", str(data_dir / "veltrix.db"))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    with TestClient(create_app(load_settings(env_file=None), embedder=fake_embedder)) as client:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and client.get("/health").status_code != 200:
            time.sleep(0.05)
        assert client.get("/health").status_code == 200
    conn = sqlite3.connect(data_dir / "veltrix.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert set(NEW_TABLES) <= tables

    ghost = tmp_path / "ghost.db"
    monkeypatch.setenv("DB_PATH", str(ghost))
    with TestClient(create_app(load_settings(env_file=None), embedder=fake_embedder)) as client:
        time.sleep(0.5)
        assert client.get("/health").status_code == 503
    assert not Path(ghost).exists()
