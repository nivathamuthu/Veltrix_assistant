import sqlite3
from datetime import datetime, timezone

import pytest

from app.db import SESSION_FORBIDDEN, SESSION_OK, claim_or_check_session, run_migrations


@pytest.fixture()
def db(synthetic_db):
    run_migrations(synthetic_db)
    return synthetic_db


def _claim(db, session_id, employee_id, **kw):
    with db.transaction() as conn:
        return claim_or_check_session(conn, session_id, employee_id, **kw)


def _owner(db, session_id):
    with db.read() as conn:
        row = conn.execute("SELECT employee_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["employee_id"] if row else None


def test_constants_have_the_documented_values():
    assert (SESSION_OK, SESSION_FORBIDDEN) == ("ok", "forbidden")


def test_first_message_claims_the_session(db):
    assert _owner(db, "S1") is None
    assert _claim(db, "S1", "E1") == "ok"
    assert _owner(db, "S1") == "E1"


def test_owner_can_keep_using_their_session(db):
    assert _claim(db, "S1", "E1") == "ok"
    assert _claim(db, "S1", "E1") == "ok"
    with db.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = 'S1'").fetchone()[0] == 1


def test_another_employee_is_forbidden_and_ownership_is_unchanged(db):
    _claim(db, "S1", "E1")
    assert _claim(db, "S1", "E2") == "forbidden"
    assert _claim(db, "S1", "H1") == "forbidden"  # even HR cannot take over a session
    assert _owner(db, "S1") == "E1"
    assert _claim(db, "S1", "E1") == "ok"  # and the owner is unaffected


def test_forbidden_does_not_depend_on_role_or_region_of_the_intruder(db):
    _claim(db, "S1", "E4")
    for intruder in ("E1", "M1", "H1", "M2"):
        assert _claim(db, "S1", intruder) == "forbidden"


def test_sessions_are_independent(db):
    assert _claim(db, "S1", "E1") == "ok"
    assert _claim(db, "S2", "E2") == "ok"
    assert _claim(db, "S2", "E1") == "forbidden"
    assert _claim(db, "S1", "E2") == "forbidden"


def test_ownership_survives_new_connections(db):
    """Ownership lives in the database, not in process memory."""
    _claim(db, "S1", "E1")
    with db.write() as conn:  # a completely separate connection
        assert claim_or_check_session(conn, "S1", "E2") == "forbidden"
        assert claim_or_check_session(conn, "S1", "E1") == "ok"


def test_session_ids_are_case_sensitive_and_stored_literally(db):
    weird = "'; DROP TABLE sessions; --"
    assert _claim(db, "abc", "E1") == "ok"
    assert _claim(db, "ABC", "E2") == "ok"
    assert _claim(db, weird, "E1") == "ok"
    assert _claim(db, weird, "E2") == "forbidden"
    assert _owner(db, "abc") == "E1" and _owner(db, "ABC") == "E2" and _owner(db, weird) == "E1"
    with db.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3


def test_created_at_is_iso_utc_and_can_be_injected(db):
    when = datetime(2026, 10, 15, 8, 30, tzinfo=timezone.utc)
    _claim(db, "S1", "E1", now=when)
    with db.read() as conn:
        stored = conn.execute("SELECT created_at FROM sessions WHERE session_id = 'S1'").fetchone()[0]
    assert stored == "2026-10-15T08:30:00+00:00"
    _claim(db, "S2", "E1")
    with db.read() as conn:
        auto = conn.execute("SELECT created_at FROM sessions WHERE session_id = 'S2'").fetchone()[0]
    assert datetime.fromisoformat(auto).tzinfo is not None


@pytest.mark.parametrize("session_id, employee_id", [("", "E1"), ("S1", ""), (None, "E1"), ("S1", None)])
def test_empty_arguments_are_rejected_and_nothing_is_stored(db, session_id, employee_id):
    with pytest.raises(ValueError):
        _claim(db, session_id, employee_id)
    with db.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_unknown_employee_cannot_claim_a_session(db):
    """The API returns 401 before this; the foreign key is the backstop."""
    with pytest.raises(sqlite3.IntegrityError):
        _claim(db, "S1", "NO_SUCH_EMPLOYEE")
    assert _owner(db, "S1") is None


def test_a_rolled_back_claim_leaves_the_session_unclaimed(db):
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            assert claim_or_check_session(conn, "S1", "E1") == "ok"
            raise RuntimeError
    assert _owner(db, "S1") is None
    assert _claim(db, "S1", "E2") == "ok"  # so the next sender may claim it


def test_racing_claims_have_exactly_one_winner(db):
    """Two connections claim the same new id; the loser is told 'forbidden'."""
    results = []
    with db.write() as first, db.write() as second:
        results.append(claim_or_check_session(first, "RACE", "E1"))
        results.append(claim_or_check_session(second, "RACE", "E2"))
    assert results == ["ok", "forbidden"]
    assert _owner(db, "RACE") == "E1"


def test_forbidden_looks_the_same_for_every_existing_owner(db):
    """The result never says who owns the session."""
    _claim(db, "A", "E1")
    _claim(db, "B", "M1")
    assert _claim(db, "A", "E3") == _claim(db, "B", "E3") == "forbidden"
