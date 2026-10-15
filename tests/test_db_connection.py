import ast
import sqlite3
from pathlib import Path

import pytest

from app.db import Database


def test_rows_are_dict_like_and_foreign_keys_are_on(synthetic_db):
    with synthetic_db.read() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        row = conn.execute("SELECT employee_id, region FROM employees WHERE employee_id = ?", ("E1",)).fetchone()
        assert row["employee_id"] == "E1" and row["region"] == "IN"


def test_read_connection_cannot_write(synthetic_db):
    with synthetic_db.read() as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE leave_balances SET balance_days = 999")
    with synthetic_db.read() as conn:
        assert conn.execute("SELECT MAX(balance_days) FROM leave_balances").fetchone()[0] != 999


def test_missing_database_raises_and_is_not_created(tmp_path: Path):
    missing = tmp_path / "nope.db"
    for opener in (Database(missing).read, Database(missing).write, Database(missing).transaction):
        with pytest.raises(sqlite3.OperationalError):
            with opener():
                pass
    assert not missing.exists()


def test_connection_is_closed_after_the_block(synthetic_db):
    with synthetic_db.read() as conn:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_path_with_spaces_works(tmp_path: Path):
    from tests.db_helpers import build_synthetic_db

    folder = tmp_path / "my data dir"
    folder.mkdir()
    db = Database(build_synthetic_db(folder / "veltrix copy.db"))
    with db.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] > 0


# --- transactions ---------------------------------------------------------


def _balance(db, employee_id="E1", leave_type="annual"):
    with db.read() as conn:
        return conn.execute(
            "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
            (employee_id, leave_type),
        ).fetchone()["balance_days"]


def _request_count(db):
    with db.read() as conn:
        return conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]


def _two_step_write(conn):
    """What apply_leave will do: insert a request and decrement the balance."""
    conn.execute(
        "INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)",
        ("NEW1", "E1", "annual", "2026-12-01", "2026-12-02", 2, "pending", "2026-10-15T00:00:00"),
    )
    conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days - ? "
        "WHERE employee_id = ? AND leave_type = ?",
        (2, "E1", "annual"),
    )


def test_transaction_commits_on_success(synthetic_db):
    before_requests, before_balance = _request_count(synthetic_db), _balance(synthetic_db)
    with synthetic_db.transaction() as conn:
        _two_step_write(conn)
    assert _request_count(synthetic_db) == before_requests + 1
    assert _balance(synthetic_db) == before_balance - 2


def test_transaction_rolls_back_every_statement_on_exception(synthetic_db):
    before_requests, before_balance = _request_count(synthetic_db), _balance(synthetic_db)
    with pytest.raises(RuntimeError, match="boom"):
        with synthetic_db.transaction() as conn:
            _two_step_write(conn)
            raise RuntimeError("boom")
    assert _request_count(synthetic_db) == before_requests
    assert _balance(synthetic_db) == before_balance


def test_transaction_rolls_back_when_a_later_statement_fails(synthetic_db):
    """A constraint error half-way through must undo the earlier statement."""
    before_requests, before_balance = _request_count(synthetic_db), _balance(synthetic_db)
    with pytest.raises(sqlite3.IntegrityError):
        with synthetic_db.transaction() as conn:
            _two_step_write(conn)
            # duplicate primary key -> IntegrityError
            conn.execute(
                "INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)",
                ("R1", "E1", "annual", "2026-12-01", "2026-12-02", 2, "pending", "x"),
            )
    assert _request_count(synthetic_db) == before_requests
    assert _balance(synthetic_db) == before_balance


def test_foreign_keys_are_enforced_inside_transactions(synthetic_db):
    before = _request_count(synthetic_db)
    with pytest.raises(sqlite3.IntegrityError):
        with synthetic_db.transaction() as conn:
            conn.execute(
                "INSERT INTO leave_requests VALUES (?,?,?,?,?,?,?,?)",
                ("BAD", "NO_SUCH_EMPLOYEE", "annual", "2026-12-01", "2026-12-02", 2, "pending", "x"),
            )
    assert _request_count(synthetic_db) == before


def test_rollback_also_happens_for_keyboard_interrupt_style_exits(synthetic_db):
    before = _request_count(synthetic_db)
    with pytest.raises(KeyboardInterrupt):
        with synthetic_db.transaction() as conn:
            _two_step_write(conn)
            raise KeyboardInterrupt
    assert _request_count(synthetic_db) == before


def test_transaction_takes_the_write_lock_immediately(synthetic_db):
    """A second writer must wait (and here time out) while the first is open."""
    with synthetic_db.transaction():
        other = sqlite3.connect(synthetic_db.path, timeout=0.1, isolation_level=None)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        finally:
            other.close()


def test_a_failed_transaction_does_not_leave_the_database_locked(synthetic_db):
    with pytest.raises(RuntimeError):
        with synthetic_db.transaction():
            raise RuntimeError
    with synthetic_db.transaction() as conn:  # would raise "locked" if a lock leaked
        conn.execute("SELECT 1")


def test_rollback_works_on_a_copy_of_the_real_database(real_db_copy):
    with real_db_copy.read() as conn:
        before = conn.execute("SELECT COUNT(*), SUM(balance_days) FROM leave_balances").fetchone()
        victim = conn.execute("SELECT employee_id, leave_type FROM leave_balances LIMIT 1").fetchone()
    with pytest.raises(RuntimeError):
        with real_db_copy.transaction() as conn:
            conn.execute(
                "UPDATE leave_balances SET balance_days = 0 WHERE employee_id = ? AND leave_type = ?",
                (victim["employee_id"], victim["leave_type"]),
            )
            conn.execute("DELETE FROM leave_requests")
            raise RuntimeError
    with real_db_copy.read() as conn:
        after = conn.execute("SELECT COUNT(*), SUM(balance_days) FROM leave_balances").fetchone()
        assert tuple(after) == tuple(before)
        assert conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] > 0


# --- all SQL is parameterised ---------------------------------------------

_DB_PACKAGE = Path(__file__).resolve().parent.parent / "app" / "db"
_EXECUTE_METHODS = {"execute", "executemany", "executescript"}


def test_sql_is_never_string_formatted():
    """Every execute() call in app/db passes a constant string (or a module
    constant) as the statement: no f-strings, %-formatting, .format() or
    concatenation with runtime values at the call site."""
    offenders = []
    calls = 0
    for source in _DB_PACKAGE.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _EXECUTE_METHODS
            ):
                calls += 1
                statement = node.args[0] if node.args else None
                allowed = isinstance(statement, ast.Name) or (
                    isinstance(statement, ast.Constant) and isinstance(statement.value, str)
                )
                if node.func.attr == "executescript" or not allowed:
                    offenders.append(f"{source.name}:{node.lineno}")
    assert calls > 5, "the scan found suspiciously few execute() calls"
    assert not offenders, f"non-constant or script SQL at: {offenders}"


def test_module_level_sql_constants_are_built_only_from_constants():
    """The module-level SQL strings may concatenate other string constants
    (column lists) but nothing computed at runtime."""
    for source in _DB_PACKAGE.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("_SQL") for t in node.targets
            ):
                for sub in ast.walk(node.value):
                    assert not isinstance(sub, (ast.JoinedStr, ast.Call)), (
                        f"{source.name}:{node.lineno} builds SQL at import time with "
                        f"{type(sub).__name__}"
                    )
                    assert not (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mod))
