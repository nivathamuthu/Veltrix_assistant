"""SQLite access: one short-lived connection per operation.

* ``Database.read()``        read-only connection (``mode=ro``): cannot write.
* ``Database.write()``       read-write, autocommit: for single statements.
* ``Database.transaction()`` read-write, ``BEGIN IMMEDIATE`` ... commit on
  success, roll back on *any* exception. Use it for multi-statement writes
  (``apply_leave`` / ``cancel_leave``): the write lock is taken up front, so a
  balance check and the update that follows cannot interleave with another
  writer.

Every connection has ``sqlite3.Row`` rows and foreign keys ON. Connections
open with ``mode=ro``/``mode=rw`` only, so a wrong ``DB_PATH`` raises instead
of silently creating an empty database file.

All SQL in this package is parameterised; values are never formatted into SQL.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

# Seconds to wait for another writer's lock. Well under the 30 s request limit.
BUSY_TIMEOUT_SECONDS = 5.0


def _open(path: Path, *, readonly: bool) -> sqlite3.Connection:
    mode = "ro" if readonly else "rw"
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=" + mode
    # isolation_level=None -> the sqlite3 module never opens transactions
    # behind our back; transactions are explicit (see Database.transaction).
    conn = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
    except BaseException:
        conn.close()
        raise
    return conn


class Database:
    """Handle on the SQLite file at ``path``. Holds no open connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = _open(self.path, readonly=True)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        conn = _open(self.path, readonly=False)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit if the block finishes, roll back if it raises anything."""
        conn = _open(self.path, readonly=False)
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
