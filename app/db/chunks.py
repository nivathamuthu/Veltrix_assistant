"""The ``chunks`` table: a new table, rebuilt from scratch at every startup.

No access check here. Eligibility is applied by the search index *before*
anything is scored (see app/ingestion/index.py).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Iterable

from app.db.models import Chunk

_SQL_CLEAR = "DELETE FROM chunks"
_SQL_INSERT = (
    "INSERT INTO chunks (chunk_id, doc_id, locator, regions, effective_date, "
    "doc_type, access_level, text, token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_SQL_COUNT = "SELECT COUNT(*) AS n FROM chunks"
_SQL_DOC_LEVELS = "SELECT DISTINCT doc_id, access_level FROM chunks"
_SQL_ALL = (
    "SELECT chunk_id, doc_id, locator, regions, effective_date, doc_type, access_level, "
    "text, token_count FROM chunks ORDER BY chunk_id"
)


def replace_chunks(conn: sqlite3.Connection, chunks: Iterable[Chunk]) -> int:
    """Delete every row and insert ``chunks``. Call inside ``Database.transaction()``
    so a failure keeps the previous contents (and never leaves duplicates)."""
    rows = [
        (
            c.chunk_id, c.doc_id, c.locator, json.dumps(list(c.regions)),
            c.effective_date, c.doc_type, c.access_level, c.text, c.token_count,
        )
        for c in chunks
    ]
    conn.execute(_SQL_CLEAR)
    conn.executemany(_SQL_INSERT, rows)
    return len(rows)


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute(_SQL_COUNT).fetchone()["n"]


def readable_doc_ids(conn: sqlite3.Connection, access_levels: Iterable[str]) -> frozenset[str]:
    """The doc_ids that have chunks with one of ``access_levels``. The caller passes the
    levels its role may read (policy_resolver.readable_access_levels); no levels, no ids.
    Used as a last-line filter on citations, never to find documents to send."""
    levels = {a for a in access_levels if isinstance(a, str)}
    if not levels:
        return frozenset()
    rows = conn.execute(_SQL_DOC_LEVELS)  # constant SQL; the filter is applied in Python
    return frozenset(r["doc_id"] for r in rows if r["access_level"] in levels)


_SQL_DOC_SECTIONS = "SELECT locator, text FROM chunks WHERE doc_id = ? ORDER BY chunk_id"


def doc_sections(conn: sqlite3.Connection, doc_id: str) -> list[tuple[str, str]]:
    """``(locator, text)`` of each chunk of ONE document, in document order. Used to find the
    section of a calendar document to cite; the caller has already checked that the requester
    may read the document (the calendar lookup filters by access level)."""
    return [(r["locator"], r["text"]) for r in conn.execute(_SQL_DOC_SECTIONS, (doc_id,))]


def load_chunks(conn: sqlite3.Connection) -> list[Chunk]:
    """All stored chunks, ordered by chunk_id (segments are not stored)."""
    return [
        Chunk(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"], locator=r["locator"],
            regions=tuple(json.loads(r["regions"])), effective_date=r["effective_date"],
            doc_type=r["doc_type"], access_level=r["access_level"], text=r["text"],
            token_count=r["token_count"],
        )
        for r in conn.execute(_SQL_ALL)
    ]
