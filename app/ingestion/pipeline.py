"""Startup ingestion.

For now this checks that the configured inputs exist, runs the idempotent DB
migrations and primes the log redactor from the employee table. Later phases
extend ``run_ingestion`` with
parsing + OCR (3a), chunking and indexing (3b) and holiday extraction (3c).
Nothing here may hard-code corpus facts.
"""
from __future__ import annotations

import json
import sqlite3
import time

from app.config import Settings
from app.db import Database, get_pii_values, run_migrations
from app.ingestion.state import IngestionState
from app.logging_setup import get_logger, register_pii

log = get_logger(__name__)


class IngestionError(RuntimeError):
    """Fatal startup problem (missing inputs). The service stays at 503."""


def _check_inputs(settings: Settings) -> int:
    """Return the number of manifest entries; raise if an input is unusable."""
    if not settings.corpus_dir.is_dir():
        raise IngestionError(f"CORPUS_DIR not found: {settings.corpus_dir}")
    if not settings.manifest_path.is_file():
        raise IngestionError(f"MANIFEST_PATH not found: {settings.manifest_path}")
    if not settings.db_path.is_file():
        raise IngestionError(f"DB_PATH not found: {settings.db_path}")
    try:
        manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IngestionError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, list):
        raise IngestionError("manifest must be a JSON array")
    return len(manifest)


def _migrate(db: Database) -> None:
    """Create the new tables/indexes (idempotent; existing tables untouched)."""
    try:
        run_migrations(db)
    except sqlite3.Error as exc:
        raise IngestionError(f"database migration failed: {exc}") from exc


def _prime_log_redactor(db: Database) -> int:
    """Register every employee's name, email and phone with the log redactor.

    The values are never logged, only the count.
    """
    try:
        with db.read() as conn:
            names, emails, phones = get_pii_values(conn)
    except sqlite3.Error as exc:
        raise IngestionError(f"cannot read employees table: {exc}") from exc
    register_pii(names=names, emails=emails, phones=phones)
    return len(names)


def run_ingestion(settings: Settings, state: IngestionState) -> None:
    """Blocking ingestion; run it in a worker thread, never on the event loop."""
    started = time.monotonic()
    log.info("ingestion started")
    try:
        if settings.ingest_startup_delay_seconds:
            time.sleep(settings.ingest_startup_delay_seconds)
        doc_count = _check_inputs(settings)
        db = Database(settings.db_path)
        _migrate(db)
        employee_count = _prime_log_redactor(db)
        # Phase 3a-3c plug in here: parse, OCR, chunk, index, extract holidays.
    except Exception as exc:  # noqa: BLE001 - any failure must leave /health at 503
        state.mark_failed(f"{type(exc).__name__}: {exc}")
        log.error("ingestion failed: %s", type(exc).__name__)
        log.debug("ingestion failure detail", exc_info=True)
        return
    state.mark_ready()
    log.info(
        "ingestion finished in %.2fs (manifest_entries=%d, employees_registered=%d)",
        time.monotonic() - started,
        doc_count,
        employee_count,
    )
