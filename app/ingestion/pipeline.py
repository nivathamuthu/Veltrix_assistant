"""Startup ingestion.

Order (the service answers /health with 503 until the last step):

1. check the configured inputs exist;
2. run the idempotent DB migrations and prime the log redactor;
3. load the embedding model from the local cache (fails fast and clearly,
   before any slow OCR);
4. parse + OCR + chunk every manifest document (bad documents are skipped);
5. clear and refill the ``chunks`` table, build the BM25 + vector index from
   scratch, write it to INDEX_DIR;
6. mark ready.

Holiday extraction (3c) plugs in after step 5. Nothing here hard-codes corpus
facts.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.db import Database, get_pii_values, replace_chunks, run_migrations
from app.ingestion.corpus import CorpusLoad, load_corpus
from app.ingestion.embeddings import Embedder, FastEmbedEmbedder
from app.ingestion.index import HybridIndex
from app.ingestion.manifest import ManifestError
from app.ingestion.state import IngestionState
from app.logging_setup import get_logger, register_pii

log = get_logger(__name__)


class IngestionError(RuntimeError):
    """Fatal startup problem (missing inputs). The service stays at 503."""


@dataclass
class BuiltIndex:
    index: HybridIndex
    corpus: CorpusLoad
    seconds: dict[str, float] = field(default_factory=dict)


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


def build_search_index(settings: Settings, db: Database, embedder: Embedder) -> BuiltIndex:
    """Steps 4-5: corpus -> chunks table -> index on disk. Rebuilds from scratch."""
    seconds: dict[str, float] = {}

    started = time.monotonic()
    try:
        corpus = load_corpus(settings.corpus_dir, settings.manifest_path)
    except ManifestError as exc:
        raise IngestionError(str(exc)) from exc
    seconds["parse_and_chunk"] = time.monotonic() - started
    if not corpus.chunks:
        raise IngestionError("no document could be indexed (see the skip warnings above)")

    started = time.monotonic()
    try:
        with db.transaction() as conn:  # clear + refill in one step: no duplicates
            replace_chunks(conn, corpus.chunks)
    except sqlite3.Error as exc:
        raise IngestionError(f"cannot write the chunks table ({type(exc).__name__})") from exc
    seconds["store_chunks"] = time.monotonic() - started

    started = time.monotonic()
    index = HybridIndex.build(corpus.chunks, embedder)
    seconds["embed_and_index"] = time.monotonic() - started

    started = time.monotonic()
    try:
        index.save(settings.index_dir)
    except OSError as exc:
        raise IngestionError(
            f"cannot write the index to INDEX_DIR={settings.index_dir} ({type(exc).__name__})"
        ) from exc
    seconds["save_index"] = time.monotonic() - started
    return BuiltIndex(index=index, corpus=corpus, seconds=seconds)


def run_ingestion(
    settings: Settings, state: IngestionState, embedder: Embedder | None = None
) -> None:
    """Blocking ingestion; run it in a worker thread, never on the event loop.

    ``embedder`` is injected by tests; production loads the cached model.
    """
    started = time.monotonic()
    log.info("ingestion started")
    try:
        if settings.ingest_startup_delay_seconds:
            time.sleep(settings.ingest_startup_delay_seconds)
        doc_count = _check_inputs(settings)
        db = Database(settings.db_path)
        _migrate(db)
        employee_count = _prime_log_redactor(db)
        load_started = time.monotonic()
        if embedder is None:
            embedder = FastEmbedEmbedder(settings.embed_model, settings.embed_cache_dir)
        model_load_seconds = time.monotonic() - load_started
        built = build_search_index(settings, db, embedder)
        # Phase 3c plugs in here: holiday extraction from the calendar chunks.
    except Exception as exc:  # noqa: BLE001 - any failure must leave /health at 503
        state.mark_failed(f"{type(exc).__name__}: {exc}")
        log.error("ingestion failed: %s", type(exc).__name__)
        log.debug("ingestion failure detail", exc_info=True)
        return
    state.mark_ready(
        built.index,
        details={
            "seconds": {"load_embedding_model": model_load_seconds, **built.seconds},
            "total_seconds": time.monotonic() - started,
            "skipped_missing": built.corpus.skipped_missing,
            "skipped_unparsed": built.corpus.skipped_unparsed,
            "skipped_empty": built.corpus.skipped_empty,
            "unmanifested_files": built.corpus.unmanifested_files,
            "chunks_per_doc": built.corpus.chunks_per_doc,
        },
    )
    log.info(
        "ingestion finished in %.2fs (manifest_entries=%d, chunks=%d, docs_indexed=%d, "
        "employees_registered=%d, parse_and_chunk=%.2fs, embed_and_index=%.2fs)",
        time.monotonic() - started,
        doc_count,
        len(built.corpus.chunks),
        len(built.corpus.chunks_per_doc),
        employee_count,
        built.seconds["parse_and_chunk"],
        built.seconds["embed_and_index"],
    )
