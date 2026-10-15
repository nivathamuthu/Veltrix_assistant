"""Manifest + corpus folder + parsers + chunker -> chunks.

Nothing here can stop startup because of one bad document:

* manifest entry whose file is missing        -> skipped, logged by doc_id
* manifest entry whose file failed to parse   -> skipped, logged by doc_id
* corpus file with no manifest entry          -> skipped, logged by file name
  (without metadata we cannot tell who may read it, so it is never indexed)

Only identifiers and error type names are logged, never document content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.db.models import Chunk
from app.ingestion.chunker import chunk_document
from app.ingestion.manifest import ManifestEntry, load_manifest
from app.ingestion.parsers import ParsedDocument, parse_document
from app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LoadedDocument:
    """A manifest entry with its parsed file (kept so later steps, such as holiday
    extraction, reuse it instead of parsing, and OCR-ing, a second time)."""

    entry: ManifestEntry
    parsed: ParsedDocument


@dataclass
class CorpusLoad:
    entries: list[ManifestEntry] = field(default_factory=list)  # every valid manifest entry
    documents: list[LoadedDocument] = field(default_factory=list)  # those that parsed
    chunks: list[Chunk] = field(default_factory=list)
    chunks_per_doc: dict[str, int] = field(default_factory=dict)
    skipped_missing: list[str] = field(default_factory=list)  # doc_ids
    skipped_unparsed: list[str] = field(default_factory=list)  # doc_ids
    skipped_empty: list[str] = field(default_factory=list)  # doc_ids with no text
    unmanifested_files: list[str] = field(default_factory=list)  # file names


def _corpus_files(corpus_dir: Path) -> list[Path]:
    return sorted(
        p for p in corpus_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )


def load_corpus(corpus_dir: Path, manifest_path: Path) -> CorpusLoad:
    """Parse and chunk every document that has a manifest entry and a readable file."""
    corpus_dir = Path(corpus_dir)
    entries = load_manifest(manifest_path)
    result = CorpusLoad(entries=list(entries))

    listed_files = {e.file for e in entries}
    for path in _corpus_files(corpus_dir):
        if path.name not in listed_files:
            log.warning("skipping corpus file with no manifest entry: %s", path.name)
            result.unmanifested_files.append(path.name)

    for entry in entries:
        _load_entry(corpus_dir, entry, result)

    log.info(
        "corpus loaded: docs=%d chunks=%d skipped_missing=%d skipped_unparsed=%d "
        "skipped_empty=%d unmanifested_files=%d",
        len(result.chunks_per_doc),
        len(result.chunks),
        len(result.skipped_missing),
        len(result.skipped_unparsed),
        len(result.skipped_empty),
        len(result.unmanifested_files),
    )
    return result


def _load_entry(corpus_dir: Path, entry: ManifestEntry, result: CorpusLoad) -> None:
    path = corpus_dir / entry.file
    if not path.is_file():
        log.warning("skipping doc_id=%s: file is missing", entry.doc_id)
        result.skipped_missing.append(entry.doc_id)
        return
    parsed = parse_document(path)  # never raises
    if not parsed.ok:
        log.warning("skipping doc_id=%s: file failed to parse (%s)", entry.doc_id, parsed.error)
        result.skipped_unparsed.append(entry.doc_id)
        return
    result.documents.append(LoadedDocument(entry, parsed))
    chunks = chunk_document(entry, parsed)
    if not chunks:
        log.warning("skipping doc_id=%s: no text was extracted", entry.doc_id)
        result.skipped_empty.append(entry.doc_id)
        return
    result.chunks.extend(chunks)
    result.chunks_per_doc[entry.doc_id] = len(chunks)
