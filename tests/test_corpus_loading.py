"""Manifest handling, skipping rules and rebuilds (Phase 3b)."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from app.config import load_settings
from app.db import Database, count_chunks, load_chunks, run_migrations
from app.ingestion.chunker import HARD_MAX_TOKENS
from app.ingestion.corpus import load_corpus
from app.ingestion.manifest import ManifestError, load_manifest
from app.ingestion.pipeline import build_search_index, run_ingestion
from app.ingestion.state import IngestionState
from tests.fake_embedder import FakeEmbedder

DEFAULT_META = {
    "doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2026-01-01",
    "access_level": "all", "title": "Synthetic",
}

LEAVE_MD = (
    "# Leave Rules\n\nStaff may carry over 28 days of leave into the next year.\n\n"
    "## Sick leave\n\nA certificate is needed after 3 days of sick leave.\n\n"
    "| Tenure | Accrual |\n|---|---|\n| 0-2 years | 1.5 days/month |\n| 3+ years | 2 days/month |\n"
)


def write_corpus(root: Path, files: dict[str, tuple[str, dict]], extra_manifest=()) -> Path:
    """Create root/corpus/<files> and root/manifest.json. ``files`` maps a file name
    to (content, manifest overrides including doc_id)."""
    (root / "corpus").mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (content, meta) in files.items():
        (root / "corpus" / name).write_text(content, encoding="utf-8")
        manifest.append({**DEFAULT_META, "file": name, **meta})
    manifest.extend(extra_manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    return write_corpus(
        tmp_path / "c",
        {
            "leave.md": (LEAVE_MD, {"doc_id": "D-LEAVE"}),
            "other.md": ("# Other\n\nSome other policy text.\n", {"doc_id": "D-OTHER", "regions": ["DE"]}),
        },
    )


# --- manifest -------------------------------------------------------------


def test_manifest_entries_are_loaded_with_all_fields(corpus_root):
    entries = load_manifest(corpus_root / "manifest.json")
    assert [e.doc_id for e in entries] == ["D-LEAVE", "D-OTHER"]
    assert entries[1].regions == ("DE",)


def test_bad_manifest_entries_are_skipped_and_only_the_doc_id_is_logged(tmp_path, caplog):
    good = {**DEFAULT_META, "doc_id": "D-GOOD", "file": "good.md"}
    bad = [
        {**good, "doc_id": "D-NODATE", "file": "a.md", "effective_date": "not-a-date"},
        {**good, "doc_id": "D-TRAVERSAL", "file": "../secret.md"},
        {**good, "doc_id": "D-NOREGION", "file": "b.md", "regions": []},
        {**good, "doc_id": "D-DUP", "file": "c.md"},
        {**good, "doc_id": "D-GOOD", "file": "d.md"},  # duplicate doc_id
        {**good, "doc_id": "D-SAMEFILE", "file": "good.md"},  # duplicate file
        "not even an object",
        {"doc_id": "D-EMPTY"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps([good, *bad]), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        entries = load_manifest(tmp_path / "manifest.json")
    assert [e.doc_id for e in entries] == ["D-GOOD", "D-DUP"]
    logged = caplog.text
    for doc_id in ("D-NODATE", "D-TRAVERSAL", "D-NOREGION", "D-SAMEFILE", "D-EMPTY"):
        assert doc_id in logged
    assert "Synthetic" not in logged  # titles and other content are never logged


def test_unreadable_manifest_raises_a_clear_error(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "manifest.json")
    (tmp_path / "manifest.json").write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "manifest.json")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "missing.json")


# --- skipping ---------------------------------------------------------------


def test_manifest_entry_with_a_missing_file_is_skipped(corpus_root, caplog):
    (corpus_root / "corpus" / "other.md").unlink()
    with caplog.at_level(logging.WARNING):
        result = load_corpus(corpus_root / "corpus", corpus_root / "manifest.json")
    assert result.skipped_missing == ["D-OTHER"]
    assert set(result.chunks_per_doc) == {"D-LEAVE"}
    assert "D-OTHER" in caplog.text


def test_corpus_file_without_a_manifest_entry_is_skipped_and_never_indexed(corpus_root, caplog):
    (corpus_root / "corpus" / "stray.md").write_text("# Stray\n\nSTRAYTEXT secret.\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        result = load_corpus(corpus_root / "corpus", corpus_root / "manifest.json")
    assert result.unmanifested_files == ["stray.md"]
    assert not any("STRAYTEXT" in c.text for c in result.chunks)
    assert "stray.md" in caplog.text and "STRAYTEXT" not in caplog.text


def test_a_file_that_fails_to_parse_is_skipped(tmp_path, caplog):
    root = write_corpus(
        tmp_path,
        {
            "ok.md": ("# Fine\n\nReadable text.\n", {"doc_id": "D-OK"}),
            "broken.pdf": ("%PDF-1.4 definitely not a pdf SECRETBYTES", {"doc_id": "D-BROKEN"}),
            "odd.xyz": ("unsupported format", {"doc_id": "D-ODD"}),
        },
    )
    with caplog.at_level(logging.WARNING):
        result = load_corpus(root / "corpus", root / "manifest.json")
    assert set(result.chunks_per_doc) == {"D-OK"}
    assert sorted(result.skipped_unparsed) == ["D-BROKEN", "D-ODD"]
    assert "D-BROKEN" in caplog.text and "SECRETBYTES" not in caplog.text


def test_hidden_files_in_the_corpus_folder_are_ignored(corpus_root):
    (corpus_root / "corpus" / ".DS_Store").write_text("junk", encoding="utf-8")
    result = load_corpus(corpus_root / "corpus", corpus_root / "manifest.json")
    assert result.unmanifested_files == []


# --- startup never crashes -------------------------------------------------


@pytest.fixture()
def pipeline_settings(settings, corpus_root, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(corpus_root / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(corpus_root / "manifest.json"))
    return load_settings(env_file=None)


def test_startup_survives_missing_unmanifested_and_broken_documents(
    settings, tmp_path, monkeypatch, fake_embedder
):
    root = write_corpus(
        tmp_path / "mixed",
        {
            "ok.md": (LEAVE_MD, {"doc_id": "D-OK"}),
            "broken.pdf": ("not a pdf", {"doc_id": "D-BROKEN"}),
            "stray.md": ("# Stray\n\nnot in the manifest\n", {"doc_id": "D-STRAY"}),
        },
        extra_manifest=[{**DEFAULT_META, "doc_id": "D-GONE", "file": "gone.md"}],
    )
    # take stray.md out of the manifest again: present on disk, no entry
    manifest = [e for e in json.loads((root / "manifest.json").read_text()) if e["doc_id"] != "D-STRAY"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("CORPUS_DIR", str(root / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(root / "manifest.json"))

    state = IngestionState()
    run_ingestion(load_settings(env_file=None), state, fake_embedder)

    assert state.ready and state.error is None
    ids = {c.doc_id for c in load_chunks(_open(settings))}
    assert ids == {"D-OK"}


def test_startup_stays_not_ready_when_nothing_can_be_indexed(settings, tmp_path, monkeypatch, fake_embedder):
    root = write_corpus(tmp_path / "none", {"broken.pdf": ("nope", {"doc_id": "D-BROKEN"})})
    monkeypatch.setenv("CORPUS_DIR", str(root / "corpus"))
    monkeypatch.setenv("MANIFEST_PATH", str(root / "manifest.json"))
    state = IngestionState()
    run_ingestion(load_settings(env_file=None), state, fake_embedder)
    assert not state.ready
    assert "no document could be indexed" in state.error


def test_startup_stays_not_ready_with_an_unreadable_manifest(settings, monkeypatch, fake_embedder):
    settings.manifest_path.write_text("{broken", encoding="utf-8")
    state = IngestionState()
    run_ingestion(settings, state, fake_embedder)
    assert not state.ready and state.error


# --- rebuilds ----------------------------------------------------------------


def _open(settings) -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _build(settings, embedder=None):
    db = Database(settings.db_path)
    run_migrations(db)
    return build_search_index(settings, db, embedder or FakeEmbedder()), db


def test_rebuilding_twice_gives_identical_chunks_and_no_duplicates(pipeline_settings):
    first, db = _build(pipeline_settings)
    with db.read() as conn:
        rows_first = [tuple(r) for r in conn.execute("SELECT * FROM chunks ORDER BY chunk_id")]
    second, db = _build(pipeline_settings)
    with db.read() as conn:
        rows_second = [tuple(r) for r in conn.execute("SELECT * FROM chunks ORDER BY chunk_id")]
        distinct = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunks").fetchone()[0]
        total = count_chunks(conn)

    assert len(first.corpus.chunks) == len(second.corpus.chunks) > 0
    assert total == distinct == len(second.corpus.chunks)  # cleared and refilled
    assert rows_first == rows_second
    assert len(first.index) == len(second.index) == total


def test_rebuild_removes_chunks_of_documents_that_disappeared(pipeline_settings, corpus_root):
    _build(pipeline_settings)
    manifest = json.loads((corpus_root / "manifest.json").read_text())
    (corpus_root / "manifest.json").write_text(json.dumps(manifest[:1]), encoding="utf-8")
    _, db = _build(pipeline_settings)
    with db.read() as conn:
        assert {c.doc_id for c in load_chunks(conn)} == {"D-LEAVE"}


def test_rebuild_writes_the_index_files_and_replaces_stale_ones(pipeline_settings):
    index_dir = pipeline_settings.index_dir
    index_dir.mkdir(parents=True)
    (index_dir / "vectors.npy").write_bytes(b"stale")
    built, _ = _build(pipeline_settings)
    meta = json.loads((index_dir / "index_meta.json").read_text())
    assert meta["chunk_count"] == len(built.corpus.chunks)
    ids = json.loads((index_dir / "chunk_ids.json").read_text())
    assert ids == [c.chunk_id for c in built.corpus.chunks]
    import numpy as np

    assert np.load(index_dir / "vectors.npy").shape == (len(ids), built.index.stats.dim)


def test_changing_a_number_in_a_document_changes_the_chunk_text_after_rebuild(
    pipeline_settings, corpus_root
):
    before, db = _build(pipeline_settings)
    text_before = "\n".join(c.text for c in before.corpus.chunks)
    assert "28 days" in text_before and "35 days" not in text_before

    leave = corpus_root / "corpus" / "leave.md"
    leave.write_text(leave.read_text(encoding="utf-8").replace("28 days", "35 days"), encoding="utf-8")
    after, db = _build(pipeline_settings)
    text_after = "\n".join(c.text for c in after.corpus.chunks)
    assert "35 days" in text_after and "28 days" not in text_after
    with db.read() as conn:  # and the table holds the new text, not the old
        stored = "\n".join(c.text for c in load_chunks(conn))
    assert "35 days" in stored and "28 days" not in stored


def test_the_chunks_table_carries_manifest_metadata_and_token_counts(pipeline_settings):
    from app.tokens import count_tokens

    built, db = _build(pipeline_settings)
    with db.read() as conn:
        stored = load_chunks(conn)
    by_doc = {c.doc_id: c for c in stored}
    assert by_doc["D-OTHER"].regions == ("DE",)
    for c in stored:
        assert c.effective_date == "2026-01-01" and c.access_level == "all" and c.doc_type == "policy"
        assert c.token_count == count_tokens(c.text) > 0


# --- the real corpus (skipped when data/ is absent) -----------------------


DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.skipif(not (DATA / "manifest.json").is_file(), reason="data/ not present")
def test_real_corpus_respects_limits_and_copies_manifest_metadata():
    entries = {e.doc_id: e for e in load_manifest(DATA / "manifest.json")}
    result = load_corpus(DATA / "corpus", DATA / "manifest.json")
    assert result.chunks and not result.unmanifested_files
    for c in result.chunks:
        assert c.token_count <= HARD_MAX_TOKENS
        entry = entries[c.doc_id]
        assert (c.regions, c.effective_date, c.doc_type, c.access_level) == (
            entry.regions, entry.effective_date, entry.doc_type, entry.access_level,
        )
    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids))
    # every manifest document that parsed produced at least one chunk
    for doc_id in result.chunks_per_doc:
        assert result.chunks_per_doc[doc_id] >= 1


@pytest.mark.skipif(not (DATA / "manifest.json").is_file(), reason="data/ not present")
def test_real_corpus_tables_are_never_cut_mid_row():
    result = load_corpus(DATA / "corpus", DATA / "manifest.json")
    for c in result.chunks:
        body = c.text.split("\n\n", 1)[-1]
        for line in body.split("\n"):
            if line.startswith("Row:"):
                # a complete row has every column: "Row: a | b ..." never ends on a bare separator
                assert not line.rstrip().endswith("|")
