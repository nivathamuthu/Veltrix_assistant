"""The real embedding model, loaded offline. Skipped where it was not downloaded
(i.e. outside the Docker image, unless `python scripts/download_models.py` was run)."""
from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pytest

from app.db.models import Chunk
from app.ingestion.embeddings import EmbeddingModelUnavailable, FastEmbedEmbedder
from app.ingestion.index import HybridIndex, SearchFilters
from app.tokens import count_tokens

MODEL = "BAAI/bge-small-en-v1.5"
# Read at import: the autouse fixture in conftest clears the environment per test.
CACHE_DIR = Path(os.environ.get("EMBED_CACHE_DIR", "./models"))


@pytest.fixture(scope="module")
def real_embedder():
    try:
        return FastEmbedEmbedder(MODEL, CACHE_DIR)
    except EmbeddingModelUnavailable:
        pytest.skip(f"embedding model not available in {CACHE_DIR}")


@pytest.fixture()
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id, locator="h", regions=("GLOBAL",),
        effective_date="2026-01-01", doc_type="policy", access_level="all",
        text=text, token_count=count_tokens(text),
    )


def test_model_loads_and_embeds_with_the_network_blocked(no_network):
    try:
        embedder = FastEmbedEmbedder(MODEL, CACHE_DIR)  # a fresh load, not the cached fixture
    except EmbeddingModelUnavailable:
        pytest.skip(f"embedding model not available in {CACHE_DIR}")
    vectors = embedder.embed_passages(["Employees may work from home.", "Travel expenses."])
    assert vectors.shape == (2, 384) and embedder.dim == 384
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert embedder.embed_query("work from home").shape == (384,)


def test_model_ranks_by_meaning_not_only_by_shared_words(real_embedder):
    chunks = [
        _chunk("REMOTE", "Employees may work from home up to three days per week with manager approval."),
        _chunk("EXPENSE", "Travel expenses are reimbursed when receipts are submitted within thirty days."),
        _chunk("HOLIDAY", "The Berlin office is closed on public holidays and on two company days."),
    ]
    index = HybridIndex.build(chunks, real_embedder)
    filters = SearchFilters(allowed_access_levels=frozenset({"all"}), region="DE")
    # "telework" and "remotely" appear nowhere in the text: only meaning can connect them
    assert index.search("how often can I telework", filters, 3)[0].chunk_id == "REMOTE"
    assert index.search("which days is the office shut", filters, 3)[0].chunk_id == "HOLIDAY"


def test_a_missing_model_directory_is_a_clear_error_and_never_a_download(tmp_path, no_network):
    with pytest.raises(EmbeddingModelUnavailable) as caught:
        FastEmbedEmbedder(MODEL, tmp_path / "nothing_here")
    assert "EMBED_CACHE_DIR" in str(caught.value)
