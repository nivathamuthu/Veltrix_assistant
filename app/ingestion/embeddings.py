"""Dense embeddings: BAAI/bge-small-en-v1.5 through fastembed (ONNX, CPU).

The model is downloaded once, while the Docker image is built (see
Dockerfile), into ``EMBED_CACHE_DIR``. At runtime it is loaded from there with
``local_files_only=True``: no network access is attempted. If the files are not
there the service says so clearly and stays "not ready" instead of silently
falling back to a download.

Everything that needs vectors depends on the small ``Embedder`` interface, so
tests can use a lightweight fake and never need the model.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from app.logging_setup import get_logger

log = get_logger(__name__)

_BATCH_SIZE = 32


class EmbeddingModelUnavailable(RuntimeError):
    """The embedding model is not in the local cache (or cannot be loaded)."""


class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors (cosine = dot product)."""

    name: str
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Shape ``(dim,)``."""
        ...


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class FastEmbedEmbedder:
    """Loads the cached ONNX model offline. Construct once per process."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.name = model_name
        started = time.monotonic()
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir),
                local_files_only=True,  # never touch the network
            )
            self.dim = int(next(iter(self._model.embed(["dimension probe"]))).shape[0])
        except Exception as exc:  # noqa: BLE001 - missing files, bad name, onnx errors
            raise EmbeddingModelUnavailable(
                f"embedding model '{model_name}' could not be loaded from '{cache_dir}' "
                f"({type(exc).__name__}). The model is downloaded while the Docker image "
                "is built (docker compose build); run it once with network access, or "
                "point EMBED_CACHE_DIR at a folder that already contains it."
            ) from exc
        log.info(
            "embedding model loaded: name=%s dim=%d in %.2fs",
            model_name, self.dim, time.monotonic() - started,
        )

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self._model.embed(list(texts), batch_size=_BATCH_SIZE))
        return normalise_rows(np.vstack(vectors))

    def embed_query(self, query: str) -> np.ndarray:
        vector = next(iter(self._model.embed([query])))
        return normalise_rows(np.asarray(vector))
