"""Download everything the service needs at runtime, once, at image build time.

* the embedding model (fastembed, ONNX) into EMBED_CACHE_DIR
* the tiktoken ``cl100k_base`` encoding into TIKTOKEN_CACHE_DIR

The running container has no network access requirement: it loads both from
these folders (HF_HUB_OFFLINE=1, local_files_only=True). Run it manually with
network access for a local, non-Docker setup:

    python scripts/download_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CACHE = Path(os.environ.get("EMBED_CACHE_DIR", "./models"))


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=MODEL, cache_dir=str(CACHE))  # downloads
    dim = len(next(iter(model.embed(["download check"]))))
    print(f"embedding model ready: {MODEL} dim={dim} cache={CACHE}")

    import tiktoken

    tokens = len(tiktoken.get_encoding("cl100k_base").encode("download check"))
    print(f"tiktoken cl100k_base ready ({tokens} tokens in the check string)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
