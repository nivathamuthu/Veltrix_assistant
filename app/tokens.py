"""Token counting with tiktoken ``cl100k_base``.

The 3,000-token corpus limit per LLM call (Phase 5) and the chunk sizes
(Phase 3b) are both measured with this encoding, whichever LLM is used.

The encoding file is read from ``TIKTOKEN_CACHE_DIR``. The Docker image
downloads it at build time, so no network access is needed at runtime.
"""
from __future__ import annotations

from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Number of cl100k_base tokens in ``text``.

    Text such as ``<|endoftext|>`` is counted as ordinary characters instead of
    raising: document content must never be able to break the counter.
    """
    if not text:
        return 0
    return len(get_encoding().encode(text, disallowed_special=()))


def encode(text: str) -> list[int]:
    return get_encoding().encode(text, disallowed_special=())


def decode(token_ids: list[int]) -> str:
    return get_encoding().decode(token_ids)
