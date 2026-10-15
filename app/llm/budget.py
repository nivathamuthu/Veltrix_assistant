"""The ONE place that counts and enforces corpus tokens (project rule 3).

No more than ``MAX_CORPUS_TOKENS`` (3,000, tiktoken ``cl100k_base``) of corpus
text may reach an LLM in a single call. "Corpus text" is anything derived from
the documents: retrieved chunks, notes or summaries computed from them, document
text read from the database, and the label line (doc_id, locator, effective
date) in front of each of those.

``enforce_budget`` is the only function that decides what fits. It works on
WHOLE parts in rank order: the lowest-ranked part is dropped first, and only a
single remaining part that is still too big is cut (by tokens). It never returns
a block over the limit, whatever it is given, and it reports exactly which
doc_ids and locators survived, so a citation can only ever name text that was
actually sent.

The count is of the exact block that will be sent (parts rendered and joined),
not a sum of per-part counts: tokenisation across a boundary is not guaranteed
to add up.

Nothing in this module logs corpus text.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime

from app.tokens import count_tokens as _count_tokens
from app.tokens import decode as _decode
from app.tokens import encode as _encode

MAX_CORPUS_TOKENS = 3000  # the hard ceiling; a caller may lower the limit, never raise it

PART_SEPARATOR = "\n\n"
BLOCK_OPEN = "<documents>\n"
BLOCK_CLOSE = "\n</documents>"

KIND_CHUNK = "chunk"  # a retrieved chunk
KIND_NOTE = "note"  # a summary / calculation / note derived from corpus text
KIND_DOCUMENT = "document"  # document text read from the database
_KINDS = (KIND_CHUNK, KIND_NOTE, KIND_DOCUMENT)

# Document text must not be able to end the corpus block early and pose as the
# prompt that follows it. Neutralised before counting, so the count stays exact.
_CLOSING_TAG = re.compile(r"<\s*/\s*documents\s*>", re.IGNORECASE)
_NEUTRAL_CLOSE = "<\\/documents>"
_WHITESPACE = re.compile(r"\s+")


def count_tokens(text: str) -> int:
    """Number of ``cl100k_base`` tokens in ``text``. Special-token look-alikes such
    as ``<|endoftext|>`` are counted as ordinary characters, never rejected."""
    return _count_tokens(text)


# --------------------------------------------------------------------------- #
# Parts and context
# --------------------------------------------------------------------------- #


def _one_line(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


@dataclass(frozen=True)
class CorpusPart:
    """One piece of corpus-derived text with where it came from.

    ``doc_id`` and ``locator`` are required: a part that cannot be cited has no
    business in a prompt. A note computed from several documents should be
    split into one part per source document.
    """

    text: str
    doc_id: str
    locator: str
    effective_date: str | date
    kind: str = KIND_CHUNK

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        for name in ("doc_id", "locator"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not _one_line(value):
                raise ValueError(f"{name} is required")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}")
        stamp = self.effective_date
        if isinstance(stamp, datetime):
            stamp = stamp.date()
        if isinstance(stamp, date):
            iso = stamp.isoformat()
        elif isinstance(stamp, str):
            iso = date.fromisoformat(stamp).isoformat()  # ValueError for anything not YYYY-MM-DD
        else:
            raise TypeError("effective_date must be a date or an ISO date string")
        object.__setattr__(self, "effective_date", iso)

    @property
    def label(self) -> str:
        """The label line. One line whatever the metadata contains."""
        return f"[{_one_line(self.doc_id)} | {_one_line(self.locator)} | effective {self.effective_date}]"

    def render(self) -> str:
        safe_text = _CLOSING_TAG.sub(lambda _match: _NEUTRAL_CLOSE, self.text)
        return f"{self.label}\n{safe_text}"


@dataclass(frozen=True)
class Source:
    """Where a piece of sent text came from: what a citation may name."""

    doc_id: str
    locator: str


class CorpusContext:
    """An ordered list of corpus-derived parts, best first.

    Add parts in rank order: when the budget is tight the LAST ones go first.
    """

    def __init__(self, parts: Iterable[CorpusPart] = ()) -> None:
        self._parts: list[CorpusPart] = []
        for part in parts:
            self.add(part)

    def add(self, part: CorpusPart) -> "CorpusContext":
        if not isinstance(part, CorpusPart):
            raise TypeError("a CorpusContext holds CorpusPart objects only")
        self._parts.append(part)
        return self

    def add_chunk(self, chunk) -> "CorpusContext":
        """A retrieved chunk (anything with doc_id, locator, effective_date and text)."""
        return self.add(
            CorpusPart(chunk.text, chunk.doc_id, chunk.locator, chunk.effective_date, KIND_CHUNK)
        )

    def add_note(self, text: str, *, doc_id: str, locator: str, effective_date) -> "CorpusContext":
        """A summary or calculation derived from corpus text. It counts like corpus text."""
        return self.add(CorpusPart(text, doc_id, locator, effective_date, KIND_NOTE))

    def add_document_text(self, text: str, *, doc_id: str, locator: str, effective_date) -> "CorpusContext":
        """Document text read from the database."""
        return self.add(CorpusPart(text, doc_id, locator, effective_date, KIND_DOCUMENT))

    @property
    def parts(self) -> tuple[CorpusPart, ...]:
        return tuple(self._parts)

    def __iter__(self) -> Iterator[CorpusPart]:
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)


def render_block(parts: Iterable[CorpusPart]) -> str:
    """The exact corpus text for these parts, as it is counted and as it is sent."""
    return PART_SEPARATOR.join(p.render() for p in parts)


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BudgetResult:
    parts: tuple[CorpusPart, ...]  # what is kept (a lone survivor may be truncated)
    block: str  # render_block(parts)
    tokens: int  # count_tokens(block): exact, never above ``limit``
    limit: int
    kept: tuple[Source, ...]  # doc_id + locator of every kept part, in rank order
    dropped: tuple[Source, ...]  # whole parts removed to fit, lowest rank (dropped first) first
    truncated: tuple[Source, ...]  # parts that were cut

    @property
    def kept_doc_ids(self) -> tuple[str, ...]:
        """Distinct doc_ids that were sent, in rank order."""
        return tuple(dict.fromkeys(s.doc_id for s in self.kept))


def _validated_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if not 0 <= limit <= MAX_CORPUS_TOKENS:
        raise ValueError(f"limit must be between 0 and {MAX_CORPUS_TOKENS}")
    return limit


def _candidates(parts: Iterable[CorpusPart]) -> list[CorpusPart]:
    """Drop blank parts and exact repeats (same document, locator and text)."""
    seen: set[tuple[str, str, str]] = set()
    result = []
    for part in parts:
        if not isinstance(part, CorpusPart):
            raise TypeError("enforce_budget takes CorpusPart objects only")
        key = (part.doc_id, part.locator, part.text)
        if not part.text.strip() or key in seen:
            continue
        seen.add(key)
        result.append(part)
    return result


def _truncate(part: CorpusPart, limit: int) -> CorpusPart | None:
    """``part`` cut by tokens so that, alone, it fits ``limit``. None if not even one
    token of its text fits next to its label (a citation needs text that was sent)."""
    tokens = _encode(part.text)

    def cut(n: int) -> CorpusPart:
        return CorpusPart(_decode(tokens[:n]), part.doc_id, part.locator, part.effective_date, part.kind)

    def fits(n: int) -> bool:
        return count_tokens(render_block([cut(n)])) <= limit

    low, high = 0, len(tokens)  # the largest n that fits lies in [low, high]
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    n = low
    while n > 0 and not fits(n):  # decoding a partial character can shift the count: step back
        n -= 1
    if n == 0:
        return None
    shortened = cut(n)
    return shortened if shortened.text.strip() else None


def _estimate_start(counts: list[int], limit: int) -> int:
    """How many leading parts could possibly fit, from per-part counts plus slack for
    the joins. Only a starting point: the exact count of the real block decides."""
    separator = count_tokens(PART_SEPARATOR)
    total, fitting = 0, 0
    for index, tokens in enumerate(counts):
        total += tokens + (separator if index else 0)
        if total > limit + 4 * (index + 1):
            break
        fitting = index + 1
    return fitting


def enforce_budget(
    parts: Iterable[CorpusPart] | CorpusContext, limit: int = MAX_CORPUS_TOKENS
) -> BudgetResult:
    """Fit ``parts`` (best first) into ``limit`` corpus tokens.

    1. Blank parts and exact repeats are ignored.
    2. While the block is over the limit, the lowest-ranked WHOLE part is dropped.
    3. If one part remains and is still over, it is truncated by tokens.
    4. A part whose label alone does not fit is dropped.

    The returned block always satisfies ``count_tokens(block) <= limit`` and
    ``limit <= MAX_CORPUS_TOKENS``; ``limit`` above the ceiling is refused.
    """
    limit = _validated_limit(limit)
    candidates = _candidates(parts)
    if not candidates or limit == 0:
        return _result((), limit, dropped=candidates)

    counts = [count_tokens(p.render()) for p in candidates]
    kept_count = len(candidates)
    start = _estimate_start(counts, limit)
    if start < len(candidates):
        kept_count = max(start, 1)  # everything past the estimate is certainly too much
    kept = candidates[:kept_count]
    while len(kept) > 1 and count_tokens(render_block(kept)) > limit:
        kept = kept[:-1]

    truncated: list[CorpusPart] = []
    if count_tokens(render_block(kept)) > limit:  # a single part, still too big
        shortened = _truncate(kept[0], limit)
        if shortened is None:
            kept = []
        else:
            kept = [shortened]
            truncated = [shortened]
    dropped = candidates[len(kept):] if not truncated else candidates[1:]
    if not kept:
        dropped = candidates
    return _result(tuple(kept), limit, dropped=dropped, truncated=truncated)


def _result(
    kept: tuple[CorpusPart, ...], limit: int, *, dropped=(), truncated=()
) -> BudgetResult:
    block = render_block(kept)
    tokens = count_tokens(block)
    if tokens > limit:  # cannot happen; refuse to return anything over the limit
        raise AssertionError("enforce_budget produced a block over the limit")

    def sources(items) -> tuple[Source, ...]:
        return tuple(Source(p.doc_id, p.locator) for p in items)

    return BudgetResult(
        parts=kept,
        block=block,
        tokens=tokens,
        limit=limit,
        kept=sources(kept),
        dropped=sources(reversed(list(dropped))),  # lowest rank first: the order they were dropped in
        truncated=sources(truncated),
    )
