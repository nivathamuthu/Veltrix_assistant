"""Turn a ParsedDocument into chunks.

Sizes are measured with tiktoken cl100k_base (app/tokens.py):

* target 200-350 tokens per chunk, hard maximum 450;
* the section heading is part of the chunk text ("Heading\\n\\ntext") so FAQ
  questions and headings are searchable;
* a section that does not fit is split on sentence boundaries into balanced
  pieces; consecutive pieces of the SAME section overlap by ~30 tokens (and
  nothing else overlaps);
* a table is never split inside a row: a small table stays whole, a large one
  is split between rows with the heading repeated in every piece (each row
  already repeats its column names, see parsers/tables.py);
* tiny neighbouring prose sections of one document are merged so no chunk is
  nearly empty; merging never crosses a table or a document;
* all manifest metadata is copied onto every chunk of the document.

Everything here is deterministic: the same input gives the same chunk_ids and
the same text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.db.models import Chunk, Segment
from app.ingestion.manifest import ManifestEntry
from app.ingestion.parsers.models import ParsedDocument, Section
from app.logging_setup import get_logger
from app.tokens import count_tokens, decode, encode

log = get_logger(__name__)

TARGET_MIN_TOKENS = 200
TARGET_MAX_TOKENS = 350
HARD_MAX_TOKENS = 450
OVERLAP_TOKENS = 30
# A chunk below this is "nearly empty" and gets merged into a neighbour
# (when that does not break the hard maximum, a table boundary or a document).
NEARLY_EMPTY_TOKENS = 80
# Slack for separators/rounding when packing units by their individual counts.
_PACK_SAFETY_TOKENS = 8
_MIN_PIECE_LIMIT = 40

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\u201c\u2022])")


@dataclass(frozen=True)
class _Piece:
    """One section, or one split-off part of a large section."""

    heading: str | None
    page: int | None
    body: str
    section_idx: int
    kind: str  # "text" or "table"


def chunk_id_for(doc_id: str, ordinal: int) -> str:
    """Deterministic: the same document structure always yields the same ids."""
    return f"{doc_id}::{ordinal:03d}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _locator(piece: _Piece, fallback: str) -> str:
    if piece.heading:
        return piece.heading
    if piece.page is not None:
        return f"p. {piece.page}"
    return fallback


def _render(pieces: list[_Piece], fallback: str) -> tuple[str, list[Segment]]:
    """Chunk text plus where each piece's locator applies inside it."""
    parts: list[str] = []
    segments: list[Segment] = []
    position = 0
    for i, piece in enumerate(pieces):
        same_section_as_previous = i > 0 and pieces[i - 1].section_idx == piece.section_idx
        if piece.heading and not same_section_as_previous:
            block = f"{piece.heading}\n\n{piece.body}" if piece.body else piece.heading
        else:
            block = piece.body
        if parts:
            position += 2  # the "\n\n" joining blocks
        segments.append(Segment(_locator(piece, fallback), position, position + len(block)))
        parts.append(block)
        position += len(block)
    return "\n\n".join(parts), segments


def _tokens_of(pieces: list[_Piece], fallback: str) -> int:
    return count_tokens(_render(pieces, fallback)[0])


# --------------------------------------------------------------------------- #
# Splitting one section
# --------------------------------------------------------------------------- #


def _split_words_to_limit(text: str, limit: int) -> list[str]:
    """Break text that has no usable boundary into <= limit-token parts."""
    ids = encode(text)
    return [decode(ids[i : i + limit]) for i in range(0, len(ids), limit)]


def _prose_units(text: str, limit: int) -> list[tuple[str, str]]:
    """``(separator_before, unit)`` pairs; every unit is <= limit tokens."""
    units: list[tuple[str, str]] = []
    for paragraph in (p.strip() for p in text.split("\n")):
        if not paragraph:
            continue
        first = True
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            fragments = (
                [sentence]
                if count_tokens(sentence) <= limit
                else _split_words_to_limit(sentence, limit)
            )
            for fragment in fragments:
                sep = "" if not units else ("\n" if first else " ")
                units.append((sep, fragment))
                first = False
    return units


def _table_units(text: str, limit: int, doc_id: str) -> list[tuple[str, str]]:
    """Rows are the units. A row longer than ``limit`` cannot be kept whole."""
    units: list[tuple[str, str]] = []
    for row in (r.strip() for r in text.split("\n")):
        if not row:
            continue
        if count_tokens(row) <= limit:
            units.append(("" if not units else "\n", row))
        else:
            log.warning("doc_id=%s: a table row exceeds the chunk limit and is split", doc_id)
            for fragment in _split_words_to_limit(row, limit):
                units.append(("" if not units else "\n", fragment))
    return units


def _pack(units: list[tuple[str, str]], limit: int) -> list[list[tuple[str, str]]]:
    """Group units into balanced bins of at most ``limit`` tokens each."""
    sizes = [count_tokens(unit) for _, unit in units]
    bins_needed = max(1, -(-sum(sizes) // limit))
    while True:
        bins = _pack_into(units, sizes, limit, bins_needed)
        # Units have a size of their own, so the arithmetic minimum can be one
        # bin short; try one more rather than leave a tiny leftover bin.
        if len(bins) <= bins_needed or bins_needed >= len(units):
            return bins
        bins_needed += 1


def _pack_into(
    units: list[tuple[str, str]], sizes: list[int], limit: int, bin_count: int
) -> list[list[tuple[str, str]]]:
    remaining_tokens = sum(sizes)
    remaining_bins = bin_count
    bins: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    used = 0
    for unit, size in zip(units, sizes):
        # The target is recomputed from what is LEFT, so rounding in earlier
        # bins cannot pile up as a tiny last bin.
        target = remaining_tokens / remaining_bins
        closes = used + size > limit or (
            remaining_bins > 1 and (used + size - target) > (target - used)
        )
        if current and closes:
            bins.append(current)
            remaining_tokens -= used
            remaining_bins = max(1, remaining_bins - 1)
            current, used = [], 0
        current.append(unit)
        used += size
    if current:
        bins.append(current)
    return bins


def _join(units: list[tuple[str, str]]) -> str:
    out: list[str] = []
    for i, (sep, text) in enumerate(units):
        out.append(("" if i == 0 else (sep or " ")) + text)
    return "".join(out)


def _tail_overlap(body: str, max_tokens: int) -> str:
    """The last ~max_tokens tokens of ``body``, starting on a word boundary."""
    words = body.split()
    taken: list[str] = []
    for word in reversed(words):
        candidate = " ".join([word, *taken])
        if count_tokens(candidate) > max_tokens:
            break
        taken.insert(0, word)
    return " ".join(taken)


def _section_pieces(section: Section, idx: int, doc_id: str) -> list[_Piece]:
    heading = section.heading.strip() if section.heading else None
    text = section.text.strip()
    heading_tokens = count_tokens(heading + "\n\n") if heading else 0
    is_table = section.kind == "table"

    def make(body: str) -> _Piece:
        return _Piece(heading, section.page, body, idx, "table" if is_table else "text")

    if heading_tokens + count_tokens(text) <= TARGET_MAX_TOKENS:
        return [make(text)]

    if is_table:
        limit = max(_MIN_PIECE_LIMIT, TARGET_MAX_TOKENS - heading_tokens - _PACK_SAFETY_TOKENS)
        bins = _pack(_table_units(text, limit, doc_id), limit)
        return [make("\n".join(unit for _, unit in b)) for b in bins]

    limit = max(
        _MIN_PIECE_LIMIT,
        TARGET_MAX_TOKENS - heading_tokens - OVERLAP_TOKENS - _PACK_SAFETY_TOKENS,
    )
    bins = _pack(_prose_units(text, limit), limit)
    own_bodies = [_join(b) for b in bins]
    pieces: list[_Piece] = []
    for i, own in enumerate(own_bodies):
        if i == 0:
            pieces.append(make(own))
        else:
            overlap = _tail_overlap(own_bodies[i - 1], OVERLAP_TOKENS)
            pieces.append(make(f"{overlap} {own}" if overlap else own))
    return pieces


# --------------------------------------------------------------------------- #
# Merging tiny neighbours (prose only, within one document, never across a table)
# --------------------------------------------------------------------------- #


def _can_join(left: list[_Piece], right: list[_Piece]) -> bool:
    # Pieces of one split section are never glued back together: their overlap
    # would duplicate text and the split was made for size.
    return left[-1].section_idx != right[0].section_idx


def _merge_run(pieces: list[_Piece], fallback: str) -> list[list[_Piece]]:
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    for piece in pieces:
        if not current:
            current = [piece]
            continue
        if (
            _can_join(current, [piece])
            and _tokens_of(current, fallback) < TARGET_MIN_TOKENS
            and _tokens_of([*current, piece], fallback) <= TARGET_MAX_TOKENS
        ):
            current.append(piece)
        else:
            groups.append(current)
            current = [piece]
    if current:
        groups.append(current)

    # Second pass: fold any nearly-empty group into the smaller neighbour that
    # can take it without exceeding the hard maximum.
    i = 0
    while i < len(groups):
        group = groups[i]
        if _tokens_of(group, fallback) < NEARLY_EMPTY_TOKENS:
            options: list[tuple[int, int]] = []  # (merged tokens, neighbour index)
            if i > 0 and _can_join(groups[i - 1], group):
                merged = _tokens_of([*groups[i - 1], *group], fallback)
                if merged <= HARD_MAX_TOKENS:
                    options.append((merged, i - 1))
            if i + 1 < len(groups) and _can_join(group, groups[i + 1]):
                merged = _tokens_of([*group, *groups[i + 1]], fallback)
                if merged <= HARD_MAX_TOKENS:
                    options.append((merged, i + 1))
            if options:
                _, neighbour = min(options)
                if neighbour < i:
                    groups[neighbour] = [*groups[neighbour], *group]
                else:
                    groups[neighbour] = [*group, *groups[neighbour]]
                del groups[i]
                i = max(0, min(i, neighbour))
                continue
        i += 1
    return groups


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _blocks(sections: list[Section], doc_id: str) -> list[list[list[_Piece]]]:
    """Document order -> list of blocks; each block is a list of piece groups.

    A table section is its own block (never merged with anything). Consecutive
    text sections form one block that may be merged internally.
    """
    blocks: list[list[list[_Piece]]] = []
    run: list[_Piece] = []

    def flush_run() -> None:
        nonlocal run
        if run:
            blocks.append(_merge_run(run, doc_id))
            run = []

    for idx, section in enumerate(sections):
        if not section.text.strip() and not (section.heading or "").strip():
            continue
        pieces = _section_pieces(section, idx, doc_id)
        if section.kind == "table":
            flush_run()
            blocks.append([[p] for p in pieces])
        else:
            run.extend(pieces)
    flush_run()
    return blocks


def chunk_document(entry: ManifestEntry, doc: ParsedDocument) -> list[Chunk]:
    """Chunks for one parsed document, with metadata copied from ``entry``."""
    chunks: list[Chunk] = []
    fallback = entry.title
    ordinal = 0

    def emit(text: str, segments: list[Segment]) -> None:
        nonlocal ordinal
        ordinal += 1
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(entry.doc_id, ordinal),
                doc_id=entry.doc_id,
                locator=segments[0].locator,
                regions=entry.regions,
                effective_date=entry.effective_date,
                doc_type=entry.doc_type,
                access_level=entry.access_level,
                text=text,
                token_count=count_tokens(text),
                segments=tuple(segments),
            )
        )

    for block in _blocks(doc.sections, entry.doc_id):
        for group in block:
            text, segments = _render(group, fallback)
            if count_tokens(text) <= HARD_MAX_TOKENS:
                emit(text, segments)
                continue
            # Last-resort guard (e.g. a huge heading): never exceed the hard max.
            log.warning("doc_id=%s: oversized chunk split at the token limit", entry.doc_id)
            # Headroom: re-encoding a decoded slice can differ by a token or two.
            for fragment in _split_words_to_limit(text, HARD_MAX_TOKENS - 10):
                emit(fragment, [Segment(segments[0].locator, 0, len(fragment))])
    return chunks
