"""Chunking rules (Phase 3b). All input is synthetic."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import (
    HARD_MAX_TOKENS,
    NEARLY_EMPTY_TOKENS,
    OVERLAP_TOKENS,
    TARGET_MAX_TOKENS,
    TARGET_MIN_TOKENS,
    _tail_overlap,
    chunk_document,
)
from app.ingestion.manifest import ManifestEntry
from app.ingestion.parsers.models import ParsedDocument, Section
from app.tokens import count_tokens

ENTRY = ManifestEntry(
    doc_id="SYN-DOC", file="syn.pdf", title="Synthetic Document", doc_type="addendum",
    regions=("IN", "US"), effective_date="2026-03-01", access_level="managers",
)


def doc(*sections: Section) -> ParsedDocument:
    return ParsedDocument(path=Path("syn.pdf"), parser="pdf", sections=list(sections))


def prose(sentence_count: int, tag: str = "a") -> str:
    """Distinct sentences (~14 tokens each) so overlap and splits are traceable."""
    return " ".join(
        f"Rule {tag}{i} says that staff in group {i} must record item {i * 7} before Friday."
        for i in range(sentence_count)
    )


def text_section(heading, body, page=1) -> Section:
    return Section(heading=heading, page=page, text=body, kind="text")


def table_rows(n: int) -> list[str]:
    return [
        f"Row: Tenure {i}-{i + 1} years | Accrual {1 + i / 10:.1f} days/month | "
        f"Note band {i} applies after confirmation"
        for i in range(n)
    ]


def table_section(heading, rows, page=2) -> Section:
    return Section(heading=heading, page=page, text="\n".join(rows), kind="table")


# --- sizes ---------------------------------------------------------------


def test_long_section_is_split_within_the_size_limits():
    chunks = chunk_document(ENTRY, doc(text_section("Long Rules", prose(120))))
    assert len(chunks) > 3
    for c in chunks:
        assert c.token_count == count_tokens(c.text)
        assert c.token_count <= TARGET_MAX_TOKENS <= HARD_MAX_TOKENS
    # balanced split: no near-empty tail
    assert min(c.token_count for c in chunks) >= TARGET_MIN_TOKENS * 0.6


def test_hard_maximum_holds_even_for_text_without_any_boundaries():
    blob = "x" * 6000  # no spaces, no sentences
    chunks = chunk_document(ENTRY, doc(text_section("Blob", blob)))
    assert len(chunks) >= 2
    assert all(c.token_count <= HARD_MAX_TOKENS for c in chunks)


def test_document_text_that_looks_like_a_tokenizer_special_token_is_just_text():
    chunks = chunk_document(ENTRY, doc(text_section("Odd", "Keep <|endoftext|> as plain text.")))
    assert "<|endoftext|>" in chunks[0].text


# --- headings and locators -----------------------------------------------


def test_heading_is_part_of_the_chunk_text_and_is_the_locator():
    chunks = chunk_document(
        ENTRY, doc(text_section("5. Sick leave", "Sick leave needs a certificate after 3 days."))
    )
    assert chunks[0].text.startswith("5. Sick leave\n\nSick leave needs a certificate")
    assert chunks[0].locator == "5. Sick leave"


def test_section_without_heading_uses_the_page_as_locator():
    chunks = chunk_document(ENTRY, doc(text_section(None, "Plain paragraph on a page.", page=3)))
    assert chunks[0].locator == "p. 3"
    assert chunks[0].text == "Plain paragraph on a page."


def test_every_piece_of_a_split_section_keeps_the_heading():
    chunks = chunk_document(ENTRY, doc(text_section("Long Rules", prose(120))))
    assert all(c.text.startswith("Long Rules\n\n") for c in chunks)
    assert all(c.locator == "Long Rules" for c in chunks)


def test_locator_falls_back_to_the_title_when_there_is_no_heading_or_page():
    chunks = chunk_document(ENTRY, doc(text_section(None, "Floating text.", page=None)))
    assert chunks[0].locator == ENTRY.title


# --- overlap ----------------------------------------------------------------


def test_consecutive_pieces_of_one_section_overlap_by_about_30_tokens():
    chunks = chunk_document(ENTRY, doc(text_section("Long Rules", prose(120))))
    for earlier, later in zip(chunks, chunks[1:]):
        earlier_body = earlier.text.split("\n\n", 1)[1]
        later_body = later.text.split("\n\n", 1)[1]
        tail = _tail_overlap(earlier_body, OVERLAP_TOKENS)
        assert later_body.startswith(tail)
        assert OVERLAP_TOKENS - 12 <= count_tokens(tail) <= OVERLAP_TOKENS


def test_different_sections_never_overlap():
    first = text_section("Section One", prose(60, "a"))
    second = text_section("Section Two", prose(60, "b"))
    chunks = chunk_document(ENTRY, doc(first, second))
    one = [c for c in chunks if c.locator == "Section One"]
    two = [c for c in chunks if c.locator == "Section Two"]
    assert one and two
    last_of_one = one[-1].text.split("\n\n", 1)[1]
    assert not two[0].text.split("\n\n", 1)[1].startswith(_tail_overlap(last_of_one, OVERLAP_TOKENS))
    assert "Rule b0 " not in one[-1].text and "Rule a59 " not in two[0].text


def test_a_small_section_has_no_overlap_text_at_all():
    chunks = chunk_document(ENTRY, doc(text_section("Tiny", "One short sentence.")))
    assert chunks[0].text == "Tiny\n\nOne short sentence."


# --- tables ---------------------------------------------------------------


def test_small_table_stays_in_one_chunk():
    rows = table_rows(6)
    chunks = chunk_document(ENTRY, doc(table_section("Accrual Schedule", rows)))
    assert len(chunks) == 1
    assert chunks[0].text == "Accrual Schedule\n\n" + "\n".join(rows)


def test_large_table_splits_between_rows_and_repeats_the_heading():
    rows = table_rows(60)
    assert count_tokens("\n".join(rows)) > TARGET_MAX_TOKENS * 2
    chunks = chunk_document(ENTRY, doc(table_section("Accrual Schedule", rows)))
    assert len(chunks) >= 2

    seen: list[str] = []
    for c in chunks:
        assert c.text.startswith("Accrual Schedule\n\n")  # header context in every piece
        assert c.token_count <= TARGET_MAX_TOKENS
        lines = c.text.split("\n\n", 1)[1].split("\n")
        assert all(line in rows for line in lines)  # only whole, unmodified rows
        seen.extend(lines)
    assert seen == rows  # every row exactly once, in order: nothing cut, nothing lost


def test_table_pieces_carry_no_overlap():
    rows = table_rows(60)
    chunks = chunk_document(ENTRY, doc(table_section("Accrual Schedule", rows)))
    all_lines = [ln for c in chunks for ln in c.text.split("\n\n", 1)[1].split("\n")]
    assert len(all_lines) == len(set(all_lines))


# --- merging ------------------------------------------------------------


def tiny_sections(n: int, prefix: str = "Q") -> list[Section]:
    return [
        text_section(f"{prefix}{i}. Question number {i}?", f"Answer {i}: ask your manager about item {i}.")
        for i in range(n)
    ]


def test_tiny_neighbouring_sections_are_merged():
    sections = tiny_sections(20)
    chunks = chunk_document(ENTRY, doc(*sections))
    assert len(chunks) < len(sections) / 3
    for c in chunks:
        assert c.token_count <= TARGET_MAX_TOKENS
    assert all(c.token_count >= NEARLY_EMPTY_TOKENS for c in chunks)
    # nothing is lost by merging
    joined = "\n".join(c.text for c in chunks)
    assert all(f"Q{i}. Question number {i}?" in joined for i in range(20))


def test_merged_chunk_remembers_where_each_section_sits():
    chunks = chunk_document(ENTRY, doc(*tiny_sections(20)))
    merged = chunks[0]
    assert len(merged.segments) > 1
    assert merged.locator == merged.segments[0].locator == "Q0. Question number 0?"
    for seg in merged.segments:
        assert seg.locator in merged.text[seg.start : seg.end]


def test_nothing_is_merged_across_a_table():
    before = text_section("Before", "Intro text before the table.")
    table = table_section("Schedule", table_rows(3))
    after = text_section("After", "Text after the table.")
    chunks = chunk_document(ENTRY, doc(before, table, after))
    assert len(chunks) == 3
    assert "Row:" not in chunks[0].text and "Row:" not in chunks[2].text
    assert chunks[1].text.startswith("Schedule\n\nRow:")
    assert "Intro text" not in chunks[1].text and "Text after" not in chunks[1].text


def test_two_adjacent_tables_stay_separate():
    chunks = chunk_document(
        ENTRY, doc(table_section("T1", table_rows(2)), table_section("T2", table_rows(2)))
    )
    assert [c.locator for c in chunks] == ["T1", "T2"]


def test_nothing_is_merged_across_documents(tmp_path):
    from app.ingestion.corpus import load_corpus
    from tests.test_corpus_loading import write_corpus

    root = write_corpus(
        tmp_path,
        {
            "one.md": ("# One\n\nTiny text one.\n", {"doc_id": "D-ONE"}),
            "two.md": ("# Two\n\nTiny text two.\n", {"doc_id": "D-TWO"}),
        },
    )
    chunks = load_corpus(root / "corpus", root / "manifest.json").chunks
    assert len(chunks) == 2
    assert {c.doc_id for c in chunks} == {"D-ONE", "D-TWO"}
    assert all(("one" in c.text) != ("two" in c.text) for c in chunks)


def test_empty_sections_produce_no_chunk():
    assert chunk_document(ENTRY, doc(text_section(None, "   "), text_section("", ""))) == []


# --- metadata and ids ------------------------------------------------------


def test_manifest_metadata_is_copied_onto_every_chunk():
    chunks = chunk_document(
        ENTRY, doc(text_section("A", prose(80)), table_section("T", table_rows(3)))
    )
    assert len(chunks) > 2
    for c in chunks:
        assert c.doc_id == "SYN-DOC"
        assert c.regions == ("IN", "US")
        assert c.effective_date == "2026-03-01"
        assert c.doc_type == "addendum"
        assert c.access_level == "managers"


def test_chunk_ids_are_deterministic_unique_and_ordered():
    sections = [text_section("A", prose(80)), table_section("T", table_rows(40)), *tiny_sections(5)]
    first = chunk_document(ENTRY, doc(*sections))
    second = chunk_document(ENTRY, doc(*sections))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
    ids = [c.chunk_id for c in first]
    assert len(set(ids)) == len(ids)
    assert ids == [f"SYN-DOC::{i:03d}" for i in range(1, len(ids) + 1)]


def test_chunk_ids_do_not_depend_on_other_documents():
    other = ManifestEntry(**{**ENTRY.__dict__, "doc_id": "OTHER"})
    a = chunk_document(ENTRY, doc(text_section("A", "Same text.")))
    b = chunk_document(other, doc(text_section("A", "Same text.")))
    assert a[0].chunk_id == "SYN-DOC::001" and b[0].chunk_id == "OTHER::001"


@pytest.mark.parametrize("count", [1, 5, 25])
def test_no_chunk_is_ever_empty(count):
    chunks = chunk_document(ENTRY, doc(*tiny_sections(count)))
    assert all(c.text.strip() and c.token_count > 0 for c in chunks)
