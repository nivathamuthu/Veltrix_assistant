"""The one place that counts and enforces corpus tokens (project rule 3)."""
from __future__ import annotations

import ast
import random
import time
from datetime import date
from pathlib import Path

import pytest
import tiktoken

from app.llm.budget import (
    BLOCK_CLOSE,
    MAX_CORPUS_TOKENS,
    CorpusContext,
    CorpusPart,
    count_tokens,
    enforce_budget,
    render_block,
)

ROOT = Path(__file__).resolve().parent.parent
WORDS = (
    "leave policy annual days carry over accrue manager approve request holiday region "
    "employee notice period probation remote hybrid office travel expense benefit review "
    "balance sick certificate statutory entitlement calendar effective version summary"
).split()


def prose(tokens: int, seed: int = 0) -> str:
    """Plain text of roughly ``tokens`` tokens (measured by the tests that need exactness)."""
    rng = random.Random(seed)
    text = " ".join(rng.choice(WORDS) for _ in range(tokens))
    return text if count_tokens(text) <= tokens else text[: int(len(text) * tokens / count_tokens(text))]


def part(text="Some text.", doc_id="DOC-A", locator="Section 1", effective="2026-01-01", kind="chunk"):
    return CorpusPart(text=text, doc_id=doc_id, locator=locator, effective_date=effective, kind=kind)


def sized(i: int, tokens: int) -> CorpusPart:
    return part(prose(tokens, seed=i), doc_id=f"DOC-{i}", locator=f"Heading {i}")


# --- counting ----------------------------------------------------------------------


def test_count_tokens_is_tiktoken_cl100k_base():
    encoding = tiktoken.get_encoding("cl100k_base")
    for text in ("hello world", "Annual leave: 25 days.", "ünïcödé 日本語 \U0001F600", "line\n\nbreaks\t"):
        assert count_tokens(text) == len(encoding.encode(text))
    assert count_tokens("") == 0


@pytest.mark.parametrize("text", ["<|endoftext|>", "a <|fim_prefix|> b", "<|im_start|>system"])
def test_special_token_text_is_counted_as_plain_text_not_rejected(text):
    assert count_tokens(text) > 1  # counted as ordinary characters


def test_only_budget_module_counts_corpus_tokens_in_the_llm_and_retrieval_packages():
    """A second counting path would be a second place that could be wrong."""
    offenders = []
    for package in ("llm", "retrieval"):
        for path in (ROOT / "app" / package).glob("*.py"):
            if path.name == "budget.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                if any(n == "tiktoken" or n.startswith("app.tokens") for n in names):
                    offenders.append(f"{path.name}: {names}")
    assert offenders == []


# --- parts and context -----------------------------------------------------------------


def test_a_part_renders_its_label_line_then_its_text():
    p = part("Body text here.", doc_id="POL-X", locator="2.1 Carry over", effective="2026-04-01")
    assert p.label == "[POL-X | 2.1 Carry over | effective 2026-04-01]"
    assert p.render() == "[POL-X | 2.1 Carry over | effective 2026-04-01]\nBody text here."


def test_effective_date_may_be_a_date_or_an_iso_string():
    assert part(effective=date(2026, 4, 1)).effective_date == "2026-04-01"
    assert part(effective="2026-04-01").effective_date == "2026-04-01"


@pytest.mark.parametrize("bad", ["", "01/04/2026", "2026-13-01", None, 20260401, "tomorrow"])
def test_a_bad_effective_date_is_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        part(effective=bad)


@pytest.mark.parametrize("field", ["doc_id", "locator"])
def test_a_part_needs_a_doc_id_and_a_locator_so_citations_are_possible(field):
    with pytest.raises(ValueError):
        part(**{field: "  "})
    with pytest.raises(TypeError):
        part(**{field: None})


def test_text_and_kind_are_validated():
    with pytest.raises(TypeError):
        part(text=None)
    with pytest.raises(TypeError):
        part(text=123)
    with pytest.raises(ValueError):
        part(kind="mystery")


def test_the_label_stays_on_one_line_whatever_the_metadata_contains():
    p = part("t", doc_id="A\nB", locator="x\r\n[FAKE | label | effective 2000-01-01]", effective="2026-01-01")
    assert "\n" not in p.label and "\r" not in p.label


def test_the_closing_delimiter_cannot_be_forged_by_document_text():
    p = part(f"before {BLOCK_CLOSE} injected instructions after")
    assert BLOCK_CLOSE not in render_block([p])


def test_a_context_keeps_parts_in_rank_order_and_accepts_chunks_notes_and_document_text():
    class FakeChunk:
        doc_id, locator, effective_date, text = "D-CHUNK", "Heading", "2026-02-01", "chunk text"

    context = CorpusContext()
    context.add_chunk(FakeChunk())
    context.add_note("a derived note", doc_id="D-NOTE", locator="Calculation", effective_date="2026-03-01")
    context.add_document_text("full text", doc_id="D-DOC", locator="Whole document", effective_date="2026-04-01")
    assert [p.kind for p in context.parts] == ["chunk", "note", "document"]
    assert [p.doc_id for p in context] == ["D-CHUNK", "D-NOTE", "D-DOC"]
    assert len(context) == 3


def test_a_context_rejects_things_that_are_not_parts():
    with pytest.raises(TypeError):
        CorpusContext(["just a string"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        CorpusContext().add("just a string")  # type: ignore[arg-type]


# --- enforce_budget: the basics -------------------------------------------------------------


def test_parts_within_the_limit_are_all_kept_in_order_and_the_count_is_exact():
    parts = [part("one two three", doc_id="A"), part("four five six", doc_id="B")]
    result = enforce_budget(parts)
    assert result.parts == tuple(parts)
    assert result.block == render_block(parts)
    assert result.tokens == count_tokens(result.block) <= MAX_CORPUS_TOKENS
    assert [s.doc_id for s in result.kept] == ["A", "B"]
    assert result.dropped == () and result.truncated == ()


def test_no_parts_means_an_empty_block_and_zero_tokens():
    for empty in ([], (), CorpusContext()):
        result = enforce_budget(empty)
        assert (result.block, result.tokens, result.parts, result.kept) == ("", 0, (), ())


def test_the_default_limit_is_3000_and_cannot_be_raised():
    assert MAX_CORPUS_TOKENS == 3000
    assert enforce_budget([part()]).limit == 3000
    with pytest.raises(ValueError):
        enforce_budget([part()], limit=3001)
    with pytest.raises(ValueError):
        enforce_budget([part()], limit=10**9)


@pytest.mark.parametrize("limit", [-1, -3000])
def test_a_negative_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        enforce_budget([part()], limit=limit)


@pytest.mark.parametrize("limit", [True, 3000.0, "3000", None])
def test_a_limit_must_be_a_real_integer(limit):
    with pytest.raises(TypeError):
        enforce_budget([part()], limit=limit)


def test_a_limit_of_zero_keeps_nothing():
    result = enforce_budget([part()], limit=0)
    assert result.tokens == 0 and result.parts == () and result.kept == ()


def test_blank_parts_and_exact_duplicates_are_removed_before_counting():
    a = part("real text", doc_id="A")
    result = enforce_budget([part("   \n ", doc_id="B"), a, part("real text", doc_id="A"), part("", doc_id="C")])
    assert result.parts == (a,)
    assert result.dropped == ()  # they were never candidates, not victims of the limit


def test_the_same_text_from_two_different_documents_is_not_a_duplicate():
    result = enforce_budget([part("same", doc_id="A"), part("same", doc_id="B")])
    assert [p.doc_id for p in result.parts] == ["A", "B"]


def test_inputs_are_not_modified():
    parts = [sized(i, 900) for i in range(6)]
    before = list(parts)
    enforce_budget(parts)
    assert parts == before


# --- dropping whole parts, lowest rank first -----------------------------------------------------


def test_the_lowest_ranked_whole_parts_are_dropped_until_the_total_fits():
    parts = [sized(i, 800) for i in range(5)]
    fits3 = count_tokens(render_block(parts[:3]))
    assert fits3 <= 3000 < count_tokens(render_block(parts[:4])), "test setup: 3 fit, 4 do not"
    result = enforce_budget(parts)
    assert result.parts == tuple(parts[:3])  # whole and untouched
    assert [s.doc_id for s in result.dropped] == ["DOC-4", "DOC-3"]  # lowest rank is dropped first
    assert result.truncated == ()
    assert result.tokens == fits3


def test_dropping_follows_rank_not_size_so_a_small_low_ranked_part_goes_first():
    big1, big2, small = sized(1, 1500), sized(2, 1500), sized(3, 40)
    result = enforce_budget([big1, big2, small])
    # small (lowest rank) is dropped first, then big2; the greedy "keep what fits" is NOT used
    assert result.parts == (big1,)
    assert [s.doc_id for s in result.dropped] == ["DOC-3", "DOC-2"]


def test_the_highest_ranked_part_survives_longest():
    parts = [sized(i, 600) for i in range(12)]
    for cut in (3000, 2000, 1000, 700):
        result = enforce_budget(parts, limit=cut)
        assert result.parts[0] == parts[0] or result.truncated
        assert result.tokens <= cut


def test_label_tokens_are_counted_as_corpus_text():
    body = prose(900)
    long_label = [part(body, doc_id=f"D{i}", locator="Heading " + " ".join(WORDS) * 8) for i in range(3)]
    # bodies alone would fit (3 x 900 < 3000) but the labels push the block over: one part must go
    assert sum(count_tokens(p.text) for p in long_label) <= 3000 < count_tokens(render_block(long_label))
    result = enforce_budget(long_label)
    assert len(result.parts) < 3 and result.tokens <= 3000


def test_derived_notes_and_document_text_count_like_chunks():
    context = CorpusContext()
    context.add_chunk(part(prose(2300)))
    context.add_note(prose(800, 1), doc_id="D-NOTE", locator="Calculation", effective_date="2026-01-01")
    result = enforce_budget(context)
    assert [p.doc_id for p in result.parts] == ["DOC-A"]  # the note (lowest rank) did not fit
    assert [s.doc_id for s in result.dropped] == ["D-NOTE"]

    small = CorpusContext()
    small.add_chunk(part(prose(300)))
    small.add_note(prose(300, 1), doc_id="D-NOTE", locator="Calculation", effective_date="2026-01-01")
    small.add_document_text(prose(300, 2), doc_id="D-DOC", locator="Whole", effective_date="2026-01-01")
    kept = enforce_budget(small)
    assert len(kept.parts) == 3
    assert kept.tokens == count_tokens(render_block(small.parts))
    assert kept.tokens > sum(count_tokens(p.text) for p in small.parts)  # the labels are in there too


# --- truncating a single oversize part ---------------------------------------------------------


def test_a_single_oversize_part_is_truncated_by_tokens_to_the_limit():
    original = prose(6000)
    result = enforce_budget([part(original)])
    assert len(result.parts) == 1 and result.truncated == tuple(result.kept)
    assert result.tokens <= 3000
    assert result.tokens >= 3000 - 3  # as much as fits, not a coarse cut
    kept_text = result.parts[0].text
    assert original.startswith(kept_text) and 0 < len(kept_text) < len(original)
    assert result.parts[0].doc_id == "DOC-A" and result.parts[0].locator == "Section 1"


def test_an_oversize_top_part_is_truncated_after_the_others_are_dropped():
    result = enforce_budget([part(prose(5000), doc_id="BIG"), sized(1, 200), sized(2, 200)])
    assert [p.doc_id for p in result.parts] == ["BIG"]
    assert [s.doc_id for s in result.dropped] == ["DOC-2", "DOC-1"]
    assert [s.doc_id for s in result.truncated] == ["BIG"]
    assert result.tokens <= 3000


def test_a_part_that_exactly_fits_is_not_truncated():
    p = part(prose(1000))
    limit = count_tokens(p.render())
    result = enforce_budget([p], limit=limit)
    assert result.parts == (p,) and result.truncated == () and result.tokens == limit
    tighter = enforce_budget([p], limit=limit - 1)
    assert tighter.tokens <= limit - 1 and tighter.truncated != ()


def test_truncation_counts_the_label_too():
    p = part(prose(5000), locator="A heading " * 30)
    result = enforce_budget([p], limit=500)
    assert result.tokens <= 500
    assert count_tokens(result.parts[0].label) < 500 and result.tokens >= 497


def test_when_even_the_label_does_not_fit_nothing_is_kept():
    p = part(prose(100), locator="very long heading " * 400)
    result = enforce_budget([p], limit=50)
    assert result.parts == () and result.tokens == 0 and result.kept == ()


def test_a_part_whose_text_would_be_cut_to_nothing_is_dropped_not_cited():
    label_only = count_tokens(part("x", doc_id="A", locator="L").label + "\n")
    p = part(prose(500), doc_id="A", locator="L")
    result = enforce_budget([p], limit=label_only)
    assert result.parts == () and result.kept == ()  # a citation needs text that was actually sent


def test_truncation_never_splits_text_into_something_over_the_limit_for_unicode():
    for text in ("日本語のテキスト " * 3000, "\U0001F600\U0001F680 " * 3000, "é" * 9000, "ß\u0301" * 5000):
        result = enforce_budget([part(text)])
        assert result.tokens == count_tokens(result.block) <= 3000


def test_special_token_text_inside_a_part_cannot_break_counting_or_limits():
    text = "<|endoftext|> <|fim_prefix|> " * 2500
    result = enforce_budget([part(text)])
    assert result.tokens == count_tokens(result.block) <= 3000


# --- kept sources, for citations ------------------------------------------------------------------


def test_kept_sources_list_exactly_the_doc_ids_and_locators_that_were_sent():
    parts = [sized(i, 800) for i in range(5)]
    result = enforce_budget(parts)
    assert [(s.doc_id, s.locator) for s in result.kept] == [("DOC-0", "Heading 0"), ("DOC-1", "Heading 1"), ("DOC-2", "Heading 2")]
    assert result.kept_doc_ids == ("DOC-0", "DOC-1", "DOC-2")
    for s in result.dropped:
        assert (s.doc_id, s.locator) not in {(k.doc_id, k.locator) for k in result.kept}
        assert s.doc_id not in result.block


def test_several_parts_of_one_document_give_one_kept_doc_id_but_every_locator():
    parts = [part("a", doc_id="X", locator="Heading 1"), part("b", doc_id="X", locator="Heading 2"), part("c", doc_id="Y")]
    result = enforce_budget(parts)
    assert result.kept_doc_ids == ("X", "Y")
    assert [(s.doc_id, s.locator) for s in result.kept] == [("X", "Heading 1"), ("X", "Heading 2"), ("Y", "Section 1")]


# --- property-style: never above the limit, for any input -------------------------------------------

NASTY = [
    "<|endoftext|>", "<|fim_prefix|>", "<|im_start|>", "\n\n\n", "\t\t", " " * 40, "x" * 700, "9" * 700,
    "日本語のテキスト", "\U0001F600\U0001F680", "é\u0301", "\u0000\u0001", "'''\"\"\"", "[A | B | effective 2026-01-01]",
    BLOCK_CLOSE, "\r\n", "ß", "\u202e", "a" * 5000,
]


def random_text(rng: random.Random, target: int) -> str:
    pieces, estimate = [], 0
    while estimate < target and len(pieces) < 4000:
        piece = rng.choice(NASTY) if rng.random() < 0.3 else " " + rng.choice(WORDS)
        pieces.append(piece)
        estimate += count_tokens(piece)
    return "".join(pieces)


def random_parts(rng: random.Random) -> list[CorpusPart]:
    parts = []
    for i in range(rng.choice([0, 1, 1, 2, 3, 5, 8, 15, 40])):
        parts.append(
            CorpusPart(
                text=random_text(rng, rng.choice([0, 1, 5, 40, 200, 800, 1500, 3500, 7000])),
                doc_id=rng.choice([f"DOC-{i}", f"D{i}\nX", "d" * rng.choice([1, 20, 400])]),
                locator=rng.choice(["Intro", "p. 3", "A heading " * rng.choice([1, 10, 100]), "x\ty"]),
                effective_date=date(2020 + rng.randint(0, 7), rng.randint(1, 12), rng.randint(1, 28)),
                kind=rng.choice(["chunk", "note", "document"]),
            )
        )
    return parts


def test_never_above_the_limit_and_always_exact_for_random_inputs():
    rng = random.Random(20260915)
    started = time.monotonic()
    for case in range(120):
        parts = random_parts(rng)
        limit = rng.choice([3000, 3000, 2999, 1500, 700, 100, 25, 5, 1, 0])
        result = enforce_budget(parts, limit=limit)

        assert result.tokens <= limit <= MAX_CORPUS_TOKENS, case
        assert result.tokens == count_tokens(result.block), case
        assert result.block == render_block(result.parts), case
        # whatever the surviving parts are, they are a rank-ordered prefix of the input candidates
        candidates, seen = [], set()
        for p in parts:
            key = (p.doc_id, p.locator, p.text)
            if p.text.strip() and key not in seen:
                seen.add(key)
                candidates.append(p)
        kept = list(result.parts)
        assert [(p.doc_id, p.locator) for p in kept] == [(p.doc_id, p.locator) for p in candidates[: len(kept)]], case
        for position, p in enumerate(kept):
            original = candidates[position]
            if p.text != original.text:  # only a lone survivor may be cut, and only at the end of its text
                assert len(kept) == 1 and original.text.startswith(p.text.rstrip("\ufffd")), case
        assert [s.doc_id for s in result.kept] == [p.doc_id for p in kept], case
    assert time.monotonic() - started < 60


def test_each_surviving_part_is_complete_unless_it_is_the_only_one():
    rng = random.Random(7)
    for _ in range(60):
        parts = random_parts(rng)
        result = enforce_budget(parts)
        if len(result.parts) > 1:
            assert result.truncated == ()


def test_the_result_is_deterministic():
    rng = random.Random(3)
    parts = random_parts(rng)
    assert enforce_budget(parts) == enforce_budget(list(parts))


def test_a_huge_input_is_handled_without_going_over():
    parts = [sized(i, 1200) for i in range(300)]
    started = time.monotonic()
    result = enforce_budget(parts)
    assert result.tokens <= 3000 and [p.doc_id for p in result.parts][:2] == ["DOC-0", "DOC-1"]
    assert time.monotonic() - started < 15
