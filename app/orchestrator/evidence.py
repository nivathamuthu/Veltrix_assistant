"""From retrieved chunks to resolved facts, in code (README B4, project rules 7 and 8).

The LLM only EXTRACTS: ``[{topic, value, doc_id, locator}]``. Everything that decides what the
answer says happens here, deterministically:

* ``build_context`` orders the chunks that go into the extraction prompt (current first, at
  most 3 upcoming, and only from documents newer than the reference date);
* ``build_claims`` turns the model's items into ``Claim`` objects whose document facts
  (type, regions, effective date, access level) come from the manifest metadata carried by the
  retrieved chunk, never from the model. A claim is dropped if its doc_id/locator was not in the
  set that was actually sent, or if its value cannot be found in that chunk's text (a value the
  excerpt does not contain is an invention);
* ``resolve_facts`` runs ``resolve_topic`` and ``upcoming_changes`` per topic (authority, then
  scope, then recency; no LLM judgement);
* ``fact_parts`` renders the resolved facts as corpus-note parts, so the composition call is
  budgeted like any other corpus text, and ``render_answer`` is the deterministic wording used
  when the model's composition does not follow the resolution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final, Iterable, Sequence

from app.llm.budget import KIND_NOTE, CorpusContext, CorpusPart
from app.llm.schema import parse_and_validate
from app.orchestrator.prompts import EXTRACTION_SCHEMA
from app.retrieval.pipeline import Retrieval, RetrievedChunk
from app.retrieval.policy_resolver import (
    STATUS_CONFLICT,
    STATUS_DECIDED,
    STATUS_NONE,
    Claim,
    Requester,
    resolve_topic,
    upcoming_changes,
)

MAX_UPCOMING_CHUNKS: Final = 3
CURRENT_FIRST: Final = 5  # current chunks placed before the upcoming ones in the prompt
CURRENT_K: Final = 12
UPCOMING_K: Final = 3
MAX_TOPIC_CHARS: Final = 80
MAX_VALUE_CHARS: Final = 120
MAX_NOT_COVERED: Final = 5

_NON_WORD: Final = re.compile(r"[^0-9a-z]+")
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")
_UNIT_PLURAL: Final = re.compile(r"\b(day|week|month|year|hour|holiday)s\b")
_DAYS_PER_MONTH: Final = re.compile(r"days?\s*/\s*month|day\s+per\s+month")
_DAYS_PER_YEAR: Final = re.compile(r"days?\s*/\s*year|day\s+per\s+year")
DROP_NOT_KEPT: Final = "not_in_kept_set"
DROP_VALUE: Final = "value_not_in_chunk_text"
_NUMBER_WORDS: Final = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve",
    "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty",
}
_CONFLICT_WORDS: Final = re.compile(r"conflict|contradict|disagree|inconsisten|differ", re.IGNORECASE)
_SETTLED_BY: Final = re.compile(r"settled by", re.IGNORECASE)
_BRACKETED_ID: Final = re.compile(r"\[[A-Za-z0-9][A-Za-z0-9._-]*(?:\s*\|[^\]]*)?\]")


# --------------------------------------------------------------------------- #
# What goes into the extraction prompt
# --------------------------------------------------------------------------- #


def prompt_chunks(retrieval: Retrieval, reference: date) -> list[RetrievedChunk]:
    """The chunks for the extraction call, best first (the budget drops the last ones first).

    Upcoming chunks are capped to 3 and kept ONLY if their document is newer than the
    reference date; they sit after the first current chunks so a tight budget does not lose
    them before the weakest current ones."""
    current = list(retrieval.current)
    upcoming = [c for c in retrieval.upcoming if c.effective_date > reference][:MAX_UPCOMING_CHUNKS]
    return current[:CURRENT_FIRST] + upcoming + current[CURRENT_FIRST:]


DOCUMENT_GROUP_LOCATOR: Final = "document"


def group_chunks_by_document(chunks: Iterable[RetrievedChunk]) -> list[list[RetrievedChunk]]:
    """Chunks of one document together, documents in first-seen (rank) order."""
    groups: dict[str, list[RetrievedChunk]] = {}
    order: list[str] = []
    for chunk in chunks:
        if chunk.doc_id not in groups:
            groups[chunk.doc_id] = []
            order.append(chunk.doc_id)
        groups[chunk.doc_id].append(chunk)
    return [groups[doc_id] for doc_id in order]


def corpus_of(chunks: Iterable[RetrievedChunk]) -> CorpusContext:
    """Corpus parts grouped by document, each group opening with doc_id and date."""
    context = CorpusContext()
    for items in group_chunks_by_document(chunks):
        first = items[0]
        context.add(
            CorpusPart(
                f"Document {first.doc_id}, effective {first.effective_date.isoformat()}. "
                "Excerpts from this document follow.",
                first.doc_id,
                DOCUMENT_GROUP_LOCATOR,
                first.effective_date,
                KIND_NOTE,
            )
        )
        for chunk in items:
            context.add_chunk(chunk)
    return context


# --------------------------------------------------------------------------- #
# Normalising what the model says
# --------------------------------------------------------------------------- #


def canon_topic(topic: str) -> str:
    return _NON_WORD.sub(" ", topic.casefold()).strip()


def canon_value(value: str) -> str:
    """The comparison form of a value: case, spacing and day/days style do not matter."""
    text = " ".join(value.casefold().split()).strip(" .;:\"'")
    return _UNIT_PLURAL.sub(r"\1", text)


_LEAVE_TOPIC: Final = re.compile(r"\b(annual leave|leave accrual|leave entitlement)\b")
_BAND_TAIL: Final = re.compile(
    r"\b(less than|more than|under|over|at least|\d+(?:\.\d+)?\s*(?:to|or)|2 to|5 or).*$"
)
_REGION_ONLY: Final = re.compile(r"^(my |your |the )?(region|employees in my region)$")
_UNIT: Final = (
    r"days?\s+earned\s+per\s+(?:full\s+)?(?:month|year)|"
    r"days?\s+per\s+(?:month|year)|"
    r"days?\s*/\s*(?:month|year)"
)
_RATE: Final = re.compile(
    rf"(?:(?P<n1>\d+(?:\.\d+)?)\s*(?:{_UNIT})|(?:{_UNIT})\s*(?P<n2>\d+(?:\.\d+)?))",
    re.IGNORECASE,
)
_META_ROW: Final = re.compile(
    r"^(document id|version|effective date|applies to|owner|classification)\b", re.I
)
_OTHER_LEAVE: Final = re.compile(
    r"\b(carry[- ]forward|wellbeing|sick leave|parental|bereavement|medical certificate)\b", re.I
)
_CARRY_HINT: Final = re.compile(r"carr(?:y|ied)(?:\s+forward|\s+over)|may not exceed", re.I)
_PAYOUT_HINT: Final = re.compile(r"encash|paid out|on leaving|leaving the company", re.I)
_ACCRUAL_HINT: Final = re.compile(
    r"accrual|earned per|days per month|length of service|tenure", re.I
)


def _display_value(value: str) -> str:
    return " ".join(value.split()).strip()


def fold_topic(topic: str) -> str:
    """One topic for every tenure band of the same leave rule.

    Models often emit 'annual leave accrual, 2 to less than 5 years' as a
    separate topic; those are complementary values of one rule. A flat
    'annual leave' / 'leave days' figure is the same rule as the schedule.
    Carry-forward, sick, parental and similar stay their own topics.
    """
    text = canon_topic(topic)
    if text in {"leave days", "leave day", "paid annual leave", "annual leave days", "annual leave allowance"}:
        return "annual leave accrual"
    if not _LEAVE_TOPIC.search(text):
        return text
    stripped = _BAND_TAIL.sub("", text).strip(" ,;-")
    stripped = re.sub(r"\bversion \d+\b", "", stripped).strip()
    if "carry" in stripped or "wellbeing" in stripped:
        return stripped or text
    if "encash" in stripped or "payout" in stripped:
        return stripped or text
    if any(w in stripped for w in ("sick", "parental", "bereavement", "maternity")):
        return stripped or text
    if "accrual" in stripped or "entitlement" in stripped or "annual leave" in stripped:
        return "annual leave accrual"
    return stripped or text


def _topic_for_chunk(topic: str, chunk_text: str, value: str = "") -> str:
    """A carry-forward / cap / payout excerpt is not the accrual rule, even if the model said so."""
    if topic != "annual leave accrual":
        return topic
    if _value_looks_like_payout(value, chunk_text):
        return "annual leave encashment"
    if _value_looks_like_carry(value, chunk_text):
        return "annual leave carry forward"
    if _PAYOUT_HINT.search(chunk_text) and not _ACCRUAL_HINT.search(chunk_text):
        return "annual leave encashment"
    if _CARRY_HINT.search(chunk_text) and not _ACCRUAL_HINT.search(chunk_text):
        return "annual leave carry forward"
    return topic


def _value_looks_like_payout(value: str, chunk_text: str) -> bool:
    number = _NUMBER.search(value)
    if not number:
        return False
    for sentence in re.split(r"[\n.;]", chunk_text):
        if number.group(0) not in sentence:
            continue
        if _PAYOUT_HINT.search(sentence) and not _ACCRUAL_HINT.search(sentence):
            return True
    return False


def _value_looks_like_carry(value: str, chunk_text: str) -> bool:
    number = _NUMBER.search(value)
    if not number:
        return False
    for sentence in re.split(r"[\n.;]", chunk_text):
        if number.group(0) not in sentence:
            continue
        if _CARRY_HINT.search(sentence) and not _ACCRUAL_HINT.search(sentence):
            return True
    return False


_FOCUS_CARRY: Final = "carry"
_FOCUS_SICK: Final = "sick"
_FOCUS_ENCASH: Final = "encashment"
_FOCUS_ACCRUAL: Final = "accrual"


def question_focus(question: str) -> str | None:
    """Which leave rule the question is asking for, or None when it is not specific."""
    text = question.casefold()
    if any(w in text for w in ("carry", "carried forward", "carry over")):
        return _FOCUS_CARRY
    if "sick" in text:
        return _FOCUS_SICK
    if any(w in text for w in ("encash", "paid out", "payout", "leaving the company")):
        return _FOCUS_ENCASH
    if any(
        w in text
        for w in (
            "accrue",
            "accrual",
            "annual leave",
            "leave days",
            "how many days",
            "entitlement",
            "balance",
        )
    ):
        return _FOCUS_ACCRUAL
    return None


def topic_matches_focus(topic: str, focus: str | None) -> bool:
    """True when a resolved topic is what the question asked about."""
    if focus is None:
        return True
    folded = fold_topic(topic)
    if focus == _FOCUS_CARRY:
        return "carry" in folded
    if focus == _FOCUS_SICK:
        return "sick" in folded
    if focus == _FOCUS_ENCASH:
        return "encash" in folded or "payout" in folded
    if focus == _FOCUS_ACCRUAL:
        if any(w in folded for w in ("carry", "encash", "payout", "sick", "parental", "bereavement", "maternity")):
            return False
        return folded == "annual leave accrual"
    return True


def facts_for_question(facts: Facts, question: str) -> Facts:
    """Drop resolved topics the question did not ask about (carry vs payout vs accrual).

    Extraction and table recovery can still surface a neighbouring number. If those
    stay in ``Facts``, composition is required to mention them. Keep the matching
    topics when any exist; otherwise leave the facts unchanged.
    """
    focus = question_focus(question)
    if focus is None:
        return facts
    wanted = tuple(topic for topic in facts.topics if topic_matches_focus(topic.topic, focus))
    if not wanted or wanted == facts.topics:
        return facts
    return Facts(facts.reference, wanted, facts.not_covered)


def filter_not_covered(not_covered: Iterable[str], answered_topics: Iterable[str]) -> tuple[str, ...]:
    """Drop model 'gaps' that a resolved topic already answers (including band splits)."""
    answered = {fold_topic(t) for t in answered_topics} | {canon_topic(t) for t in answered_topics}
    kept: list[str] = []
    for item in not_covered:
        text = _display_value(item)
        if not text:
            continue
        if answered and _REGION_ONLY.match(canon_topic(text)):
            continue
        if fold_topic(text) in answered:
            continue
        kept.append(text)
    return tuple(dict.fromkeys(kept))[:MAX_NOT_COVERED]


def claims_from_table_rows(
    chunks: Sequence[RetrievedChunk], sent: Iterable[tuple[str, str]]
) -> list[dict]:
    """Accrual/entitlement rates from ingested ``Row:`` lines. No corpus numbers here."""
    sent_pairs = set(sent)
    raw: list[dict] = []
    for chunk in chunks:
        if (chunk.doc_id, chunk.locator) not in sent_pairs:
            continue
        for line in chunk.text.splitlines():
            parsed = _parse_rate_row(line)
            if parsed is None:
                continue
            topic, value = parsed
            raw.append(
                {"topic": topic, "value": value, "doc_id": chunk.doc_id, "locator": chunk.locator}
            )
    return raw


def _parse_rate_row(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped.lower().startswith("row:"):
        return None
    body = stripped.split(":", 1)[-1]
    cells = [cell.strip() for cell in body.split("|") if cell.strip()]
    if not cells or _META_ROW.match(cells[0]) or _OTHER_LEAVE.search(body):
        return None
    joined = " ".join(cells).lower()
    if not any(w in joined for w in ("accrual", "annual leave", "length of service", "tenure", "entitlement")):
        return None
    monthly = [(cell, match) for cell in cells if (match := _RATE.search(cell)) and "month" in match.group(0).lower()]
    if not monthly:
        return None
    cell, match = monthly[-1]  # last monthly cell: this document's column in a comparison table
    number = match.group("n1") or match.group("n2")
    band = _band_from_row(cells)
    value = f"{number} days per month"
    if band:
        value = f"{value} for {band}"
    return "annual leave accrual", value[:MAX_VALUE_CHARS]


def _band_from_row(cells: Sequence[str]) -> str:
    first = cells[0]
    for pattern in (
        r"(?i)length of service\s+(.+)",
        r"(?i)tenure\s+(.+)",
        r"(?i)annual leave accrual[,\s]+(.+)",
        r"(?i)topic\s+(.+)",
    ):
        found = re.search(pattern, first)
        if found:
            band = found.group(1).strip(" ,;-")
            band = re.sub(r"(?i)^annual leave accrual[,\s]+", "", band).strip()
            return band
    if not _RATE.search(first):
        return first
    return ""


def normalise_support_text(text: str) -> str:
    """Whitespace, case, plural units, and days/month vs days per month do not matter."""
    folded = " ".join(text.casefold().split())
    folded = _DAYS_PER_MONTH.sub("days per month", folded)
    folded = _DAYS_PER_YEAR.sub("days per year", folded)
    return _UNIT_PLURAL.sub(r"\1", folded)


def _supported(value: str, text: str) -> bool:
    """Is the value grounded in the chunk? Accept a normalised phrase or the numbers.

    A tenure-band value such as a monthly rate plus a band does not have to appear
    as one contiguous phrase; the digits (or their spelled-out form) must.
    """
    if not value or not text:
        return False
    needle = normalise_support_text(value)
    haystack = normalise_support_text(text)
    if needle and needle in haystack:
        return True
    numbers = _NUMBER.findall(value)
    if not numbers:
        compact = _NON_WORD.sub(" ", needle).strip()
        return bool(compact) and compact in _NON_WORD.sub(" ", haystack)
    words = set(re.findall(r"[a-z]+", haystack))
    for n in numbers:
        if re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", haystack):
            continue
        word = _NUMBER_WORDS.get(n)
        if word and word in words:
            continue
        return False
    return True


@dataclass(frozen=True)
class Extraction:
    claims: tuple[dict, ...]
    not_covered: tuple[str, ...]


def parse_extraction(text: object) -> Extraction:
    """The model's JSON, validated. Raises ``InvalidModelOutput`` when it is not valid."""
    data = parse_and_validate(text, EXTRACTION_SCHEMA)
    covered = tuple(
        _display_value(s)[:MAX_VALUE_CHARS] for s in data["not_covered"] if isinstance(s, str) and s.strip()
    )[:MAX_NOT_COVERED]
    return Extraction(tuple(data["claims"]), covered)


def build_claims(
    raw_claims: Iterable[dict],
    chunks: Sequence[RetrievedChunk],
    sent: Iterable[tuple[str, str]],
) -> tuple[list[Claim], dict[Claim, str]]:
    """``Claim`` objects for the model's items; the second result maps each claim to the value
    text to show. ``sent`` is the (doc_id, locator) pairs the guard actually kept."""
    claims, shown, _dropped, _names, _extracted, _model_kept = build_claims_report(raw_claims, chunks, sent)
    return claims, shown


def build_claims_report(
    raw_claims: Iterable[dict],
    chunks: Sequence[RetrievedChunk],
    sent: Iterable[tuple[str, str]],
) -> tuple[list[Claim], dict[Claim, str], list[str], dict[Claim, str], int, int]:
    """Like ``build_claims``, plus drop-reasons, source names, and model extract/kept counts.

    ``extracted`` is the number of claims the model returned, before filtering.
    ``model_kept`` is how many of those survived. Table-row claims are added after
    and do not change either count.
    """
    model_raw = list(raw_claims)
    sent_pairs = set(sent)
    by_source: dict[tuple[str, str], RetrievedChunk] = {}
    for chunk in chunks:
        by_source.setdefault((chunk.doc_id, chunk.locator), chunk)

    claims: list[Claim] = []
    shown: dict[Claim, str] = {}
    names: dict[Claim, str] = {}
    dropped: list[str] = []
    table_raw = claims_from_table_rows(chunks, sent)
    table_sources = {(item["doc_id"], item["locator"]) for item in table_raw}

    def _ingest(items: Iterable[dict], *, from_table: bool) -> int:
        kept = 0
        for item in items:
            source = (item["doc_id"], item["locator"])
            chunk = by_source.get(source)
            display = _display_value(item["value"])
            if chunk is None or source not in sent_pairs:
                chunk = _chunk_for_mislabelled_claim(item["doc_id"], display, chunks, sent_pairs)
                if chunk is None:
                    dropped.append(DROP_NOT_KEPT)
                    continue
                source = (chunk.doc_id, chunk.locator)
                if not from_table and source in table_sources:
                    continue
            topic = _topic_for_chunk(fold_topic(item["topic"]), chunk.text, display)
            value = canon_value(item["value"])
            if not topic or not value or len(topic) > MAX_TOPIC_CHARS or len(display) > MAX_VALUE_CHARS:
                dropped.append(DROP_VALUE)
                continue
            if not _supported(display, chunk.text):
                dropped.append(DROP_VALUE)
                continue
            claim = Claim(topic=topic, value=value, doc=chunk.doc_meta(), locator=chunk.locator)
            if claim not in shown:
                claims.append(claim)
                shown[claim] = display
                names[claim] = _source_name(chunk, chunks)
                kept += 1
        return kept

    model_kept = _ingest(
        (item for item in model_raw if (item.get("doc_id"), item.get("locator")) not in table_sources),
        from_table=False,
    )
    _ingest(table_raw, from_table=True)
    return claims, shown, dropped, names, len(model_raw), model_kept


_QUESTION_STOP: Final = frozenset(
    {
        "the", "and", "for", "how", "many", "what", "which", "does", "do", "did", "are", "is",
        "was", "were", "can", "i", "we", "you", "my", "our", "your", "in", "of", "a", "an",
        "to", "or", "that", "this", "with", "from", "on", "at", "by", "get", "got", "about",
        "me", "it", "if", "not", "be", "will", "would", "please", "tell",
    }
)


def question_terms(question: str) -> frozenset[str]:
    """Content words from the question (no corpus values). Used to spot skipped documents."""
    return frozenset(w for w in re.findall(r"[a-z]{3,}", question.casefold()) if w not in _QUESTION_STOP)


def skipped_numeric_documents(
    question: str,
    chunks: Sequence[RetrievedChunk],
    sent: Iterable[tuple[str, str]],
    claimed_doc_ids: Iterable[str],
) -> list[RetrievedChunk]:
    """Sent chunks of documents that state numbers about the question but produced no model claim."""
    sent_pairs = set(sent)
    claimed = {d for d in claimed_doc_ids if d}
    terms = question_terms(question)
    skipped: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.doc_id in claimed or chunk.doc_id in seen:
            continue
        if (chunk.doc_id, chunk.locator) not in sent_pairs:
            continue
        if not _NUMBER.search(chunk.text):
            continue
        words = set(re.findall(r"[a-z]{3,}", chunk.text.casefold()))
        if terms and not (terms & words):
            continue
        seen.add(chunk.doc_id)
        skipped.extend(c for c in chunks if c.doc_id == chunk.doc_id and (c.doc_id, c.locator) in sent_pairs)
    return skipped


def claims_per_doc(claims: Sequence[Claim]) -> str:
    counts: dict[str, int] = {}
    for claim in claims:
        counts[claim.doc.doc_id] = counts.get(claim.doc.doc_id, 0) + 1
    return ",".join(f"{doc}:{counts[doc]}" for doc in sorted(counts)) or "none"


def _source_name(chunk: RetrievedChunk, chunks: Sequence[RetrievedChunk] | None = None) -> str:
    """A plain document name for compose: a title line, not a numbered section locator."""
    loc = (chunk.locator or "").strip()
    if loc and not re.fullmatch(r"p\.\s*\d+", loc, re.I) and not re.match(r"\d+\.", loc):
        return loc[:MAX_VALUE_CHARS]
    pool = [c for c in (chunks or ()) if c.doc_id == chunk.doc_id] or [chunk]
    for other in pool:
        for line in other.text.splitlines():
            text = line.strip()
            if text and not text.lower().startswith("row:") and not re.match(r"\d+\.", text):
                return text[:MAX_VALUE_CHARS]
    return loc[:MAX_VALUE_CHARS] if loc else ""


def _chunk_for_mislabelled_claim(
    doc_id: object,
    display: str,
    chunks: Sequence[RetrievedChunk],
    sent_pairs: set[tuple[str, str]],
) -> RetrievedChunk | None:
    """The one sent chunk of ``doc_id`` that contains ``display``, if that pair is unique.

    Models often copy a nearby heading instead of the excerpt label. The value still has
    to appear in the excerpt (``_supported``); we never invent a source.
    """
    if not isinstance(doc_id, str) or not doc_id or not display:
        return None
    matches = [
        chunk
        for chunk in chunks
        if chunk.doc_id == doc_id
        and (chunk.doc_id, chunk.locator) in sent_pairs
        and _supported(display, chunk.text)
    ]
    return matches[0] if len(matches) == 1 else None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClaimRef:
    doc_id: str
    locator: str
    effective_date: date
    doc_type: str
    value: str  # as to be shown
    title: str = ""  # plain document name for compose; never a bracketed id


@dataclass(frozen=True)
class TopicFact:
    topic: str
    status: str  # STATUS_DECIDED | STATUS_CONFLICT | STATUS_NONE
    decided_by: str | None = None
    claims: tuple[ClaimRef, ...] = ()  # decided: the claims that carry the answer; conflict: the tied ones
    upcoming: tuple[ClaimRef, ...] = ()  # all winning values from the next-change document


@dataclass(frozen=True)
class Facts:
    reference: date
    topics: tuple[TopicFact, ...] = ()
    not_covered: tuple[str, ...] = ()

    @property
    def has_answer(self) -> bool:
        return any(t.status in (STATUS_DECIDED, STATUS_CONFLICT) or t.upcoming for t in self.topics)

    def sources(self) -> tuple[tuple[str, str], ...]:
        """(doc_id, locator) of every claim the answer rests on: winners, conflicting, upcoming."""
        seen: dict[tuple[str, str], None] = {}
        for topic in self.topics:
            for ref in (*topic.claims, *topic.upcoming):
                seen.setdefault((ref.doc_id, ref.locator), None)
        return tuple(seen)


def _ref(claim: Claim, shown: dict[Claim, str], names: dict[Claim, str] | None = None) -> ClaimRef:
    return ClaimRef(
        doc_id=claim.doc.doc_id,
        locator=claim.locator,
        effective_date=claim.doc.effective_date,
        doc_type=claim.doc.doc_type,
        value=shown.get(claim, str(claim.value)),
        title=(names or {}).get(claim, ""),
    )


def resolve_facts(
    claims: Sequence[Claim],
    shown: dict[Claim, str],
    requester: Requester,
    reference: date,
    not_covered: Iterable[str] = (),
    names: dict[Claim, str] | None = None,
) -> Facts:
    """Per topic: ``resolve_topic`` and ``upcoming_changes`` (pure Python). A topic whose claims
    are all ineligible and with no upcoming change counts as not covered."""
    by_topic: dict[str, list[Claim]] = {}
    for claim in claims:
        by_topic.setdefault(claim.topic, []).append(claim)

    topics: list[TopicFact] = []
    missing = list(not_covered)
    name_map = names or {}
    for topic, items in by_topic.items():
        resolution = resolve_topic(items, reference, requester)
        upcoming = upcoming_changes(items, requester, reference)
        if resolution.status == STATUS_NONE and not upcoming:
            missing.append(topic)
            continue
        ordered = (
            tuple(_ref(c, shown, name_map) for c in resolution.claims) if resolution.status != STATUS_NONE else ()
        )
        topics.append(
            TopicFact(
                topic=topic,
                status=resolution.status,
                decided_by=resolution.decided_by,
                claims=ordered,
                upcoming=tuple(_ref(c, shown, name_map) for c in upcoming),
            )
        )
    return Facts(reference, tuple(topics), filter_not_covered(missing, (t.topic for t in topics)))


# --------------------------------------------------------------------------- #
# Facts as corpus notes (budgeted) and the deterministic wording
# --------------------------------------------------------------------------- #


def fact_parts(facts: Facts) -> list[CorpusPart]:
    """One note part per source document of the facts (decided, conflicting, upcoming)."""
    parts: list[CorpusPart] = []
    for topic in facts.topics:
        for ref in topic.claims:
            if topic.status == STATUS_DECIDED:
                body = (
                    f"Rule: {topic.topic}\n"
                    f"In force on {facts.reference.isoformat()}: {ref.value}\n"
                    f"Source name: {_plain_name(ref)}\n"
                    f"Effective date: {ref.effective_date.isoformat()}"
                )
            else:
                body = (
                    f"Rule: {topic.topic}\n"
                    f"UNRESOLVED CONFLICT, one of the conflicting sources says: {ref.value}\n"
                    f"Source name: {_plain_name(ref)}\n"
                    f"Effective date: {ref.effective_date.isoformat()}"
                )
            parts.append(_note(body, ref))
        for ref in topic.upcoming:
            body = (
                f"Rule: {topic.topic}\n"
                f"UPCOMING from {ref.effective_date.isoformat()} (not in force yet): {ref.value}\n"
                f"Source name: {_plain_name(ref)}\n"
                f"Effective date: {ref.effective_date.isoformat()}"
            )
            parts.append(_note(body, ref))
    return parts


def _plain_name(ref: ClaimRef) -> str:
    return ref.title or ref.locator or "the source document"


def _note(text: str, ref: ClaimRef) -> CorpusPart:
    return CorpusPart(text, ref.doc_id, ref.locator, ref.effective_date, KIND_NOTE)


def _title(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def render_answer(facts: Facts) -> str:
    """The answer in fixed wording (used when the composed text does not follow the facts)."""
    lines: list[str] = []
    for topic in facts.topics:
        if topic.status == STATUS_DECIDED:
            refs = topic.claims
            values = "; ".join(dict.fromkeys(r.value for r in refs))
            name = _plain_name(refs[0])
            lines.append(
                f"{_title(topic.topic)} is currently {values}, from {name}, "
                f"effective {refs[0].effective_date.isoformat()}."
            )
        elif topic.status == STATUS_CONFLICT:
            parts = "; ".join(f"{_plain_name(r)} ({r.effective_date.isoformat()}) says {r.value}" for r in topic.claims)
            lines.append(
                f"{_title(topic.topic)}: the documents conflict and I cannot tell which applies. {parts}. "
                "Please contact HR to confirm."
            )
        else:
            lines.append(f"I found no rule in force for {topic.topic} on {facts.reference.isoformat()}.")
        if topic.upcoming:
            first = topic.upcoming[0]
            upcoming_values = "; ".join(dict.fromkeys(r.value for r in topic.upcoming))
            lines.append(
                f"From {first.effective_date.isoformat()} this changes to {upcoming_values}, "
                f"from {_plain_name(first)}. This is not in force yet."
            )
    return "\n".join(lines)


def mentions(answer: str, value: str) -> bool:
    numbers = _NUMBER.findall(value)
    if numbers:
        return all(re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", answer) for n in numbers)
    return _NON_WORD.sub(" ", value.casefold()).strip() in _NON_WORD.sub(" ", answer.casefold())


def follows_facts(answer: str, facts: Facts) -> bool:
    """Does the composed answer state what the resolution decided? A decided topic: its winning
    value; a conflict: every value and a word saying they conflict; an upcoming change: its
    value and its effective date. Anything less is replaced by ``render_answer``."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    if _SETTLED_BY.search(answer) or _BRACKETED_ID.search(answer):
        return False
    for topic in facts.topics:
        if topic.status == STATUS_DECIDED and not all(mentions(answer, r.value) for r in topic.claims):
            return False
        if topic.status == STATUS_CONFLICT:
            if not all(mentions(answer, r.value) for r in topic.claims) or not _CONFLICT_WORDS.search(answer):
                return False
        if topic.upcoming:
            if topic.upcoming[0].effective_date.isoformat() not in answer:
                return False
            if not all(mentions(answer, r.value) for r in topic.upcoming):
                return False
    return True
