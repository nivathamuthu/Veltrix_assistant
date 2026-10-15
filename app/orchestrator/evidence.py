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
* ``resolve_facts`` runs ``resolve_topic`` and ``upcoming_change`` per topic (authority, then
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
    DECIDED_BY_AGREEMENT,
    DECIDED_BY_AUTHORITY,
    DECIDED_BY_ONLY_CLAIM,
    DECIDED_BY_RECENCY,
    DECIDED_BY_SCOPE,
    STATUS_CONFLICT,
    STATUS_DECIDED,
    STATUS_NONE,
    Claim,
    Requester,
    resolve_topic,
    upcoming_change,
)

MAX_UPCOMING_CHUNKS: Final = 3
CURRENT_FIRST: Final = 5  # current chunks placed before the upcoming ones in the prompt
CURRENT_K: Final = 8
UPCOMING_K: Final = 3
MAX_TOPIC_CHARS: Final = 80
MAX_VALUE_CHARS: Final = 120
MAX_NOT_COVERED: Final = 5

_NON_WORD: Final = re.compile(r"[^0-9a-z]+")
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")
_UNIT_PLURAL: Final = re.compile(r"\b(day|week|month|year|hour|holiday)s\b")
_NUMBER_WORDS: Final = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve",
    "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty",
}
_DECIDED_BY_TEXT: Final = {
    DECIDED_BY_ONLY_CLAIM: "the only applicable document",
    DECIDED_BY_AUTHORITY: "document authority",
    DECIDED_BY_SCOPE: "regional scope",
    DECIDED_BY_RECENCY: "most recent effective date",
    DECIDED_BY_AGREEMENT: "the applicable documents agree",
}
_CONFLICT_WORDS: Final = re.compile(r"conflict|contradict|disagree|inconsisten|differ", re.IGNORECASE)


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


def corpus_of(chunks: Iterable[RetrievedChunk]) -> CorpusContext:
    context = CorpusContext()
    for chunk in chunks:
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


def _display_value(value: str) -> str:
    return " ".join(value.split()).strip()


def _supported(value: str, text: str) -> bool:
    """Is the value in the chunk text? Digits (or their spelled-out form) must appear; a value
    without digits must appear as a phrase. Models normalise, they do not invent."""
    lowered = text.casefold()
    numbers = _NUMBER.findall(value)
    if numbers:
        words = set(re.findall(r"[a-z]+", lowered))
        for n in numbers:
            if re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", lowered):
                continue
            word = _NUMBER_WORDS.get(n)
            if word and word in words:
                continue
            return False
        return True
    needle = _NON_WORD.sub(" ", value.casefold()).strip()
    return bool(needle) and needle in _NON_WORD.sub(" ", lowered)


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
    sent_pairs = set(sent)
    by_source: dict[tuple[str, str], RetrievedChunk] = {}
    for chunk in chunks:
        by_source.setdefault((chunk.doc_id, chunk.locator), chunk)

    claims: list[Claim] = []
    shown: dict[Claim, str] = {}
    for item in raw_claims:
        source = (item["doc_id"], item["locator"])
        chunk = by_source.get(source)
        if chunk is None or source not in sent_pairs:
            continue  # not in the kept set: an invented or unsent source
        topic = canon_topic(item["topic"])
        display = _display_value(item["value"])
        value = canon_value(item["value"])
        if not topic or not value or len(topic) > MAX_TOPIC_CHARS or len(display) > MAX_VALUE_CHARS:
            continue
        if not _supported(display, chunk.text):
            continue  # the excerpt does not contain this value
        claim = Claim(topic=topic, value=value, doc=chunk.doc_meta(), locator=chunk.locator)
        if claim not in shown:
            claims.append(claim)
            shown[claim] = display
    return claims, shown


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


@dataclass(frozen=True)
class TopicFact:
    topic: str
    status: str  # STATUS_DECIDED | STATUS_CONFLICT | STATUS_NONE
    decided_by: str | None = None
    claims: tuple[ClaimRef, ...] = ()  # decided: the claims that carry the answer; conflict: the tied ones
    upcoming: ClaimRef | None = None


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
            for ref in (*topic.claims, *((topic.upcoming,) if topic.upcoming else ())):
                seen.setdefault((ref.doc_id, ref.locator), None)
        return tuple(seen)


def _ref(claim: Claim, shown: dict[Claim, str]) -> ClaimRef:
    return ClaimRef(
        doc_id=claim.doc.doc_id,
        locator=claim.locator,
        effective_date=claim.doc.effective_date,
        doc_type=claim.doc.doc_type,
        value=shown.get(claim, str(claim.value)),
    )


def resolve_facts(
    claims: Sequence[Claim],
    shown: dict[Claim, str],
    requester: Requester,
    reference: date,
    not_covered: Iterable[str] = (),
) -> Facts:
    """Per topic: ``resolve_topic`` and ``upcoming_change`` (pure Python). A topic whose claims
    are all ineligible and with no upcoming change counts as not covered."""
    by_topic: dict[str, list[Claim]] = {}
    for claim in claims:
        by_topic.setdefault(claim.topic, []).append(claim)

    topics: list[TopicFact] = []
    missing = list(not_covered)
    for topic, items in by_topic.items():
        resolution = resolve_topic(items, reference, requester)
        upcoming = upcoming_change(items, requester, reference)
        if resolution.status == STATUS_NONE and upcoming is None:
            missing.append(topic)
            continue
        ordered = (
            tuple(_ref(c, shown) for c in resolution.claims) if resolution.status != STATUS_NONE else ()
        )
        topics.append(
            TopicFact(
                topic=topic,
                status=resolution.status,
                decided_by=resolution.decided_by,
                claims=ordered,
                upcoming=_ref(upcoming, shown) if upcoming else None,
            )
        )
    return Facts(reference, tuple(topics), tuple(dict.fromkeys(missing))[:MAX_NOT_COVERED])


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
                    f"Settled by: {_DECIDED_BY_TEXT.get(topic.decided_by or '', 'precedence')}"
                )
            else:
                body = (
                    f"Rule: {topic.topic}\n"
                    f"UNRESOLVED CONFLICT, one of the conflicting sources says: {ref.value}"
                )
            parts.append(_note(body, ref))
        if topic.upcoming:
            ref = topic.upcoming
            body = (
                f"Rule: {topic.topic}\n"
                f"UPCOMING from {ref.effective_date.isoformat()} (not in force yet): {ref.value}"
            )
            parts.append(_note(body, ref))
    return parts


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
            sources = "; ".join(dict.fromkeys(f"{r.doc_id}, {r.locator}" for r in refs))
            lines.append(
                f"{_title(topic.topic)}: {refs[0].value} (in force on {facts.reference.isoformat()}; "
                f"source: {sources})."
            )
        elif topic.status == STATUS_CONFLICT:
            parts = "; ".join(f"{r.doc_id}, {r.locator} says {r.value}" for r in topic.claims)
            lines.append(
                f"{_title(topic.topic)}: the documents conflict and I cannot tell which applies. {parts}. "
                "Please contact HR to confirm."
            )
        else:
            lines.append(f"I found no rule in force for {topic.topic} on {facts.reference.isoformat()}.")
        if topic.upcoming:
            u = topic.upcoming
            lines.append(
                f"Upcoming change from {u.effective_date.isoformat()}: {u.value} "
                f"(source: {u.doc_id}, {u.locator}). This is not in force yet."
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
    for topic in facts.topics:
        if topic.status == STATUS_DECIDED and not mentions(answer, topic.claims[0].value):
            return False
        if topic.status == STATUS_CONFLICT:
            if not all(mentions(answer, r.value) for r in topic.claims) or not _CONFLICT_WORDS.search(answer):
                return False
        if topic.upcoming and (
            topic.upcoming.effective_date.isoformat() not in answer or not mentions(answer, topic.upcoming.value)
        ):
            return False
    return True
