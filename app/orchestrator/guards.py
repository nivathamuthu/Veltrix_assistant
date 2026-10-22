"""Deterministic guards (README B3, B1 rules 5-7): they run BEFORE any LLM.

A message that asks for bulk data (everyone's balances, a list of employees, an export),
for the salary of someone else, for anyone's phone number or email, for the system prompt,
or that tries "ignore your instructions" is refused here, in code, with ONE fixed polite
message. The analyzer (the LLM) never sees it.

What is NOT a guard trigger
    * A manager asking about their own team ("who in my team is on leave next week?"): the
      tools decide that from the database.
    * A role claim ("I am HR", "I'm an admin"): ignored, never refused. Role comes only
      from the database, so the claim has no effect either way.
    * Questions that merely contain a word like "employees", "email" or "instructions".

The patterns are deliberately narrow (a verb or possessive plus a noun) to keep false
positives low, and they run on a normalised copy of the text (case, spacing, punctuation,
typographic apostrophes). A refusal says what the assistant can help with and nothing about
what was asked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

KIND_INJECTION: Final = "injection"
KIND_SYSTEM_PROMPT: Final = "system_prompt"
KIND_SALARY: Final = "salary"
KIND_CONTACT: Final = "contact_details"
KIND_BULK: Final = "bulk_data"

GUARD_REFUSAL_MESSAGE: Final = (
    "I can't help with that request. I can answer questions about Veltrix leave, holidays and HR "
    "policy, show your own leave balance and leave requests, help you apply for or cancel your own "
    "leave, and, if you are a manager, show when your direct reports are on leave."
)

# Off-topic messages (general knowledge, coding help, jokes, anything outside HR and leave): a
# polite refusal that says what the assistant CAN help with. Fixed text; it echoes nothing.
OFF_TOPIC_MESSAGE: Final = (
    "Sorry, that's outside what I can help with. I can help with your leave balance and leave "
    "requests, applying for and cancelling leave, team leave if you are a manager, and Veltrix HR "
    "policies and holidays."
)


@dataclass(frozen=True)
class GuardDecision:
    refuse: bool
    kind: str | None = None


_ALLOW = GuardDecision(False, None)

_PEOPLE = r"(?:employees?|staff|people|colleagues|coworkers|workers|personnel|users|team\s+members)"
_SHOW = r"(?:list|show|give|get|send|export|download|dump|print|display|provide|fetch)"
_LEAVE_NOUN = r"(?:leave|balances?|requests?|data|details|records?|information|info|numbers|days)"
_SALARY_NOUN = r"(?:salary|salaries|compensation|wages?|ctc|remuneration)"
_ROLES = (
    r"(?:manager|team|colleagues?|boss|reports?|peers?|coworkers?|staff|employees|people|ceo|hr|director|lead)"
)

_INJECTION = (
    # "ignore / disregard / forget / override / bypass ... instructions / rules / checks"
    r"\b(?:ignore|disregard|forget|override|bypass|circumvent)\s+"
    r"(?:(?:all|any|the|your|my|these|those|previous|prior|above|earlier|preceding|safety|permission|"
    r"security|system|everything)\s+)*"
    r"(?:instructions?|rules|guidelines|prompts?|restrictions|checks|filters|guardrails|permissions|safeguards)\b",
    r"\byou\s+are\s+now\b",
    r"\bpretend\s+(?:that\s+)?(?:you\s+are|you're|to\s+be)\b",
    r"\bact\s+as\s+(?:an?\s+|the\s+)?(?:admin|administrator|hr|developer|system|root|unrestricted|"
    r"different|new|jailbroken|dan)\b",
    r"\b(?:enable|enter|activate|switch\s+to)\s+(?:developer|debug|admin|god|dan|jailbreak)\s+mode\b",
    r"\bnew\s+instructions?\b",
    r"\bjailbreak(?:ed|ing)?\b",
    r"\bdo\s+anything\s+now\b",
)

_PROMPT_VERBS = (
    r"(?:reveal|show|print|repeat|display|tell|share|output|give|leak|read|paste|recite|dump|disclose|"
    r"expose|what|whats|what's|list|summari[sz]e|describe|quote)"
)
_SYSTEM_PROMPT = (
    rf"\b{_PROMPT_VERBS}\b.{{0,40}}\b(?:your|ur)\s+(?:(?:system|initial|original|hidden|secret|developer|"
    r"internal|full|exact)\s+)*(?:prompt|instructions?|rules|guidelines|configuration|system\s+message|"
    r"programming|directives)\b",
    r"\b(?:system|hidden|secret|developer)\s+(?:prompt|message|instructions?)\b",
    r"\b(?:initial|original)\s+(?:prompt|instructions?)\b",
    r"\b(?:rules|instructions?|guidelines)\s+(?:you\s+(?:were|are|have\s+been)\s+(?:given|told)|given\s+to\s+you)\b",
    r"\bwhat\s+(?:were|are)\s+you\s+(?:told|instructed|programmed)\b",
)

_SALARY = (
    # possessive: "Priya's salary" (the owner is checked in code), "everyone's salary"
    rf"\b[a-z]+'s\s+{_SALARY_NOUN}\b",
    rf"\b(?:everyone|everybody|all|every\s+\w+|each\s+\w+)(?:'s?)?\s+(?:[\w']+\s+){{0,2}}{_SALARY_NOUN}\b",
    rf"\b{_SALARY_NOUN}\b.{{0,30}}\b(?:of|for)\s+(?:everyone|everybody|all\b|every\b|each\b|him\b|her\b|them\b|"
    rf"(?:my|our|the)\s+{_ROLES}\b)",
    rf"\b{_SALARY_NOUN}\s+of\s+(?!(?:the|a|an|my|our|this|that|these|those|any|each|every|all)\b)[a-z]+",
    rf"\b(?:his|her|their)\s+(?:{_SALARY_NOUN}|pay)\b",
    r"\bhow\s+much\s+(?:does|do|is|are|did)\s+(?!i\b|we\b|you\b|my\b|the\s+company\b)[a-z']+(?:\s+[a-z']+)?\s+"
    r"(?:earn|make|get\s+paid|gets\s+paid|paid|receive)\b",
    r"\bwho\s+(?:earns|makes|is\s+paid|gets\s+paid)\b",
    rf"\b{_SALARY_NOUN}\s+(?:list|report|data|details|information|info|sheet|spreadsheet|table|breakdown|ranges?)\b",
    r"\bwhat are the (?:salary|compensation) bands?\b",
    r"\b(?:show|tell me)\s+(?:the\s+)?(?:salary|compensation)\s+bands?\b",
)
# "Priya's salary" is someone else's; "the company's salary bands" is not a person
_NOT_A_PERSON = frozenset({"company", "organization", "organisation", "firm", "veltrix", "policy", "it", "that", "what", "who", "there", "here"})

_CONTACT = (
    r"\b(?:phone|mobile|cell|telephone|whatsapp|contact)\s+(?:phone\s+)?(?:numbers?|nos?)\b",
    r"\be ?mail\s+(?:address(?:es)?|ids?)\b",
    r"\b(?:their|his|her|my|your)\s+e ?mails?\b|'s\s+e ?mails?\b",
    r"\be ?mails?\s+of\b",
    r"\bcontact\s+(?:details|info|information)\b",
)

_BULK = (
    # everyone's leave balances / balances of everyone / all employees' balances
    rf"\b(?:everyone|everybody)'s(?:\s+[\w']+){{0,2}}?\s+{_LEAVE_NOUN}\b",
    rf"\b(?:balances?|requests?|leave\s+(?:balances?|requests?|data|records?)|data|details|records?|information)"
    rf"\s+(?:of|for)\s+(?:everyone|everybody|(?:all|every|each)\s+(?:the\s+)?{_PEOPLE})\b",
    rf"\b(?:all|every|each)\s+(?:the\s+)?{_PEOPLE}(?:'s?)?\s+(?:\w+\s+)?(?:leave\s+)?"
    r"(?:balances?|requests?|data|details|records?|information|numbers)\b",
    r"\ball\s+(?:the\s+)?(?:leave\s+)?(?:balances|requests)\b",
    rf"\b{_SHOW}\b.{{0,25}}\b(?:all|every|each)\s+(?:the\s+)?{_PEOPLE}\b",
    r"\b(?:whole|entire|full|complete)\s+(?:company|organi[sz]ation|team|department|staff|workforce|firm)(?:'s?)?\b"
    r".{0,30}\b(?:leave|balances?|data|requests?|details|records?|information|numbers|calendar|list|salary|salaries)\b",
    r"\b(?:employee|staff|personnel|company|workforce)\s+(?:directory|list|roster|database|register|export|dump|headcount)\b",
    rf"\b{_SHOW}\b.{{0,20}}\b(?:payroll|directory|roster|headcount|org\s+chart)\b",
    r"\b(?:csv|spreadsheet|excel|xlsx|export|dump|download)\b.{0,40}\b(?:employees?|staff|everyone|everybody|"
    r"payroll|directory|database|workforce|all\s+(?:leave|balances|requests))\b",
)


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


_RULES: Final = (
    (KIND_INJECTION, _compile(_INJECTION)),
    (KIND_SYSTEM_PROMPT, _compile(_SYSTEM_PROMPT)),
    (KIND_SALARY, _compile(_SALARY)),
    (KIND_CONTACT, _compile(_CONTACT)),
    (KIND_BULK, _compile(_BULK)),
)
_POSSESSIVE = re.compile(rf"\b([a-z]+)'s\s+{_SALARY_NOUN}\b")


def _normalise(message: str) -> str:
    text = message.casefold().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    return re.sub(r"[^\w']+", " ", text).strip()


def _salary_of_a_person(text: str) -> bool:
    return any(m.group(1) not in _NOT_A_PERSON for m in _POSSESSIVE.finditer(text))


def check_guards(message: str) -> GuardDecision:
    """Refuse (with a ``kind``) or let the message through. Text that is not a string is not
    refused here (the endpoint rejects it as a bad request before this runs)."""
    if not isinstance(message, str):
        return _ALLOW
    text = _normalise(message)
    if not text:
        return _ALLOW
    for kind, patterns in _RULES:
        for pattern in patterns:
            if pattern.search(text):
                if kind == KIND_SALARY and pattern.pattern.startswith(r"\b[a-z]+'s") and not _salary_of_a_person(text):
                    continue
                return GuardDecision(True, kind)
    return _ALLOW
