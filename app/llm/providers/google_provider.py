"""Google Gemini adapter (thin): same interface, same retry and timeout rules.

``POST /v1beta/models/{model}:generateContent``. The key goes in a header, never in the URL
(HTTP client libraries log URLs). Thinking uses exactly one setting: ``thinkingLevel``
``LOW`` on ``gemini-3*`` (``MINIMAL`` is rejected by gemini-3.8-flash), ``thinkingBudget`` 0
on Gemini 2.5 and older. A
requested JSON schema switches on ``responseMimeType: application/json`` and is also added
to the system instruction; the caller validates the result as always.

The answer is the joined ``text`` of ``candidates[0].content.parts``, ignoring thought
parts. Empty, blocked or truncated replies are a clean ``LLMUnavailable``.
"""
from __future__ import annotations

from typing import Any, Final

from app.llm.providers.base import LLMUnavailable, Provider, schema_instruction

DEFAULT_BASE_URL: Final = "https://generativelanguage.googleapis.com/v1beta"
MAX_OUTPUT_TOKENS: Final = 4096
_ROLES: Final = {"user": "user", "assistant": "model"}
_OK_FINISH: Final = frozenset({"STOP", "FINISH_REASON_UNSPECIFIED", ""})
_TRUNCATED_FINISH: Final = frozenset({"MAX_TOKENS"})
# gemini-3.8-flash: "Thinking level MINIMAL is not supported for this model."
GEMINI_3_THINKING_LEVEL: Final = "LOW"


def thinking_config_for(model: str) -> dict[str, Any]:
    """Exactly one thinking setting. Gemini 3 rejects ``thinkingBudget``; 2.5 rejects ``thinkingLevel``."""
    name = (model or "").strip().lower()
    if name.startswith("gemini-3"):
        return {"thinkingLevel": GEMINI_3_THINKING_LEVEL}
    return {"thinkingBudget": 0}


def _finish_of(candidate: dict[str, Any]) -> str:
    raw = candidate.get("finishReason") or candidate.get("finish_reason") or ""
    return str(raw) if raw else ""


def _join_answer_parts(parts: object) -> str:
    """Visible answer text only: skip thought parts and non-text parts."""
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True:
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return "".join(texts)


def read_google_text(data: object) -> str:
    """The model's answer, or ``LLMUnavailable`` for empty / blocked / truncated."""
    if not isinstance(data, dict):
        raise LLMUnavailable("bad_response")
    feedback = data.get("promptFeedback")
    if isinstance(feedback, dict):
        block = feedback.get("blockReason") or feedback.get("block_reason")
        if block:
            raise LLMUnavailable("blocked", finish_reason=str(block))
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMUnavailable("empty")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise LLMUnavailable("bad_response")
    finish = _finish_of(candidate)
    if finish in _TRUNCATED_FINISH:
        raise LLMUnavailable("truncated", finish_reason=finish)
    if finish and finish not in _OK_FINISH:
        raise LLMUnavailable("blocked", finish_reason=finish)
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    text = _join_answer_parts(parts)
    if not text.strip():
        raise LLMUnavailable("empty", finish_reason=finish or None)
    return text


class GoogleProvider(Provider):
    name = "google"
    default_model = "gemini-3.8-flash"  # used only when LLM_MODEL is empty

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")

    def _complete_once(
        self, system: str, messages: list[dict], json_schema: dict | None, timeout: float
    ) -> str:
        generation: dict[str, Any] = {
            "temperature": 0,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig": thinking_config_for(self.model),
        }
        if json_schema is not None:
            generation["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": system + (schema_instruction(json_schema) if json_schema else "")}]
            },
            "contents": [
                {"role": _ROLES.get(m["role"], "user"), "parts": [{"text": m["content"]}]} for m in messages
            ],
            "generationConfig": generation,
        }
        data = self._post(
            f"{self._base_url}/models/{self.model}:generateContent",
            {"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
            payload,
            timeout,
        )
        return read_google_text(data)
