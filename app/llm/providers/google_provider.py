"""Google Gemini adapter (thin): same interface, same retry and timeout rules.

``POST /v1beta/models/{model}:generateContent``. The key goes in a header, never in the URL
(HTTP client libraries log URLs). A requested JSON schema switches on JSON output and is also
added to the system instruction; the caller validates the result as always.
"""
from __future__ import annotations

from typing import Any, Final

from app.llm.providers.base import LLMUnavailable, Provider, schema_instruction

DEFAULT_BASE_URL: Final = "https://generativelanguage.googleapis.com/v1beta"
_ROLES: Final = {"user": "user", "assistant": "model"}


class GoogleProvider(Provider):
    name = "google"
    default_model = "gemini-2.0-flash"  # used only when LLM_MODEL is empty

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")

    def _complete_once(
        self, system: str, messages: list[dict], json_schema: dict | None, timeout: float
    ) -> str:
        generation: dict[str, Any] = {"temperature": 0}
        if json_schema is not None:
            generation["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system + (schema_instruction(json_schema) if json_schema else "")}]},
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
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMUnavailable("bad_response") from None
