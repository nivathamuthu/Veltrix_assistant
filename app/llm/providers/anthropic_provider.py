"""Anthropic Messages adapter (thin): same interface, same retry and timeout rules.

``POST /v1/messages``. There is no schema switch here: a requested JSON schema is added to
the system prompt as an instruction, and the caller validates the result as always.
"""
from __future__ import annotations

from typing import Any, Final

from app.llm.providers.base import LLMUnavailable, Provider, schema_instruction

DEFAULT_BASE_URL: Final = "https://api.anthropic.com/v1"
API_VERSION: Final = "2023-06-01"
MAX_OUTPUT_TOKENS: Final = 1500


class AnthropicProvider(Provider):
    name = "anthropic"
    default_model = "claude-3-5-haiku-latest"  # used only when LLM_MODEL is empty

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")

    def _complete_once(
        self, system: str, messages: list[dict], json_schema: dict | None, timeout: float
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "system": system + (schema_instruction(json_schema) if json_schema else ""),
            "messages": messages,
        }
        data = self._post(
            f"{self._base_url}/messages",
            {"x-api-key": self._api_key, "anthropic-version": API_VERSION, "content-type": "application/json"},
            payload,
            timeout,
        )
        try:
            blocks = data["content"]
            return "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        except (KeyError, TypeError, AttributeError):
            raise LLMUnavailable("bad_response") from None
