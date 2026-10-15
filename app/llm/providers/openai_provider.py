"""OpenAI Chat Completions adapter (the fully implemented provider).

Talks to ``POST /v1/chat/completions`` over HTTPS with ``httpx`` (no SDK: the retry and
timeout rules live in ``Provider`` and are the same for every provider). A JSON schema is
sent as a strict structured-output request, so the model is constrained to the shape the
caller will validate anyway.

Temperature is always 0, except for the model families that reject the parameter outright
(the o-series and gpt-5 reasoning models answer HTTP 400 to any temperature). For those the
parameter is left out; there is no other way to call them.
"""
from __future__ import annotations

from typing import Any, Final

from app.llm.providers.base import LLMUnavailable, Provider

DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
_NO_TEMPERATURE_PREFIXES: Final = ("o1", "o3", "o4", "gpt-5")


def accepts_temperature(model: str) -> bool:
    name = model.strip().lower()
    return not name.startswith(_NO_TEMPERATURE_PREFIXES)


def strict_schema(schema: dict) -> tuple[str, dict]:
    """``(name, schema)`` for OpenAI's ``json_schema`` response format."""
    body = {k: v for k, v in schema.items() if k not in ("title", "description")}
    name = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(schema.get("title") or "answer"))
    return (name[:64] or "answer"), body


class OpenAIProvider(Provider):
    name = "openai"
    default_model = "gpt-4o-mini"  # used only when LLM_MODEL is empty

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")

    def _complete_once(
        self, system: str, messages: list[dict], json_schema: dict | None, timeout: float
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if accepts_temperature(self.model):
            payload["temperature"] = 0
        if json_schema is not None:
            name, body = strict_schema(json_schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": body},
            }
        data = self._post(
            f"{self._base_url}/chat/completions",
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            payload,
            timeout,
        )
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMUnavailable("bad_response") from None
        # A refusal or an empty message has no content: the caller treats "" as invalid output.
        return content if isinstance(content, str) else ""
