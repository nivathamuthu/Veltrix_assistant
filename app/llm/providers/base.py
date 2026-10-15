"""The one provider interface: ``complete(system, messages, json_schema=None) -> text``.

Every adapter (OpenAI, Anthropic, Google) subclasses ``Provider`` and only says how to build
one HTTP request and where the text is in the reply. This base class owns the rules that must
be identical for all of them:

* temperature 0 (adapters put it in their payload);
* a per-call timeout (``LLM_TIMEOUT_SECONDS``), shortened to whatever is left of the request
  deadline (``app.llm.deadline``);
* AT MOST ONE retry, and only for a timeout, HTTP 429 or HTTP 5xx; a second failure, a 4xx
  other than 429 (bad key, bad request) or any unexpected error becomes ``LLMUnavailable``;
* nothing sensitive is ever logged: no prompts, no responses, no keys, no exception text from
  the HTTP library (which can echo request details). Only the provider name and a fixed
  reason (``timeout``, ``http_429``, ...) appear in a log line.

Calls reach an adapter only through ``app.llm.client`` (which wraps ``guard.call_with_corpus``),
so the 3,000-token corpus budget and the per-call log line apply to every call.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Final

import httpx

from app.llm import deadline

logger = logging.getLogger("app.llm.provider")

MIN_CALL_SECONDS: Final = 1.0  # not worth starting a call with less than this left
DEADLINE_MARGIN_SECONDS: Final = 0.5  # keep a little time for the code after the call
RETRY_DELAY_SECONDS: Final = 0.5


class LLMUnavailable(RuntimeError):
    """The provider could not answer (after at most one retry). The message is a fixed,
    short category; it never contains a prompt, a response, a key or library error text."""

    def __init__(self, reason: str = "unavailable") -> None:
        super().__init__(f"the language model is unavailable ({reason})")
        self.reason = reason


class TransientError(Exception):
    """Internal: a failure worth one retry (timeout, 429, 5xx)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Provider(ABC):
    name: str = "provider"
    default_model: str = ""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "",
        timeout: float = 20.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("an API key is required")
        self._api_key = api_key.strip()
        self.model = (model or "").strip() or self.default_model
        self.timeout = float(timeout)
        self._client = client
        self._sleep = sleep

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self.model!r})"

    # --- the public interface ---------------------------------------------------------------------------------

    def complete(self, system: str, messages: list[dict], json_schema: dict | None = None) -> str:
        """The model's text for ``system`` + ``messages`` (temperature 0).

        ``json_schema`` (a JSON Schema dict with a ``title``) asks for a JSON object; the
        caller still validates what comes back. Raises ``LLMUnavailable`` and nothing else.
        """
        for attempt in (1, 2):
            timeout = self._call_timeout()
            try:
                return self._complete_once(system, list(messages), json_schema, timeout)
            except TransientError as exc:
                if attempt == 2:
                    logger.warning("llm_failed provider=%s reason=%s", self.name, exc.reason)
                    raise LLMUnavailable(exc.reason) from None
                logger.warning("llm_retry provider=%s reason=%s", self.name, exc.reason)
                self._sleep(RETRY_DELAY_SECONDS)
            except LLMUnavailable as exc:
                logger.warning("llm_failed provider=%s reason=%s", self.name, exc.reason)
                raise
            except Exception as exc:  # anything unexpected: no detail, no stack trace to the caller
                logger.warning("llm_failed provider=%s reason=%s", self.name, type(exc).__name__)
                raise LLMUnavailable("error") from None
        raise LLMUnavailable("error")  # pragma: no cover (the loop always returns or raises)

    # --- for the adapters -------------------------------------------------------------------------------------

    @abstractmethod
    def _complete_once(
        self, system: str, messages: list[dict], json_schema: dict | None, timeout: float
    ) -> str:
        """One HTTP round trip. Raise ``TransientError`` for a timeout / 429 / 5xx and
        ``LLMUnavailable`` for anything that a retry cannot fix."""

    def _call_timeout(self) -> float:
        timeout = self.timeout
        left = deadline.remaining()
        if left is not None:
            timeout = min(timeout, left - DEADLINE_MARGIN_SECONDS)
        if timeout < MIN_CALL_SECONDS:
            raise LLMUnavailable("deadline")
        return timeout

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(follow_redirects=False)
        return self._client

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> Any:
        try:
            response = self._http().post(
                url,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0)),
            )
        except httpx.TimeoutException:
            raise TransientError("timeout") from None
        except httpx.HTTPError:
            raise LLMUnavailable("connection") from None
        status = response.status_code
        if status == 429 or status >= 500:
            raise TransientError(f"http_{status}")
        if status >= 400:
            raise LLMUnavailable(f"http_{status}")
        try:
            return response.json()
        except ValueError:
            raise LLMUnavailable("bad_response") from None


def schema_instruction(json_schema: dict) -> str:
    """For providers without a native structured-output switch: the schema as an instruction.
    Part of the SYSTEM prompt; it holds the shape of the answer, never corpus text."""
    import json

    return (
        "\n\nRespond with ONLY one JSON object (no prose, no code fence) that matches this "
        "JSON Schema:\n" + json.dumps(json_schema, separators=(",", ":"))
    )
