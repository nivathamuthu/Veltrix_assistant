"""The one provider interface: ``complete(system, messages, json_schema=None) -> text``.

Every adapter (OpenAI, Anthropic, Google) subclasses ``Provider`` and only says how to build
one HTTP request and where the text is in the reply. This base class owns the rules that must
be identical for all of them:

* temperature 0 (adapters put it in their payload);
* a per-call timeout (``LLM_TIMEOUT_SECONDS``), shortened to whatever is left of the request
  deadline (``app.llm.deadline``);
* AT MOST ONE retry, and only for a timeout, HTTP 429 or HTTP 5xx; a second failure, a 4xx
  other than 429 (bad key, bad request) or any unexpected error becomes ``LLMUnavailable``;
* a retry is skipped when too little of the request deadline remains;
* outbound calls can be paced (``min_interval``) so a free-tier per-minute limit is not burst;
* a 429 waits the provider's ``Retry-After`` / ``retryDelay`` and is retried once only if
  that delay fits the request deadline; with no delay it is not retried;
* nothing sensitive is ever logged: no prompts, no responses, no keys, no exception text from
  the HTTP library (which can echo request details). Each failed call logs one line with the
  provider, HTTP status, error status/code, finishReason, attempt and elapsed seconds.

Calls reach an adapter only through ``app.llm.client`` (which wraps ``guard.call_with_corpus``),
so the 3,000-token corpus budget and the per-call log line apply to every call.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Final

import httpx

from app.llm import deadline

logger = logging.getLogger("app.llm.provider")

MIN_CALL_SECONDS: Final = 1.0  # not worth starting a call with less than this left
DEADLINE_MARGIN_SECONDS: Final = 0.5  # keep a little time for the code after the call
RETRY_DELAY_SECONDS: Final = 0.5


def _field(value: object) -> str:
    """A log token: never empty, never a prompt or key (callers pass only codes)."""
    if value is None or value == "":
        return "-"
    return str(value)


def error_fields(body: object) -> tuple[str | None, str | None]:
    """``(status, code)`` from a provider error object. The message is ignored."""
    if not isinstance(body, dict):
        return None, None
    err = body.get("error")
    if not isinstance(err, dict):
        return None, None
    status = err.get("status") or err.get("type")
    code = err.get("code")
    return (
        str(status) if isinstance(status, (str, int)) else None,
        str(code) if isinstance(code, (str, int)) else None,
    )


def _duration_seconds(value: object) -> float | None:
    """A Google ``retryDelay``: ``"17s"``, ``1.5``, or ``{"seconds": 17, "nanos": 0}``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, dict):
        try:
            seconds = float(value.get("seconds") or 0)
            nanos = float(value.get("nanos") or 0)
        except (TypeError, ValueError):
            return None
        return max(0.0, seconds + nanos / 1_000_000_000)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("s", "S")):
            text = text[:-1]
        try:
            return max(0.0, float(text))
        except ValueError:
            return None
    return None


def retry_after_seconds(response: httpx.Response, body: object) -> float | None:
    """Seconds Google asked us to wait: ``Retry-After``, then ``error.details[].retryDelay``.

    The error message is never returned or logged.
    """
    raw = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if raw:
        parsed = _duration_seconds(raw)
        if parsed is not None:
            return parsed
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return None
    details = body["error"].get("details")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            parsed = _duration_seconds(item.get("retryDelay") or item.get("retry_delay"))
            if parsed is not None:
                return parsed
    return None


class CallPacer:
    """Space outbound calls so a free-tier per-minute limit is not burst."""

    def __init__(self, min_interval: float = 0.0) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self, sleep: Callable[[float], None], remaining: float | None) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_at - now)
            if remaining is not None and delay > 0 and remaining < delay + MIN_CALL_SECONDS:
                raise LLMUnavailable("deadline")
            self._next_at = max(now, self._next_at) + self.min_interval
        if delay > 0:
            sleep(delay)


class LLMUnavailable(RuntimeError):
    """The provider could not answer (after at most one retry). The message is a fixed,
    short category; it never contains a prompt, a response, a key or library error text."""

    def __init__(
        self,
        reason: str = "unavailable",
        *,
        http_status: object = None,
        error_status: object = None,
        error_code: object = None,
        finish_reason: object = None,
    ) -> None:
        super().__init__(f"the language model is unavailable ({reason})")
        self.reason = reason
        self.http_status = http_status
        self.error_status = error_status
        self.error_code = error_code
        self.finish_reason = finish_reason


class TransientError(Exception):
    """Internal: a failure worth one retry (timeout, 429, 5xx)."""

    def __init__(
        self,
        reason: str,
        *,
        http_status: object = None,
        error_status: object = None,
        error_code: object = None,
        finish_reason: object = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.http_status = http_status
        self.error_status = error_status
        self.error_code = error_code
        self.finish_reason = finish_reason
        self.retry_after = retry_after


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
        min_interval: float = 0.0,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("an API key is required")
        self._api_key = api_key.strip()
        self.model = (model or "").strip() or self.default_model
        self.timeout = float(timeout)
        self._client = client
        self._sleep = sleep
        self._pacer = CallPacer(min_interval)

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self.model!r})"

    # --- the public interface ---------------------------------------------------------------------------------

    def complete(self, system: str, messages: list[dict], json_schema: dict | None = None) -> str:
        """The model's text for ``system`` + ``messages`` (temperature 0).

        ``json_schema`` (a JSON Schema dict with a ``title``) asks for a JSON object; the
        caller still validates what comes back. Raises ``LLMUnavailable`` and nothing else.
        """
        for attempt in (1, 2):
            started = time.monotonic()
            try:
                timeout = self._call_timeout()
                self._pacer.wait(self._sleep, deadline.remaining())
                return self._complete_once(system, list(messages), json_schema, timeout)
            except TransientError as exc:
                self._log_failed(exc, attempt, started)
                wait = self._retry_wait(exc)
                if attempt == 2 or not self._has_time_for_retry(wait):
                    raise LLMUnavailable(
                        "deadline" if attempt == 1 and not self._has_time_for_retry(wait) else exc.reason,
                        http_status=exc.http_status,
                        error_status=exc.error_status,
                        error_code=exc.error_code,
                        finish_reason=exc.finish_reason,
                    ) from None
                self._sleep(wait)
            except LLMUnavailable as exc:
                self._log_failed(exc, attempt, started)
                raise
            except Exception as exc:  # anything unexpected: no detail, no stack trace to the caller
                self._log_failed(exc, attempt, started)
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

    def _retry_wait(self, exc: TransientError) -> float:
        after = exc.retry_after
        if isinstance(after, (int, float)) and after > 0:
            return float(after)
        return RETRY_DELAY_SECONDS

    def _has_time_for_retry(self, wait: float = RETRY_DELAY_SECONDS) -> bool:
        left = deadline.remaining()
        if left is None:
            return True
        return left >= MIN_CALL_SECONDS + DEADLINE_MARGIN_SECONDS + max(0.0, wait)

    def _log_failed(self, exc: BaseException, attempt: int, started: float) -> None:
        logger.warning(
            "llm_failed provider=%s http_status=%s error_status=%s error_code=%s "
            "finish_reason=%s attempt=%s elapsed=%.2f",
            self.name,
            _field(getattr(exc, "http_status", None)),
            _field(getattr(exc, "error_status", None)),
            _field(getattr(exc, "error_code", None)),
            _field(getattr(exc, "finish_reason", None)),
            attempt,
            time.monotonic() - started,
        )

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
        try:
            body: Any = response.json()
        except ValueError:
            body = None
        err_status, err_code = error_fields(body)
        if status == 429:
            wait = retry_after_seconds(response, body)
            if wait is None or wait <= 0:
                # No delay from Google: do not retry instantly (that burns another quota slot).
                raise LLMUnavailable(
                    "http_429",
                    http_status=status,
                    error_status=err_status,
                    error_code=err_code,
                )
            raise TransientError(
                "http_429",
                http_status=status,
                error_status=err_status,
                error_code=err_code,
                retry_after=wait,
            )
        if status >= 500:
            raise TransientError(
                f"http_{status}",
                http_status=status,
                error_status=err_status,
                error_code=err_code,
            )
        if status >= 400:
            raise LLMUnavailable(
                f"http_{status}",
                http_status=status,
                error_status=err_status,
                error_code=err_code,
            )
        if body is None:
            raise LLMUnavailable("bad_response", http_status=status)
        return body


def schema_instruction(json_schema: dict) -> str:
    """For providers without a native structured-output switch: the schema as an instruction.
    Part of the SYSTEM prompt; it holds the shape of the answer, never corpus text."""
    import json

    return (
        "\n\nRespond with ONLY one JSON object (no prose, no code fence) that matches this "
        "JSON Schema:\n" + json.dumps(json_schema, separators=(",", ":"))
    )
