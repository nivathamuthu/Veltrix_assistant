"""One real provider call: print HTTP status and the error code/message.

    python -m app.llm.provider_check

Not part of ``make test``. Use a configured key, typically:

    docker compose exec app python -m app.llm.provider_check

Never prints the API key, the prompt, or the model reply. The request is the same
payload the adapter would send (including ``generationConfig``).
"""
from __future__ import annotations

import sys
from typing import Any

import httpx

from app.config import load_settings
from app.llm.providers import build_provider
from app.llm.providers.base import LLMUnavailable, error_fields

_MAX_MESSAGE_CHARS = 240
_PROBE_SYSTEM = "Reply with the single word ok."
_PROBE_USER = "ok"


class RecordingClient(httpx.Client):
    """The adapter's HTTP client, plus the last status and JSON body."""

    def __init__(self) -> None:
        super().__init__(follow_redirects=False)
        self.http_status: int | None = None
        self.body: Any = None

    def post(self, url: Any, **kwargs: Any) -> httpx.Response:  # type: ignore[override]
        response = super().post(url, **kwargs)
        self.http_status = response.status_code
        try:
            self.body = response.json()
        except ValueError:
            self.body = None
        return response


def _safe_message(raw: object, api_key: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "-"
    text = " ".join(raw.split())
    key = (api_key or "").strip()
    if key:
        text = text.replace(key, "[redacted]")
    return text[:_MAX_MESSAGE_CHARS] or "-"


def _message_from_body(body: object) -> object:
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("message")
    if isinstance(err, str):
        return err
    return None


def format_line(
    *,
    provider: str,
    model: str,
    http_status: object,
    error_status: object,
    error_code: object,
    error_message: object,
    api_key: str,
) -> str:
    """One line: status and error fields only. Never the key or a prompt."""
    return (
        f"provider={provider} model={model} "
        f"http_status={http_status if http_status is not None else '-'} "
        f"error_status={error_status or '-'} "
        f"error_code={error_code or '-'} "
        f"error_message={_safe_message(error_message, api_key)}"
    )


def main() -> int:
    settings = load_settings()
    client = RecordingClient()
    provider = build_provider(settings, client=client)
    if provider is None:
        print(
            "no provider configured (set LLM_PROVIDER and the matching API key).",
            file=sys.stderr,
        )
        return 2

    error_message: object = None
    try:
        provider.complete(_PROBE_SYSTEM, [{"role": "user", "content": _PROBE_USER}])
    except LLMUnavailable as exc:
        error_message = _message_from_body(client.body)
        if error_message is None:
            error_message = exc.reason
        print(
            format_line(
                provider=provider.name,
                model=provider.model,
                http_status=client.http_status if client.http_status is not None else exc.http_status,
                error_status=exc.error_status,
                error_code=exc.error_code,
                error_message=error_message,
                api_key=settings.provider_api_key,
            )
        )
        return 1
    except httpx.HTTPError as exc:
        print(
            format_line(
                provider=provider.name,
                model=provider.model,
                http_status=client.http_status,
                error_status=type(exc).__name__,
                error_code="-",
                error_message=None,
                api_key=settings.provider_api_key,
            )
        )
        return 1

    err_status, err_code = error_fields(client.body)
    print(
        format_line(
            provider=provider.name,
            model=provider.model,
            http_status=client.http_status if client.http_status is not None else 200,
            error_status=err_status,
            error_code=err_code,
            error_message=None,
            api_key=settings.provider_api_key,
        )
    )
    return 0 if client.http_status == 200 or client.http_status is None else 1


if __name__ == "__main__":
    sys.exit(main())
