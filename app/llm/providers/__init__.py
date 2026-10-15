"""LLM provider adapters behind one interface (``Provider.complete``).

``build_provider(settings)`` returns the adapter for ``LLM_PROVIDER`` when its API key is
configured, else ``None`` (the service then keeps the offline null analyzer and placeholder
responder).
"""
from __future__ import annotations

import logging
from typing import Any

from app.llm.providers.anthropic_provider import AnthropicProvider
from app.llm.providers.base import LLMUnavailable, Provider
from app.llm.providers.google_provider import GoogleProvider
from app.llm.providers.openai_provider import OpenAIProvider

__all__ = [
    "AnthropicProvider", "GoogleProvider", "LLMUnavailable", "OpenAIProvider", "Provider",
    "build_provider",
]

_ADAPTERS = {"openai": OpenAIProvider, "anthropic": AnthropicProvider, "google": GoogleProvider}


def _quiet_http_loggers() -> None:
    # httpx logs every request URL at INFO; keep the libraries below warning level so a
    # log can never carry request details.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def build_provider(settings: Any, **overrides: Any) -> Provider | None:
    """The provider for ``settings.llm_provider``, or None when no key is configured."""
    key = settings.provider_api_key
    if not key:
        return None
    adapter = _ADAPTERS[settings.llm_provider]
    _quiet_http_loggers()
    return adapter(
        api_key=key, model=settings.llm_model, timeout=settings.llm_timeout_seconds, **overrides
    )
