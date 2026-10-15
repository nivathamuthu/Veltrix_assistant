"""Provider adapters: one interface, temperature 0, one retry, no secrets in logs."""
from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.config import load_settings
from app.llm.deadline import deadline_of
from app.llm.providers import (
    AnthropicProvider,
    GoogleProvider,
    LLMUnavailable,
    OpenAIProvider,
    build_provider,
)
from app.llm.providers.base import Provider, TransientError
from app.llm.providers.openai_provider import accepts_temperature


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _openai_ok(content="hello"):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


# --- the interface ----------------------------------------------------------------------------------------------


def test_complete_is_the_only_public_method():
    assert hasattr(Provider, "complete")
    assert OpenAIProvider.complete is Provider.complete
    assert AnthropicProvider.complete is Provider.complete
    assert GoogleProvider.complete is Provider.complete


def test_repr_and_str_never_show_the_key():
    p = OpenAIProvider(api_key="sk-super-secret", model="gpt-4o-mini")
    assert "sk-super-secret" not in repr(p) and "sk-super-secret" not in str(p)


def test_an_empty_key_is_rejected():
    with pytest.raises(ValueError):
        OpenAIProvider(api_key="  ")


def test_build_provider_is_none_without_a_key(settings):
    assert build_provider(settings) is None


def test_build_provider_picks_the_adapter_for_the_configured_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    monkeypatch.setenv("MANIFEST_PATH", str(tmp_path / "m.json"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    settings = load_settings(env_file=None)
    provider = build_provider(settings)
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"


# --- OpenAI (fully implemented) --------------------------------------------------------------------------------


def test_openai_sends_temperature_0_and_the_model_and_returns_the_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_ok("the answer"))

    p = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", client=_client(handler))
    assert p.complete("sys", [{"role": "user", "content": "hi"}]) == "the answer"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_sends_a_strict_json_schema_when_asked():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_ok('{"ok":true}'))

    schema = {"title": "analysis", "type": "object", "properties": {"ok": {"type": "boolean"}}}
    OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler)).complete(
        "s", [{"role": "user", "content": "q"}], json_schema=schema
    )
    fmt = seen["body"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "analysis"
    assert "title" not in fmt["json_schema"]["schema"]


@pytest.mark.parametrize("model", ["o1-mini", "o3-mini", "gpt-5"])
def test_reasoning_models_are_called_without_a_temperature(model):
    assert accepts_temperature(model) is False
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_ok("x"))

    OpenAIProvider(api_key="k", model=model, client=_client(handler)).complete("s", [{"role": "user", "content": "q"}])
    assert "temperature" not in seen["body"]


# --- retry / failures -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_one_retry_on_timeout_or_429_or_5xx_then_success(status):
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(status, json={"error": "try later"})
        return httpx.Response(200, json=_openai_ok("recovered"))

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    assert p.complete("s", [{"role": "user", "content": "q"}]) == "recovered"
    assert hits["n"] == 2


def test_a_second_failure_becomes_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={})

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    with pytest.raises(LLMUnavailable) as captured:
        p.complete("s", [{"role": "user", "content": "q"}])
    assert captured.value.reason == "http_503"
    assert "prompt" not in str(captured.value).lower()


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_4xx_other_than_429_is_not_retried(status):
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(status, json={"error": "no"})

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    with pytest.raises(LLMUnavailable) as captured:
        p.complete("s", [{"role": "user", "content": "q"}])
    assert hits["n"] == 1
    assert captured.value.reason == f"http_{status}"


def test_a_timeout_is_retried_once():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            raise httpx.TimeoutException("slow")
        return httpx.Response(200, json=_openai_ok("ok"))

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    assert p.complete("s", [{"role": "user", "content": "q"}]) == "ok"
    assert hits["n"] == 2


def test_a_bad_payload_is_unavailable_not_a_stack_trace():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler))
    with pytest.raises(LLMUnavailable) as captured:
        p.complete("s", [{"role": "user", "content": "q"}])
    assert captured.value.reason == "bad_response"


def test_a_deadline_that_has_already_passed_is_not_a_network_call():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, json=_openai_ok("late"))

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler))
    with deadline_of(0):
        with pytest.raises(LLMUnavailable) as captured:
            p.complete("s", [{"role": "user", "content": "q"}])
    assert captured.value.reason == "deadline"
    assert hits["n"] == 0


# --- thin adapters ------------------------------------------------------------------------------------------------


def test_anthropic_puts_the_schema_in_the_system_prompt_and_temperature_0():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    schema = {"title": "analysis", "type": "object"}
    p = AnthropicProvider(api_key="ant-key", model="claude-x", client=_client(handler))
    assert p.complete("SYS", [{"role": "user", "content": "q"}], json_schema=schema) == "ok"
    assert seen["key"] == "ant-key"
    assert seen["body"]["temperature"] == 0
    assert "SYS" in seen["body"]["system"]
    assert "JSON Schema" in seen["body"]["system"]
    assert seen["url"].endswith("/v1/messages")


def test_google_puts_the_key_in_a_header_never_the_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        )

    p = GoogleProvider(api_key="goog-key", model="gemini-x", client=_client(handler))
    assert p.complete("SYS", [{"role": "user", "content": "q"}]) == "ok"
    assert seen["key"] == "goog-key"
    assert "key=" not in seen["url"] and "goog-key" not in seen["url"]
    assert seen["body"]["generationConfig"]["temperature"] == 0


# --- logs never carry prompts, responses or keys ------------------------------------------------------------------


def test_logs_never_contain_the_key_the_prompt_or_the_response(caplog):
    secret_key = "sk-live-not-a-real-key"
    secret_prompt = "zebra-quartz-prompt-17"
    secret_reply = "zebra-quartz-reply-19"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": secret_prompt})

    p = OpenAIProvider(api_key=secret_key, model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(LLMUnavailable):
        p.complete(secret_prompt, [{"role": "user", "content": secret_prompt}])
    text = "\n".join(r.getMessage() for r in caplog.records)
    for leaked in (secret_key, secret_prompt, secret_reply, "Bearer"):
        assert leaked not in text


def test_llm_unavailable_carries_only_a_short_reason():
    err = LLMUnavailable("timeout")
    assert err.reason == "timeout"
    assert "timeout" in str(err)
    assert "httpx" not in str(err)
    # the type is the only exception complete() is documented to raise
    assert issubclass(LLMUnavailable, RuntimeError)
    assert TransientError is not LLMUnavailable
