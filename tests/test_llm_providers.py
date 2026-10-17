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
from app.llm.providers.base import (
    DEADLINE_MARGIN_SECONDS,
    MIN_CALL_SECONDS,
    RETRY_DELAY_SECONDS,
    Provider,
    TransientError,
    retry_after_seconds,
)
from app.llm.provider_check import format_line
from app.llm.providers.google_provider import GEMINI_3_THINKING_LEVEL, MAX_OUTPUT_TOKENS, thinking_config_for
from app.llm.providers.openai_provider import accepts_temperature
from app.live_smoke import DEFAULT_PAUSE_SECONDS, _parse_args, parse_only, pause_seconds, select_queries


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


@pytest.mark.parametrize("status", [500, 502, 503])
def test_one_retry_on_timeout_or_5xx_then_success(status):
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(status, json={"error": "try later"})
        return httpx.Response(200, json=_openai_ok("recovered"))

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    assert p.complete("s", [{"role": "user", "content": "q"}]) == "recovered"
    assert hits["n"] == 2


def test_429_without_a_retry_delay_is_not_retried():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED", "code": 429}})

    p = GoogleProvider(api_key="k", model="gemini-x", client=_client(handler), sleep=lambda _: None)
    with pytest.raises(LLMUnavailable) as captured:
        p.complete("s", [{"role": "user", "content": "q"}])
    assert hits["n"] == 1
    assert captured.value.reason == "http_429"


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


def _google_ok(*parts, finish="STOP"):
    return {"candidates": [{"content": {"parts": list(parts)}, "finishReason": finish}]}


def test_google_puts_the_key_in_a_header_never_the_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_google_ok({"text": "ok"}))

    p = GoogleProvider(api_key="goog-key", model="gemini-x", client=_client(handler))
    assert p.complete("SYS", [{"role": "user", "content": "q"}]) == "ok"
    assert seen["key"] == "goog-key"
    assert "key=" not in seen["url"] and "goog-key" not in seen["url"]
    gen = seen["body"]["generationConfig"]
    assert gen["temperature"] == 0
    assert gen["maxOutputTokens"] == MAX_OUTPUT_TOKENS
    assert gen["thinkingConfig"] == {"thinkingBudget": 0}
    assert "thinkingLevel" not in gen["thinkingConfig"]
    assert "responseMimeType" not in gen


def test_google_request_disables_thinking_and_sets_json_mime_when_asked():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_google_ok({"text": '{"ok":true}'}))

    schema = {"title": "analysis", "type": "object", "properties": {"ok": {"type": "boolean"}}}
    p = GoogleProvider(api_key="k", model="gemini-3.8-flash", client=_client(handler))
    assert p.complete("s", [{"role": "user", "content": "q"}], json_schema=schema) == '{"ok":true}'
    gen = seen["body"]["generationConfig"]
    assert gen["temperature"] == 0
    assert gen["maxOutputTokens"] == MAX_OUTPUT_TOKENS
    assert gen["maxOutputTokens"] >= 1024
    assert gen["thinkingConfig"] == {"thinkingLevel": GEMINI_3_THINKING_LEVEL}
    assert "thinkingBudget" not in gen["thinkingConfig"]
    assert gen["responseMimeType"] == "application/json"
    system = seen["body"]["systemInstruction"]["parts"][0]["text"]
    assert "JSON Schema" in system


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gemini-3.8-flash", {"thinkingLevel": GEMINI_3_THINKING_LEVEL}),
        ("gemini-3.0-flash", {"thinkingLevel": GEMINI_3_THINKING_LEVEL}),
        ("gemini-2.5-flash", {"thinkingBudget": 0}),
        ("gemini-2.5-pro", {"thinkingBudget": 0}),
        ("gemini-x", {"thinkingBudget": 0}),
    ],
)
def test_google_sends_exactly_one_thinking_setting(model, expected):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_google_ok({"text": "ok"}))

    GoogleProvider(api_key="k", model=model, client=_client(handler)).complete(
        "s", [{"role": "user", "content": "q"}]
    )
    thinking = seen["body"]["generationConfig"]["thinkingConfig"]
    assert thinking == expected
    assert not ({"thinkingBudget", "thinkingLevel"} <= set(thinking))
    assert thinking_config_for(model) == expected


def test_provider_check_line_never_includes_the_key_or_the_prompt():
    secret = "goog-live-not-a-real-key"
    line = format_line(
        provider="google",
        model="gemini-3.8-flash",
        http_status=400,
        error_status="INVALID_ARGUMENT",
        error_code=400,
        error_message=f"bad thinkingConfig, key={secret}",
        api_key=secret,
    )
    assert "http_status=400" in line
    assert "error_status=INVALID_ARGUMENT" in line
    assert "error_code=400" in line
    assert secret not in line
    assert "Reply with" not in line
    assert "[redacted]" in line


def test_google_joins_answer_parts_and_ignores_thoughts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_google_ok(
                {"text": "hidden", "thought": True},
                {"thoughtSignature": "sig"},
                {"text": '{"intent":'},
                {"text": '"policy"}'},
            ),
        )

    p = GoogleProvider(api_key="k", model="gemini-x", client=_client(handler))
    assert p.complete("s", [{"role": "user", "content": "q"}]) == '{"intent":"policy"}'


@pytest.mark.parametrize(
    "payload, reason, finish",
    [
        ({"candidates": [{"content": {"parts": [{"text": "{"}]}, "finishReason": "MAX_TOKENS"}]}, "truncated", "MAX_TOKENS"),
        ({"candidates": [{"finishReason": "SAFETY"}]}, "blocked", "SAFETY"),
        ({"promptFeedback": {"blockReason": "BLOCKLIST"}, "candidates": []}, "blocked", "BLOCKLIST"),
        ({"candidates": [{"content": {"parts": [{"text": "only thinking", "thought": True}]}, "finishReason": "STOP"}]}, "empty", "STOP"),
        ({"candidates": []}, "empty", None),
    ],
)
def test_google_empty_blocked_or_truncated_is_a_clean_failure(payload, reason, finish):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    p = GoogleProvider(api_key="k", model="gemini-x", client=_client(handler))
    with pytest.raises(LLMUnavailable) as captured:
        p.complete("s", [{"role": "user", "content": "q"}])
    assert captured.value.reason == reason
    if finish:
        assert captured.value.finish_reason == finish
    assert "prompt" not in str(captured.value).lower()


def test_min_interval_paces_successive_calls():
    slept: list[float] = []
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, json=_openai_ok("ok"))

    p = OpenAIProvider(
        api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=slept.append, min_interval=2.5
    )
    assert p.complete("s", [{"role": "user", "content": "q"}]) == "ok"
    assert p.complete("s", [{"role": "user", "content": "q"}]) == "ok"
    assert hits["n"] == 2
    assert slept and slept[-1] == pytest.approx(2.5, abs=0.05)


def test_429_waits_retry_after_then_succeeds():
    hits = {"n": 0}
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(
                429,
                json={"error": {"status": "RESOURCE_EXHAUSTED", "code": 429, "message": "Please retry in 8.00s"}},
                headers={"Retry-After": "8"},
            )
        return httpx.Response(200, json=_openai_ok("ok"))

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=slept.append)
    assert p.complete("s", [{"role": "user", "content": "q"}]) == "ok"
    assert hits["n"] == 2
    assert slept == [8.0]


def test_429_with_retry_after_past_the_deadline_is_not_retried():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(
            429,
            json={"error": {"status": "RESOURCE_EXHAUSTED", "code": 429}},
            headers={"Retry-After": "45"},
        )

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    with deadline_of(20):
        with pytest.raises(LLMUnavailable) as captured:
            p.complete("s", [{"role": "user", "content": "q"}])
    assert hits["n"] == 1
    assert captured.value.reason == "deadline"


def test_retry_after_seconds_reads_header_and_google_retry_delay():
    header = httpx.Response(429, json={}, headers={"Retry-After": "12"})
    assert retry_after_seconds(header, {}) == 12.0
    body = {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7.25s"}],
        }
    }
    assert retry_after_seconds(httpx.Response(429, json=body), body) == 7.25
    proto = {"error": {"details": [{"retryDelay": {"seconds": 4, "nanos": 500000000}}]}}
    assert retry_after_seconds(httpx.Response(429, json=proto), proto) == 4.5


def test_429_with_google_retry_delay_waits_then_retries_once():
    hits = {"n": 0}
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [{"retryDelay": "3s"}],
                    }
                },
            )
        return httpx.Response(200, json=_google_ok({"text": "ok"}))

    p = GoogleProvider(api_key="k", model="gemini-x", client=_client(handler), sleep=slept.append)
    assert p.complete("s", [{"role": "user", "content": "q"}]) == "ok"
    assert hits["n"] == 2
    assert slept == [3.0]


def test_retry_is_skipped_when_the_deadline_has_almost_run_out():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(503, json={})

    p = OpenAIProvider(api_key="k", model="gpt-4o-mini", client=_client(handler), sleep=lambda _: None)
    with deadline_of(MIN_CALL_SECONDS + DEADLINE_MARGIN_SECONDS + RETRY_DELAY_SECONDS - 0.2):
        with pytest.raises(LLMUnavailable) as captured:
            p.complete("s", [{"role": "user", "content": "q"}])
    assert hits["n"] == 1
    assert captured.value.reason == "deadline"


def test_failed_call_logs_status_codes_attempt_and_elapsed_never_secrets(caplog):
    secret_key = "goog-live-not-a-real-key"
    secret_prompt = "zebra-quartz-prompt-17"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": secret_prompt,
                }
            },
        )

    p = GoogleProvider(api_key=secret_key, model="gemini-x", client=_client(handler), sleep=lambda _: None)
    caplog.set_level(logging.WARNING)
    with pytest.raises(LLMUnavailable):
        p.complete(secret_prompt, [{"role": "user", "content": secret_prompt}])
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm_failed")]
    assert len(lines) == 1  # 429 with no retry delay is not retried
    first = lines[0]
    assert "provider=google" in first
    assert "http_status=429" in first
    assert "error_status=RESOURCE_EXHAUSTED" in first
    assert "error_code=429" in first
    assert "finish_reason=-" in first
    assert "attempt=1" in first
    assert "elapsed=" in first
    text = "\n".join(lines)
    for leaked in (secret_key, secret_prompt, "zebra"):
        assert leaked not in text


def test_live_smoke_pause_defaults_to_four_seconds_and_can_be_disabled():
    assert pause_seconds({}) == DEFAULT_PAUSE_SECONDS == 12.0
    assert pause_seconds({"LIVE_SMOKE_NO_PAUSE": "1"}) == 0.0
    assert pause_seconds({"LIVE_SMOKE_NO_PAUSE": "true"}) == 0.0
    assert pause_seconds({"LIVE_SMOKE_PAUSE_SECONDS": "0"}) == 0.0
    assert pause_seconds({"LIVE_SMOKE_PAUSE_SECONDS": "2.5"}) == 2.5


def test_live_smoke_only_selects_the_given_query_numbers():
    assert parse_only("") is None
    assert parse_only("1,11,12,13") == frozenset({1, 11, 12, 13})
    assert parse_only(" 2, 2, 5 ") == frozenset({2, 5})
    with pytest.raises(ValueError):
        parse_only("1,policy")
    rows = [(1, "a", "s", "q1"), (2, "a", "s", "q2"), (11, "m", "s", "q11")]
    assert [n for n, *_ in select_queries(rows, parse_only("1,11"))] == [1, 11]
    assert select_queries(rows, None) == rows
    assert _parse_args(["--only", "1,11,12,13"]).only == "1,11,12,13"


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
