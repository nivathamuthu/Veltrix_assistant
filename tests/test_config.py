from datetime import date
from pathlib import Path

import pytest

from app.config import ConfigError, load_settings


def test_defaults_point_at_data_folder():
    s = load_settings(env_file=None)
    assert s.corpus_dir == Path("./data/corpus")
    assert s.manifest_path == Path("./data/manifest.json")
    assert s.db_path == Path("./data/veltrix.db")
    assert s.llm_provider == "openai"
    assert s.app_today is None
    assert s.llm_min_interval_seconds == 0.0


def test_min_interval_defaults_to_zero_and_can_be_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "google")
    assert load_settings(env_file=None).llm_min_interval_seconds == 0.0
    monkeypatch.setenv("LLM_MIN_INTERVAL_SECONDS", "6")
    assert load_settings(env_file=None).llm_min_interval_seconds == 6.0


def test_app_today_overrides_system_date(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "2026-09-15")
    assert load_settings(env_file=None).today() == date(2026, 9, 15)


def test_system_date_used_when_app_today_empty(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "")
    assert load_settings(env_file=None).today() == date.today()


@pytest.mark.parametrize("bad", ["15-09-2026", "2026-13-01", "tomorrow"])
def test_invalid_app_today_rejected(monkeypatch, bad):
    monkeypatch.setenv("APP_TODAY", bad)
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_invalid_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "skynet")
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_repr_never_contains_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    s = load_settings(env_file=None)
    assert "sk-test-secret-value" not in repr(s)
    assert s.provider_api_key == "sk-test-secret-value"


def test_env_example_lists_every_variable():
    example = Path(__file__).resolve().parent.parent / ".env.example"
    listed = {
        line.split("=", 1)[0].strip()
        for line in example.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    config_src = (Path(__file__).resolve().parent.parent / "app" / "config.py").read_text(
        encoding="utf-8"
    )
    import re

    used = set(re.findall(r'_(?:get|parse_float)\(\s*"([A-Z_]+)"', config_src))
    assert used, "no variables found in config.py"
    assert used <= listed, f"missing from .env.example: {used - listed}"
