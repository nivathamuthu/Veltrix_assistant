"""Environment configuration.

Every variable read here must also be listed in ``.env.example``.
Nothing in this module may contain corpus facts (leave days, holiday dates, ...).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROVIDERS = ("openai", "anthropic", "google")

# Environment variable that holds the API key for each provider.
PROVIDER_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


def _get(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    manifest_path: Path
    db_path: Path
    index_dir: Path
    app_today: date | None  # None -> use the system date
    llm_provider: str
    llm_model: str
    embed_model: str
    openai_api_key: str
    anthropic_api_key: str
    google_api_key: str
    log_level: str
    ingest_startup_delay_seconds: float
    llm_timeout_seconds: float
    request_timeout_seconds: float

    def today(self) -> date:
        """Reference "today": APP_TODAY if set, else the system date."""
        return self.app_today or date.today()

    @property
    def provider_api_key(self) -> str:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
        }[self.llm_provider]

    def __repr__(self) -> str:  # never leak keys through repr / logs
        return (
            f"Settings(corpus_dir={self.corpus_dir!s}, manifest_path={self.manifest_path!s}, "
            f"db_path={self.db_path!s}, index_dir={self.index_dir!s}, "
            f"app_today={self.app_today}, llm_provider={self.llm_provider}, "
            f"llm_model={self.llm_model!r}, embed_model={self.embed_model!r})"
        )


def _parse_float(name: str, default: str) -> float:
    raw = _get(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {raw!r}")
    return value


def _parse_today(raw: str) -> date | None:
    if raw == "":
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(f"APP_TODAY must be YYYY-MM-DD, got {raw!r}") from exc


def load_settings(env_file: str | None = ".env") -> Settings:
    """Build Settings from the process environment (and an optional .env file).

    Real environment variables always win over values in the .env file.
    """
    if env_file:
        load_dotenv(env_file, override=False)

    provider = _get("LLM_PROVIDER", "openai").lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")

    return Settings(
        corpus_dir=Path(_get("CORPUS_DIR", "./data/corpus")),
        manifest_path=Path(_get("MANIFEST_PATH", "./data/manifest.json")),
        db_path=Path(_get("DB_PATH", "./data/veltrix.db")),
        index_dir=Path(_get("INDEX_DIR", "./index")),
        app_today=_parse_today(_get("APP_TODAY", "")),
        llm_provider=provider,
        llm_model=_get("LLM_MODEL", ""),
        embed_model=_get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        openai_api_key=_get("OPENAI_API_KEY", ""),
        anthropic_api_key=_get("ANTHROPIC_API_KEY", ""),
        google_api_key=_get("GOOGLE_API_KEY", ""),
        log_level=_get("LOG_LEVEL", "INFO").upper(),
        ingest_startup_delay_seconds=_parse_float("INGEST_STARTUP_DELAY_SECONDS", "0"),
        llm_timeout_seconds=_parse_float("LLM_TIMEOUT_SECONDS", "20"),
        request_timeout_seconds=_parse_float("REQUEST_TIMEOUT_SECONDS", "28"),
    )
