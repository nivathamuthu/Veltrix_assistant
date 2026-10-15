import time

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app


def _wait_for(client: TestClient, status: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health").status_code == status:
            return
        time.sleep(0.05)
    raise AssertionError(f"/health never returned {status}")


def test_health_is_503_until_ingestion_finishes_then_200(settings, monkeypatch):
    monkeypatch.setenv("INGEST_STARTUP_DELAY_SECONDS", "1.0")
    app = create_app(load_settings(env_file=None))
    with TestClient(app) as client:
        early = client.get("/health")
        assert early.status_code == 503
        _wait_for(client, 200)
        late = client.get("/health")
        assert late.status_code == 200
        assert late.json() == {"status": "ok"}


def test_health_stays_503_when_inputs_missing(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "nope.db"))
    app = create_app(load_settings(env_file=None))
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and app.state.ingestion.error is None:
            time.sleep(0.05)
        assert app.state.ingestion.error is not None
        response = client.get("/health")
        assert response.status_code == 503
        # internal error text must not leak through the public endpoint
        assert "nope.db" not in response.text


def test_ingestion_primes_log_redactor_from_db(settings):
    from app.logging_setup import redact

    app = create_app(settings)
    with TestClient(app) as client:
        _wait_for(client, 200)
    assert "Testperson" not in redact("loaded Zed Testperson")
