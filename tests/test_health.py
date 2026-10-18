import threading
import time

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from tests.fake_embedder import FakeEmbedder


def _wait_for(client: TestClient, status: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health").status_code == status:
            return
        time.sleep(0.05)
    raise AssertionError(f"/health never returned {status}")


def test_root_serves_the_chat_ui(settings, fake_embedder):
    app = create_app(settings, embedder=fake_embedder)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "PeopleOps" in response.text
        assert 'id="employee"' in response.text
        assert 'id="session"' in response.text
        assert "New chat" in response.text
        assert 'id="scope"' in response.text
        assert "/me" in response.text
        assert "E4004" in response.text
        assert "direct reports" in response.text
        assert "cannot approve" in response.text
        assert "Who in my team is on leave next week?" in response.text
        assert "source-id" in response.text
        assert "Section: " in response.text


def test_health_is_503_until_ingestion_finishes_then_200(settings, monkeypatch, fake_embedder):
    monkeypatch.setenv("INGEST_STARTUP_DELAY_SECONDS", "1.0")
    app = create_app(load_settings(env_file=None), embedder=fake_embedder)
    with TestClient(app) as client:
        early = client.get("/health")
        assert early.status_code == 503
        _wait_for(client, 200)
        late = client.get("/health")
        assert late.status_code == 200
        assert late.json() == {"status": "ok"}


def test_health_turns_200_only_after_the_index_is_built(settings):
    """While embedding is still running /health must stay 503; the index is
    attached to the state before it flips to 200."""
    release = threading.Event()

    class SlowEmbedder(FakeEmbedder):
        def embed_passages(self, texts):
            release.wait(timeout=10)
            return super().embed_passages(texts)

    app = create_app(settings, embedder=SlowEmbedder())
    with TestClient(app) as client:
        time.sleep(0.5)  # ingestion is now blocked inside the embedder
        assert client.get("/health").status_code == 503
        assert app.state.ingestion.index is None
        release.set()
        _wait_for(client, 200)
        assert app.state.ingestion.index is not None
        assert len(app.state.ingestion.index) >= 1


def test_health_stays_503_when_inputs_missing(settings, monkeypatch, tmp_path, fake_embedder):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "nope.db"))
    app = create_app(load_settings(env_file=None), embedder=fake_embedder)
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and app.state.ingestion.error is None:
            time.sleep(0.05)
        assert app.state.ingestion.error is not None
        response = client.get("/health")
        assert response.status_code == 503
        # internal error text must not leak through the public endpoint
        assert "nope.db" not in response.text


def test_missing_embedding_model_fails_clearly_and_stays_503(settings, monkeypatch, tmp_path):
    """No injected embedder and an empty model cache: a clear error, no download."""
    monkeypatch.setenv("EMBED_CACHE_DIR", str(tmp_path / "empty_model_cache"))
    app = create_app(load_settings(env_file=None))
    with TestClient(app) as client:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and app.state.ingestion.error is None:
            time.sleep(0.05)
        error = app.state.ingestion.error
        assert error and "EmbeddingModelUnavailable" in error
        assert "EMBED_CACHE_DIR" in error  # tells the operator what to fix
        response = client.get("/health")
        assert response.status_code == 503
        assert "EMBED" not in response.text  # nothing internal on the public endpoint


def test_ingestion_primes_log_redactor_from_db(settings, fake_embedder):
    from app.logging_setup import redact

    app = create_app(settings, embedder=fake_embedder)
    with TestClient(app) as client:
        _wait_for(client, 200)
    assert "Testperson" not in redact("loaded Zed Testperson")
