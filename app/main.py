"""FastAPI app: /health now; /chat arrives in later phases."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.ingestion.embeddings import Embedder
from app.ingestion.pipeline import run_ingestion
from app.ingestion.state import IngestionState
from app.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    """``embedder`` lets tests skip the real model; production leaves it None."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    state = IngestionState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("service starting (llm_provider=%s)", settings.llm_provider)
        # Ingestion runs in the background so the port opens immediately and
        # /health can answer 503 while it works.
        task = asyncio.create_task(asyncio.to_thread(run_ingestion, settings, state, embedder))
        app.state.ingestion_task = task
        try:
            yield
        finally:
            # A worker thread cannot be cancelled; do not block shutdown on it.
            task.cancel()

    app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)
    app.state.settings = settings
    app.state.ingestion = state

    @app.get("/health")
    async def health() -> JSONResponse:
        if state.ready:
            return JSONResponse({"status": "ok"})
        # Constant body: never expose internal error text on a public endpoint.
        return JSONResponse({"status": "starting"}, status_code=503)

    return app


app = create_app()
