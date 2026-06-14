from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from zepiris.api.routes import build_api_router
from zepiris.config import get_settings
from zepiris.exception_handlers import register_exception_handlers
from zepiris.services.embedding import MLInferenceEmbeddingService
from zepiris.services.iqa import MLInferenceIQAService
from zepiris.services.learning import AdaptiveThresholdLearner
from zepiris.services.ml_client import MLInferenceClient
from zepiris.services.s3_fetcher import S3ImageFetcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    base = settings.ml_inference_service_url.rstrip("/")
    ml_client = MLInferenceClient(base, timeout_seconds=settings.ml_inference_timeout_seconds)

    fetch_client = httpx.Client(timeout=settings.reference_fetch_timeout_seconds)
    s3_fetcher = S3ImageFetcher(client=fetch_client, max_bytes=settings.reference_max_bytes)

    app.state.iqa = MLInferenceIQAService(ml_client)
    app.state.embedding = MLInferenceEmbeddingService(ml_client)
    app.state.s3_fetcher = s3_fetcher
    app.state.learner = AdaptiveThresholdLearner(
        settings.learning_dir,
        min_genuine=settings.learning_min_genuine,
        min_impostor=settings.learning_min_impostor,
        far_target=settings.learning_far_target,
        enabled=settings.learning_enabled,
    )

    yield

    ml_client.close()
    s3_fetcher.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(build_api_router())
    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "zepiris.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
