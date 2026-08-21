from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import anyio.to_thread
import httpx
from fastapi import FastAPI

from zepiris.api.routes import build_api_router
from zepiris.config import get_settings
from zepiris.exception_handlers import register_exception_handlers
from zepiris.services.embedding import MLInferenceEmbeddingService
from zepiris.services.iqa import MLInferenceIQAService
from zepiris.services.learning import AdaptiveThresholdLearner
from zepiris.services.matching import LocalFaceMatcher, RemoteFaceMatcher
from zepiris.services.ml_client import AsyncMLInferenceClient, MLInferenceClient
from zepiris.services.s3_fetcher import S3ImageFetcher

logger = logging.getLogger(__name__)


def _build_matcher(settings, async_client: AsyncMLInferenceClient):
    """Choose how this process scores a pair.

    ``local`` loads the models into this process: no HTTP hop, no image crossing
    a socket, the lowest latency available. It needs the ML extras installed here
    and gives up scaling the API and the models independently, so it suits a
    single-container deployment. ``remote`` (default) keeps the two services
    split and sends both images in one call.

    A failed local load falls back to remote rather than leaving the service with
    no way to match at all.
    """
    if settings.match_mode == "local":
        try:
            from zepiris.ml_inference.face_embedding import FaceEmbeddingService

            service = FaceEmbeddingService(
                device=settings.local_device,
                tier=settings.local_face_tier,
                intra_op_threads=settings.local_intra_op_threads,
            )
            service.load_model()
            logger.info("Match mode: local (models in-process, tier=%s)", settings.local_face_tier)
            return LocalFaceMatcher(service)
        except Exception:
            logger.exception("match_mode=local failed to load models; falling back to remote")
    logger.info("Match mode: remote (%s)", settings.ml_inference_service_url)
    return RemoteFaceMatcher(async_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # In local mode there is no ML service; the clients are still constructed so
    # the legacy IQA/embedding call sites keep a valid object, but nothing on the
    # verification path touches them.
    base = (settings.ml_inference_service_url or "http://localhost:8001").rstrip("/")
    ml_client = MLInferenceClient(base, timeout_seconds=settings.ml_inference_timeout_seconds)
    async_ml_client = AsyncMLInferenceClient(
        base,
        timeout_seconds=settings.ml_inference_timeout_seconds,
        max_connections=settings.ml_max_connections,
    )

    fetch_client = httpx.AsyncClient(
        timeout=settings.reference_fetch_timeout_seconds,
        limits=httpx.Limits(
            max_connections=settings.s3_max_connections,
            max_keepalive_connections=settings.s3_max_connections,
            keepalive_expiry=60.0,
        ),
        follow_redirects=True,
    )
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
    app.state.matcher = _build_matcher(settings, async_ml_client)

    # This process is I/O-bound — it fetches images and awaits the ML service —
    # so its thread pool only backs the few remaining sync call sites. Starlette's
    # 40-thread default is the ceiling on how many of those can run at once, well
    # under the concurrency this service is expected to carry.
    anyio.to_thread.current_default_thread_limiter().total_tokens = settings.thread_pool_size

    yield

    ml_client.close()
    await async_ml_client.aclose()
    await s3_fetcher.aclose()


def create_app() -> FastAPI:
    from zepiris.logging_config import configure_logging

    configure_logging()
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
        workers=settings.api_workers,
        access_log=settings.access_log,
        # Refuse work past this many in-flight requests per worker rather than
        # queueing it in the connection backlog, where it would wait with no
        # timeout and no visibility. See api_limit_concurrency in config.py.
        limit_concurrency=settings.api_limit_concurrency or None,
        reload=False,
    )
