from fastapi import APIRouter, Request

from zepiris.deps import SettingsDep

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(settings: SettingsDep) -> dict[str, str]:
    """Lightweight readiness: config loaded. Extend with MinIO/Milvus pings if needed."""
    _ = settings
    return {"status": "ready"}


@router.get("/metrics")
def metrics(request: Request) -> dict:
    """API-side saturation counters, mirroring the ML service's /metrics.

    ``reference_bytes_cache`` is the one to watch during a load run: its hit
    rate is the fraction of requests that skipped the reference's S3 round
    trip entirely. Near zero on repeat traffic means the cache is disabled or
    undersized, and every request is paying an S3 GET it does not need.
    """
    fetcher = getattr(request.app.state, "s3_fetcher", None)
    limiter = getattr(request.app.state, "api_limiter", None)
    return {
        "admission": limiter.snapshot() if limiter is not None else {"enabled": False},
        "reference_bytes_cache": (
            fetcher.cache_snapshot() if fetcher is not None else {"enabled": False}
        ),
    }
