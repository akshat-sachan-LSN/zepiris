import os
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zepiris.version import __version__ as _package_version


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZEPIRIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "ZepIris"
    api_version: str = _package_version
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    verify_threshold: float = 0.5
    # Documents (Aadhaar/PAN) carry small/printed photos that embed weaker, so
    # the doc-match endpoint defaults to a more lenient threshold than face-match.
    doc_verify_threshold: float = 0.4
    # Minimum sharpness (variance-of-Laplacian) for the face extracted from a
    # document. Blurry phone captures of a card embed poorly and drag match
    # scores down; below this the docmatch response flags ``lowQuality`` and, if
    # the value is > 0, rejects with 422 so the caller can ask for a retake.
    # 0 = disabled (flag only, never reject). A sharp ID photo is typically
    # > 100; observed blurry captures fall in the 5–20 range.
    doc_min_sharpness: float = 0.0
    reference_fetch_timeout_seconds: float = 10.0
    # CPU embedding (antelopev2/ResNet100) plus the multi-pass detection cascade
    # can take well over httpx's 5s default on hard document images.
    ml_inference_timeout_seconds: float = 60.0

    # -- throughput / concurrency -------------------------------------------
    #: How this process scores a pair. "remote" (default) sends both images to
    #: the ML service in one call, keeping the two services independently
    #: scalable. "local" loads the models here instead — no HTTP hop at all,
    #: lowest latency, but it needs the ML extras installed alongside the API.
    match_mode: str = "remote"
    #: Face tier used when match_mode="local" — see zepiris.ml_inference.face_engine.
    local_face_tier: str = "balanced"
    local_device: str = "cpu"
    local_intra_op_threads: int = 1
    #: Connection-pool ceiling toward the ML service. httpx keeps only 20
    #: connections alive by default, so past that every request pays a fresh
    #: handshake exactly when load is highest. Size this at or above peak
    #: in-flight requests per API process.
    ml_max_connections: int = 200
    #: Connection-pool ceiling for reference-image fetches (two per request).
    s3_max_connections: int = 200
    #: Byte budget for the in-process reference-bytes cache (0 = off). The
    #: enrolled selfie is refetched from S3 on every verification even though
    #: the ML service already holds its embedding; caching the bytes here
    #: removes that GET — and S3's latency tail with it — for every repeat
    #: verification. Safe because production reference URLs are content-unique
    #: object keys; see services/s3_fetcher.py. 268435456 (256 MiB) holds
    #: roughly 800 typical references.
    s3_cache_max_bytes: int = 0
    #: How long a cached reference body may be served before it is refetched.
    #: A safety valve for overwritten keys, not a tuning knob.
    s3_cache_ttl_seconds: float = 900.0
    #: Worker threads for the few remaining sync call sites. This process is
    #: I/O-bound — it fetches images and awaits the ML service — so it needs far
    #: fewer threads than it carries concurrent requests.
    thread_pool_size: int = 64
    #: Uvicorn worker processes. The API is I/O-bound, so a couple of workers
    #: saturate a small instance; the CPU cost lives in the ML service.
    api_workers: int = 2
    #: Admission control: in-flight requests **per worker** above which the API
    #: answers 503 immediately instead of accepting the work. 0 disables it.
    #: Enforced by ASGI middleware (zepiris/api/concurrency.py), NOT uvicorn's
    #: limit_concurrency — that knob counts idle keep-alive connections, so a
    #: load balancer's warm pool or a few hundred connected clients trips it
    #: while the box is doing nothing.
    #:
    #: This is the only place end-to-end latency can be bounded. The ML service's
    #: inference limiter sheds requests that reach its handler, but under real
    #: overload the queue forms earlier — in the connection backlog, where nothing
    #: times out. Measured: with the inference queue timeout cut to 1s, sheds fell
    #: to zero while p95 rose to 14s, because the waiting had simply moved
    #: upstream of the thing doing the shedding.
    #:
    #: Size it from the latency you will accept, not from capacity: queue delay is
    #: in-flight / throughput, so at ~120 req/s a limit of 32 per worker across 2
    #: workers bounds the wait at roughly 64/120 = 0.5s. Set it too high and it
    #: stops being admission control; set it below the in-flight count a healthy
    #: load carries (throughput x latency, ~16 requests here) and it sheds traffic
    #: the service could have served.
    api_limit_concurrency: int = 0
    #: Per-request access logging. One line per request is a rounding error at
    #: 10 req/s and roughly 13 GB/day at 1000 — and it duplicates what the load
    #: balancer already records, with none of its retention controls. Off by
    #: default; the ALB is the request log.
    access_log: bool = False

    # -- adaptive learning (online threshold calibration) --------------------
    # Every scored verification is logged (scores only, never images) and the
    # /feedback endpoint lets operators confirm outcomes; per-document-type
    # thresholds (aadhaar, pan, ...) are then re-fit from the labelled scores.
    #: Threshold calibration appends one JSON line per scored verification —
    #: roughly 17 GB/day at 1000 req/s, on a service that otherwise persists
    #: nothing. It is also process-local, so on an autoscaled fleet the file dies
    #: with the instance and the samples are never joined with feedback anyway.
    #: Enable it only with a deliberate destination (shared volume or a real
    #: datastore) and a retention policy.
    learning_enabled: bool = False
    learning_dir: str = "learning"
    learning_min_genuine: int = 20
    learning_min_impostor: int = 20
    learning_far_target: float = 0.01
    reference_max_bytes: int = 5 * 1024 * 1024  # mirrors MAX_IMAGE_SIZE_BYTES

    #: Required. ML inference base URL; IQA uses POST /v1/iqa/assess, embeddings POST /v1/face/embed.
    #: Also accepts legacy env ML_INFERENCE_SERVICE_URL (no ZEPIRIS_ prefix) if this is unset.
    ml_inference_service_url: str = Field(default="", description="e.g. http://ml-inference:8001")

    @model_validator(mode="after")
    def _require_ml_inference_url(self) -> "Settings":
        u = (self.ml_inference_service_url or "").strip()
        if not u:
            u = (os.environ.get("ML_INFERENCE_SERVICE_URL") or "").strip()
        if not u:
            if self.match_mode == "local":
                # match_mode=local runs the models in this process, so there is no
                # ML service to point at. Requiring a URL here would force a
                # meaningless value into every single-container deployment.
                return self
            raise ValueError(
                "ZEPIRIS_ML_INFERENCE_SERVICE_URL is required "
                "(e.g. http://ml-inference:8001 or http://localhost:8001). "
                "Alternatively set ML_INFERENCE_SERVICE_URL, "
                "or set ZEPIRIS_MATCH_MODE=local to run the models in-process."
            )
        self.ml_inference_service_url = u
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
