"""FastAPI application for ML inference microservice.

Run standalone:
    uvicorn zepiris.ml_inference.app:app --host 0.0.0.0 --port 8001

Or via the entry point:
    zepiris-ml-inference-api
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import anyio.to_thread
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from zepiris.ml_inference.blur_detection import BlurDetectionService
from zepiris.ml_inference.concurrency import InferenceLimiter
from zepiris.ml_inference.face_embedding import FaceEmbeddingService
from zepiris.ml_inference.image_quality_assessment import (
    ImageQualityAssessmentService,
)
from zepiris.ml_inference.moire_detection import ScreenReplayDetector
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
from zepiris.ml_inference.onnx_spoof_detection import OnnxSpoofDetectionService
from zepiris.ml_inference.spoof_detection import SpoofDetectionService
from zepiris.version import __version__ as package_version

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings (env prefix ML_SERVICE_ so the inference container has its own env)
# ---------------------------------------------------------------------------
class MLServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ML_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8001

    ml_device: str = "cpu"

    # Face-match-only mode: skip loading NSFW/spoof/blur models so the service
    # starts faster and lighter. IQA is disabled; only face embedding loads.
    face_match_only: bool = True

    nsfw_model_source: str = "auto"
    nsfw_hf_repo_id: str = ""
    nsfw_hf_model_file: str = "nsfw_model.pth"
    nsfw_local_model_path: str = "/app/models/nsfw_model.pth"
    nsfw_threshold: float = 0.5

    # Liveness engine: "onnx" = MiniFASNet (Silent-Face) ensemble, recommended,
    # purpose-built for print/replay attacks; "mobilenet" = legacy MobileNetV3.
    spoof_engine: str = "onnx"
    # Ensemble of two complementary MiniFASNet models (canonical Silent-Face):
    # V2 at 2.7x crop + V1SE at 4.0x crop. The second path is optional — if it
    # is missing, the engine runs single-model. Averaged prob_live is used.
    spoof_onnx_model_path: str = "/app/models/minifasnet_v2_yakhyo.onnx"
    spoof_onnx_model_path_2: str = "/app/models/minifasnet_v1se_yakhyo.onnx"
    # prob_live (MiniFASNet class 1 = real) must exceed this to be considered
    # live. Real faces tested at 0.62-1.0; 0.4 widens the live band to reduce
    # false liveness rejections on borderline real selfies.
    spoof_onnx_live_threshold: float = 0.4

    # Legacy MobileNetV3 spoof model (used only when spoof_engine == "mobilenet").
    spoof_model_source: str = "auto"
    spoof_hf_repo_id: str = ""
    spoof_hf_model_file: str = "spoof_model.pth"
    spoof_local_model_path: str = "/app/models/spoof_model.pth"
    spoof_threshold: float = 0.7
    # Passive moiré/glare screen-replay detector. Enabled by default; an image
    # flagged as a recaptured screen is forced not-live regardless of the model.
    spoof_screen_detection_enabled: bool = True
    spoof_screen_threshold: float = 0.5

    blur_model_source: str = "auto"
    blur_hf_repo_id: str = ""
    blur_hf_model_file: str = "blur_model.pth"
    blur_local_model_path: str = "/app/models/blur_model.pth"
    blur_threshold: float = 0.5

    face_embedding_dim: int = 512
    # Detector input size. 512x512 detects large selfie faces and (via the
    # low-thresh + upscale cascade) small document faces reliably, at lower
    # latency than 640. Match scores are effectively unchanged (measured 0.750
    # @640 vs 0.743 @512). Safe now that the liveness gate is off — nothing
    # crops off this box anymore. Raise to 640 only if small faces are missed.
    face_detection_width: int = 512
    face_detection_height: int = 512
    face_area_threshold: float = 0.01
    face_enable_padding_retry: bool = True
    face_padding_fraction: float = 0.25
    # Recognition model pack. "buffalo_l" (ResNet50, w600k) loads reliably via
    # InsightFace auto-download and is the default. "antelopev2" (ResNet100) is
    # more accurate in theory but fails to load with this InsightFace/onnxruntime
    # build ("assert 'detection'"), so it is not used unless that's resolved.
    face_model_name: str = "buffalo_l"
    # Detector confidence. 0.5 (InsightFace default) gives clean, well-aligned
    # crops => best match scores. Lower it (e.g. 0.3) only if small/printed
    # document faces are being missed entirely.
    face_det_thresh: float = 0.5
    # Fallback confidence used only when the primary pass finds no face — recovers
    # small/printed document faces (Aadhaar/PAN) without hurting normal selfies.
    face_low_det_thresh: float = 0.3
    # Retry detection on an upscaled copy when no face is found (helps tiny doc faces).
    face_enable_upscale_retry: bool = True
    face_upscale_factor: float = 2.0
    # -- throughput ----------------------------------------------------------
    # Detector/recognizer pairing. "balanced" (default) keeps buffalo_l's
    # ResNet50 recognition — the model that decides match scores — behind
    # buffalo_s's much cheaper SCRFD-500M detector, roughly doubling throughput.
    # "accurate" is stock buffalo_l. "fast" also swaps recognition for
    # MobileFaceNet (~9x throughput) but moves the embedding space, so its
    # threshold must be recalibrated first — see scripts/compare_tiers.py.
    face_tier: str = "balanced"
    # ONNX Runtime threads *inside* one inference. 1 means each request is served
    # by one core and parallelism comes from serving many requests at once, which
    # is what scales: at 100 in-flight requests, per-inference fan-out
    # oversubscribes the CPU and collapses throughput. Raise only for low-
    # concurrency deployments where single-request latency is the goal.
    face_intra_op_threads: int = 1
    face_inter_op_threads: int = 1
    # Downscale inputs whose longer side exceeds this before detection (0 = off).
    # The detector letterboxes to face_detection_* anyway, so a 12 MP capture only
    # makes that resize costlier; the recognition crop is unaffected.
    face_max_input_side: int = 1600
    # Memoize detector output by image content. Only pays off when one request
    # detects the same pixels twice (it did while the liveness gate ran first).
    face_enable_det_cache: bool = False
    # Cap concurrent inferences. Past the core count, extra in-flight work adds
    # queueing latency but no throughput, so requests beyond this wait briefly and
    # are then shed with 503 rather than piling up past the client's timeout.
    # 0 = derive from the CPU count.
    max_concurrent_inferences: int = 0
    # How long a request waits for an inference slot before being shed (503).
    inference_queue_timeout_seconds: float = 20.0
    # Run throwaway inferences at startup so the first real request does not pay
    # the cold-session cost. Matters most under autoscaling, where a new instance
    # goes straight into the burst it was scaled out for. /readyz stays 503 until
    # this finishes, so the load balancer holds traffic back until then.
    warmup_on_startup: bool = True
    warmup_iterations: int = 2

    # Average each face embedding with its horizontal-mirror embedding (flip TTA).
    # Standard ArcFace trick; slightly improves robustness on low-quality inputs,
    # but doubles the recognition pass per embed. OFF by default for throughput:
    # it ~halves embed latency (measured 278ms->106ms for two embeds) for only a
    # ~0.02-0.03 genuine-score drop, well clear of the 0.5 threshold. Set True to
    # trade speed back for a little robustness on blurry document photos.
    face_enable_flip_tta: bool = False


@lru_cache
def get_ml_settings() -> MLServiceSettings:
    return MLServiceSettings()


async def _warm_up(service, iterations: int) -> None:
    """Run throwaway inferences so the first real request is not the slow one.

    A cold ONNX session pays one-off costs on its first run — memory arena
    allocation, kernel selection, first touch of the weights. Under autoscaling
    that lands on real traffic: the instance reports healthy, receives the burst
    it was scaled out for, and answers the first requests several times slower
    than it will answer the rest. Paying it here, before ``/readyz`` passes,
    moves that cost off the request path entirely.

    Uses synthetic noise rather than a face image: the point is to execute the
    graph, and detection finding nothing still runs the full detector.
    """
    import numpy as np

    def _run() -> None:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (640, 480, 3), dtype=np.uint8)
        for _ in range(max(1, iterations)):
            service.match_pair(frame, frame)

    started = time.perf_counter()
    try:
        await anyio.to_thread.run_sync(_run)
        logger.info("Warm-up complete in %.2fs", time.perf_counter() - started)
    except Exception:
        # A warm-up failure says nothing about whether real traffic will work —
        # the models loaded. Log it and let readiness proceed.
        logger.warning("Warm-up failed; serving anyway", exc_info=True)


# ---------------------------------------------------------------------------
# Lifespan — instantiate every model service once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_ml_settings()
    device = s.ml_device

    # Admission control first, so the app can shed load even if a model fails.
    app.state.inference_limiter = InferenceLimiter.from_settings(
        s.max_concurrent_inferences, s.inference_queue_timeout_seconds
    )
    # Starlette runs sync routes on a 40-slot thread pool by default. Inference
    # is already capped by the limiter above; the pool just needs enough threads
    # to service that cap plus the non-inference routes, or requests would queue
    # twice — once for a thread, then again for a slot.
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = max(40, app.state.inference_limiter.limit * 2 + 8)
    logger.info(
        "Concurrency: max_concurrent_inferences=%d queue_timeout=%.1fs thread_pool=%d",
        app.state.inference_limiter.limit,
        s.inference_queue_timeout_seconds,
        int(limiter.total_tokens),
    )

    logger.info("Loading ML models on device=%s …", device)
    failed: list[str] = []

    app.state.nsfw_service = None
    app.state.spoof_service = None
    app.state.blur_service = None

    if not s.face_match_only:
        try:
            app.state.nsfw_service = NSFWDetectionService(
                huggingface_repo_id=s.nsfw_hf_repo_id,
                huggingface_model_file=s.nsfw_hf_model_file,
                local_model_path=s.nsfw_local_model_path or None,
                model_source=s.nsfw_model_source,
                nsfw_threshold=s.nsfw_threshold,
                device=device,
            )
            app.state.nsfw_service.load_model()
        except Exception:
            logger.exception("Failed to load NSFWDetectionService")
            app.state.nsfw_service = None
            failed.append("nsfw")

    # Face embedding/detector first — the ONNX spoof engine reuses its detector
    # to crop the face for MiniFASNet.
    try:
        app.state.face_embedding_service = FaceEmbeddingService(
            embedding_dim=s.face_embedding_dim,
            detection_size=(s.face_detection_width, s.face_detection_height),
            facial_area_threshold=s.face_area_threshold,
            device=device,
            enable_padding_retry=s.face_enable_padding_retry,
            padding_fraction=s.face_padding_fraction,
            model_name=s.face_model_name,
            det_thresh=s.face_det_thresh,
            low_det_thresh=s.face_low_det_thresh,
            enable_upscale_retry=s.face_enable_upscale_retry,
            upscale_factor=s.face_upscale_factor,
            enable_flip_tta=s.face_enable_flip_tta,
            tier=s.face_tier,
            intra_op_threads=s.face_intra_op_threads,
            inter_op_threads=s.face_inter_op_threads,
            max_input_side=s.face_max_input_side,
            enable_det_cache=s.face_enable_det_cache,
        )
        app.state.face_embedding_service.load_model()
    except Exception:
        logger.exception("Failed to load FaceEmbeddingService")
        app.state.face_embedding_service = None
        failed.append("face_embedding")

    def _build_mobilenet_spoof(screen):
        svc = SpoofDetectionService(
            huggingface_repo_id=s.spoof_hf_repo_id,
            huggingface_model_file=s.spoof_hf_model_file,
            local_model_path=s.spoof_local_model_path or None,
            model_source=s.spoof_model_source,
            spoof_threshold=s.spoof_threshold,
            screen_replay_detector=screen,
            device=device,
        )
        svc.load_model()
        logger.info("Spoof engine: legacy MobileNetV3")
        return svc

    if not s.face_match_only:
        try:
            screen_detector = (
                ScreenReplayDetector(threshold=s.spoof_screen_threshold)
                if s.spoof_screen_detection_enabled
                else None
            )
            if s.spoof_engine == "onnx":
                # Resolve each model path: fall back to ./models when the configured
                # (container) path is absent, so native launches work without extra env.
                def _resolve(path: str) -> str | None:
                    if Path(path).exists():
                        return path
                    cwd_path = Path.cwd() / "models" / Path(path).name
                    if cwd_path.exists():
                        logger.warning("ONNX model not at %s; using %s", path, cwd_path)
                        return str(cwd_path)
                    return None

                # (path, crop_scale): V2 at 2.7x, V1SE at 4.0x (canonical Silent-Face).
                candidates = [
                    (_resolve(s.spoof_onnx_model_path), 2.7),
                    (_resolve(s.spoof_onnx_model_path_2), 4.0),
                ]
                models = [(p, scale) for p, scale in candidates if p is not None]
                face_svc = app.state.face_embedding_service
                try:
                    if face_svc is None:
                        raise RuntimeError("face detector unavailable")
                    if not models:
                        raise FileNotFoundError("no MiniFASNet ONNX models found")
                    app.state.spoof_service = OnnxSpoofDetectionService(
                        models=models,
                        face_detector=face_svc.detect_box,
                        live_threshold=s.spoof_onnx_live_threshold,
                        screen_replay_detector=screen_detector,
                    )
                    app.state.spoof_service.load_model()
                    logger.info("Spoof engine: MiniFASNet ONNX ensemble (%d model(s))", len(models))
                except Exception:
                    logger.exception("ONNX spoof engine failed; falling back to MobileNetV3")
                    app.state.spoof_service = _build_mobilenet_spoof(screen_detector)
            else:
                app.state.spoof_service = _build_mobilenet_spoof(screen_detector)
        except Exception:
            logger.exception("Failed to load spoof service")
            app.state.spoof_service = None
            failed.append("spoof")

        try:
            app.state.blur_service = BlurDetectionService(
                huggingface_repo_id=s.blur_hf_repo_id,
                huggingface_model_file=s.blur_hf_model_file,
                local_model_path=s.blur_local_model_path or None,
                model_source=s.blur_model_source,
                blur_threshold=s.blur_threshold,
                device=device,
            )
            app.state.blur_service.load_model()
        except Exception:
            logger.exception("Failed to load BlurDetectionService")
            app.state.blur_service = None
            failed.append("blur")

    nsfw = app.state.nsfw_service
    spoof = app.state.spoof_service
    blur = app.state.blur_service
    if nsfw is not None and spoof is not None and blur is not None:
        app.state.iqa_service = ImageQualityAssessmentService(
            nsfw_service=nsfw,
            spoof_service=spoof,
            blur_service=blur,
        )
    elif s.face_match_only:
        # Face-match-only deliberately does not load the IQA models; this is the
        # configured state, not a fault, and must not page anyone.
        app.state.iqa_service = None
        logger.info("IQA disabled: face_match_only=true (only face embedding is loaded)")
    else:
        app.state.iqa_service = None
        logger.error(
            "IQA disabled: need all three models loaded (nsfw_missing=%s spoof_missing=%s "
            "blur_missing=%s)",
            nsfw is None,
            spoof is None,
            blur is None,
        )

    if failed:
        logger.warning("ML service started with degraded models: %s", ", ".join(failed))
    else:
        logger.info("All ML models loaded successfully.")

    app.state.warmed_up = False
    if s.warmup_on_startup and app.state.face_embedding_service is not None:
        await _warm_up(app.state.face_embedding_service, s.warmup_iterations)
        app.state.warmed_up = True
    else:
        app.state.warmed_up = app.state.face_embedding_service is not None

    yield
    logger.info("ML inference service shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    from zepiris.logging_config import configure_logging
    from zepiris.ml_inference.routes import router

    configure_logging()
    application = FastAPI(
        title="ZepIris ML Inference Service",
        version=package_version,
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()


def run() -> None:
    """Entry point for ``zepiris-ml-inference-api`` console script."""
    import uvicorn

    s = get_ml_settings()
    uvicorn.run(
        "zepiris.ml_inference.app:app",
        host=s.host,
        port=s.port,
        reload=False,
    )
