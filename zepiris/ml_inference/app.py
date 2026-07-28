"""FastAPI application for ML inference microservice.

Run standalone:
    uvicorn zepiris.ml_inference.app:app --host 0.0.0.0 --port 8001

Or via the entry point:
    zepiris-ml-inference-api
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from zepiris.ml_inference.blur_detection import BlurDetectionService
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


# ---------------------------------------------------------------------------
# Lifespan — instantiate every model service once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_ml_settings()
    device = s.ml_device

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
    else:
        app.state.iqa_service = None
        logger.error(
            "IQA disabled: need all three models loaded (nsfw=%s spoof=%s blur=%s)",
            nsfw is None,
            spoof is None,
            blur is None,
        )

    if failed:
        logger.warning("ML service started with degraded models: %s", ", ".join(failed))
    else:
        logger.info("All ML models loaded successfully.")
    yield
    logger.info("ML inference service shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    from zepiris.ml_inference.routes import router

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
