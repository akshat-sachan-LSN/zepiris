"""API routes for ML inference microservice."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from zepiris.framing import FrameError, decode_pair_frame
from zepiris.ml_inference.deps import (
    BlurDep,
    FaceEmbeddingDep,
    IQADep,
    NSFWDep,
    SpoofDep,
)
from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    FaceDetectionResult,
    FaceEmbeddingResult,
    FaceMatchResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)

router = APIRouter()


class ImagePayload(BaseModel):
    """Base64-encoded image payload."""

    image_b64: str


def _decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode base64-encoded image to numpy array in RGB format.

    Args:
        image_b64: Base64-encoded image string

    Returns:
        np.ndarray: Image in RGB format, shape (H, W, 3), dtype uint8

    Raises:
        HTTPException: If decoding fails
    """
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_base64: {str(e)}") from e

    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty_image_data")

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_image_format: {str(e)}") from e

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="failed_to_decode_image")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def _decode_image_bytes(raw: bytes, *, field: str) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/…) to an RGB array.

    The binary match path hands the original upload straight through, so this is
    the only decode in the whole pipeline for that image — no base64, no
    intermediate re-encode.
    """
    if not raw:
        raise HTTPException(status_code=400, detail=f"empty_image: {field}")
    image_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail=f"failed_to_decode_image: {field}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up. Use /readyz to decide about sending traffic."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request):
    """Readiness: models loaded *and* warmed, so this instance can serve at speed.

    Distinct from ``/healthz`` because a freshly started instance accepts
    connections long before it can answer quickly — the first inference through a
    cold ONNX session pays one-off setup that a real request should not. An
    autoscaling group that routes on liveness alone sends the burst it just scaled
    out for straight into instances that are not ready for it.

    Point the load balancer's health check here.
    """
    state = request.app.state
    if getattr(state, "face_embedding_service", None) is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "face_embedding_model_unavailable"},
        )
    if not getattr(state, "warmed_up", False):
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "reason": "warming_up"}
        )
    return {"status": "ready"}


@router.get("/metrics")
def metrics(request: Request) -> dict:
    """Saturation, for autoscaling and dashboards.

    ``queue_depth`` is the signal to scale on. CPU utilization saturates near
    100% whether the service is comfortably busy or badly overloaded, and
    ``active`` is capped by the limiter for the same reason — neither separates
    the two. A non-zero queue means requests are waiting for CPU that does not
    exist on this instance yet, which is exactly when another one is needed.
    """
    limiter = request.app.state.inference_limiter
    return {
        "inference": limiter.snapshot(),
        "ready": bool(getattr(request.app.state, "warmed_up", False)),
    }


@router.post("/v1/face/match", response_model=FaceMatchResult)
async def match_faces(
    request: Request,
    service: FaceEmbeddingDep,
    want_probe_sharpness: bool = False,
) -> FaceMatchResult:
    """Score one 1:1 pair from raw image bytes — the whole hot path in one call.

    The body is both original images end to end behind a 4-byte length prefix,
    not multipart. Multipart spools any part over 1 MB to a temporary **file**,
    so every request carrying a normal phone photo wrote to disk and read it back
    — pointless I/O on a service that persists nothing, and at high request rates
    a genuine source of disk churn. A framed body stays in memory and skips
    boundary scanning entirely.

    This is on top of what the single-call design already removed per side: a
    decode, a lossless PNG re-encode (~30 ms and ~10x the bytes), base64 both
    ways, and a second HTTP round trip. Only the similarity comes back, so the
    512-float vectors never touch JSON.

    Concurrency is bounded by the service's inference limiter; callers past the
    limit wait briefly and are then shed with 503 rather than queueing past their
    own timeout.
    """
    try:
        probe_raw, reference_raw = decode_pair_frame(await request.body())
    except FrameError as exc:
        raise HTTPException(status_code=400, detail=f"malformed_frame: {exc}") from exc

    limiter = request.app.state.inference_limiter
    async with limiter.slot():
        return await run_in_threadpool(
            _match_sync,
            service,
            probe_raw,
            reference_raw,
            want_probe_sharpness,
        )


def _match_sync(
    service, probe_raw: bytes, reference_raw: bytes, want_probe_sharpness: bool
) -> FaceMatchResult:
    """Decode + embed + score, off the event loop."""
    probe_rgb = _decode_image_bytes(probe_raw, field="probe")
    reference_rgb = _decode_image_bytes(reference_raw, field="reference")
    return service.match_pair(
        probe_rgb, reference_rgb, want_probe_sharpness=want_probe_sharpness
    )


@router.post("/v1/iqa/nsfw_check", response_model=NSFWDetectionResult)
def detect_nsfw(
    service: NSFWDep,
    payload: ImagePayload,
) -> NSFWDetectionResult:
    """Run NSFW detection on an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.forward(image_rgb)


@router.post("/v1/iqa/spoof_check", response_model=SpoofDetectionResult)
def detect_spoof(
    service: SpoofDep,
    payload: ImagePayload,
) -> SpoofDetectionResult:
    """Run spoof detection on an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.forward(image_rgb)


@router.post("/v1/iqa/blur_check", response_model=BlurDetectionResult)
def detect_blur(
    service: BlurDep,
    payload: ImagePayload,
) -> BlurDetectionResult:
    """Run blur detection on an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.forward(image_rgb)


@router.post("/v1/face/embed", response_model=FaceEmbeddingResult)
def embed_face(
    service: FaceEmbeddingDep,
    payload: ImagePayload,
) -> FaceEmbeddingResult:
    """Generate face embedding from an image."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.embed(image_rgb)


@router.post("/v1/face/detect", response_model=FaceDetectionResult)
def detect_face(
    service: FaceEmbeddingDep,
    payload: ImagePayload,
) -> FaceDetectionResult:
    """Detect the primary face and return its normalized bounding box (no recognition)."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.detect_box(image_rgb)


@router.post("/v1/iqa/assess", response_model=ImageQualityAssessmentResult)
def assess_image_quality(
    service: IQADep,
    payload: ImagePayload,
) -> ImageQualityAssessmentResult:
    """Run combined image quality assessment (NSFW + spoof + blur in parallel)."""
    image_rgb = _decode_base64_image(payload.image_b64)
    return service.assess(image_rgb)
