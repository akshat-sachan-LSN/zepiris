"""API routes for ML inference microservice."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


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
