from __future__ import annotations

import base64
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from zepiris.deps import EmbeddingDep, IQADep, S3FetcherDep, SettingsDep
from zepiris.exceptions import (
    EmptyUploadError,
    ImageEncodeError,
    ImageTooLargeError,
    ReferenceFaceNotFoundError,
    ReferenceImageDecodeError,
)
from zepiris.schemas.face import (
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_SIZE_MB,
    VerificationResult,
    VerifyResponse,
)
from zepiris.services.similarity import cosine

router = APIRouter()


def _validate_image_bytes(raw: bytes) -> None:
    if not raw:
        raise EmptyUploadError()
    if len(raw) > MAX_IMAGE_SIZE_BYTES:
        mb = len(raw) / (1024 * 1024)
        raise ImageTooLargeError(mb=mb, max_mb=MAX_IMAGE_SIZE_MB)


def _decode_rgb(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _to_bgr_b64(image_rgb: np.ndarray) -> str:
    # The ML service decodes JPEG as BGR then converts BGR->RGB, so hand it BGR.
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ImageEncodeError()
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@router.post("/verify")
async def verify_face(
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    fetcher: S3FetcherDep,
    s3_url: str = Form(...),
    file: UploadFile = File(...),
    threshold: float | None = Form(None),
) -> dict:
    """Stateless 1:1 verification: live photo (upload) vs reference image (S3 URL)."""
    request_id = str(uuid.uuid4())
    raw = await file.read()
    _validate_image_bytes(raw)

    decision_threshold = threshold if threshold is not None else settings.verify_threshold

    live_rgb = _decode_rgb(raw)
    if live_rgb is None:
        return {
            "requestId": request_id,
            "decodeFailed": True,
            "faceDetected": False,
            "iqaPassed": False,
            "verificationResult": {
                "isMatch": False,
                "score": None,
                "threshold": decision_threshold,
            },
        }

    ml_struct = iqa_svc.assess(live_rgb, _to_bgr_b64(live_rgb))

    if not ml_struct.spoof.is_live:
        return {
            "requestId": request_id,
            "imageQualityAssessment": ml_struct.model_dump(),
            "iqaPassed": False,
            "livenessFailed": True,
            "faceDetected": False,
            "verificationResult": {
                "isMatch": False,
                "score": None,
                "threshold": decision_threshold,
            },
        }

    if not ml_struct.passed:
        return {
            "requestId": request_id,
            "imageQualityAssessment": ml_struct.model_dump(),
            "iqaPassed": False,
            "faceDetected": False,
            "verificationResult": {
                "isMatch": False,
                "score": None,
                "threshold": decision_threshold,
            },
        }

    live_embed = embedding_svc.embed(live_rgb)
    if not live_embed.face_detected:
        return {
            "requestId": request_id,
            "imageQualityAssessment": ml_struct.model_dump(),
            "iqaPassed": True,
            "faceDetected": False,
            "verificationResult": {
                "isMatch": False,
                "score": None,
                "threshold": decision_threshold,
            },
        }

    # Fetch + embed the reference image from S3. Fetch/decode/no-face raise 400s.
    ref_raw = fetcher.fetch(s3_url)
    ref_rgb = _decode_rgb(ref_raw)
    if ref_rgb is None:
        raise ReferenceImageDecodeError()
    ref_embed = embedding_svc.embed(ref_rgb)
    if not ref_embed.face_detected:
        raise ReferenceFaceNotFoundError()

    score = cosine(live_embed.embedding, ref_embed.embedding)
    is_match = score >= decision_threshold

    return VerifyResponse(
        request_id=request_id,
        image_quality_assessment=ml_struct,
        verification_result=VerificationResult(
            is_match=bool(is_match), score=score, threshold=decision_threshold
        ),
        face_detected=True,
        iqa_passed=True,
    ).model_dump(by_alias=True)
