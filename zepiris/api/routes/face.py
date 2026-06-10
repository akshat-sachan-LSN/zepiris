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
    ReferenceImageFetchError,
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


async def _resolve_image_bytes(
    fetcher,
    *,
    upload: UploadFile | None,
    s3_url: str | None,
    what: str,
) -> bytes:
    """Return image bytes from an uploaded file or an S3/HTTP URL.

    Exactly one source must be provided; the uploaded file takes precedence.
    ``what`` names the field(s) for the error message.
    """
    if upload is not None:
        raw = await upload.read()
        if not raw:
            raise ReferenceImageFetchError(reason="empty_upload", detail_msg=f"{what} is empty")
        return raw
    if s3_url:
        return fetcher.fetch(s3_url)
    raise ReferenceImageFetchError(reason="missing", detail_msg=f"provide {what}")


def _run_verify(
    *,
    request_id: str,
    selfie_raw: bytes,
    ref_raw: bytes,
    iqa_svc,
    embedding_svc,
    decision_threshold: float,
    run_liveness: bool,
) -> dict:
    """Core 1:1 verification: optionally gate the selfie on liveness/quality, then
    embed both images and compare with cosine similarity.

    ``run_liveness`` is True when the selfie is a live capture (uploaded ``file``).
    For pure image-to-image matches (selfie supplied as an S3 URL), it is False:
    liveness/quality gating is skipped and we just compare the two faces.
    The reference/document side is always only embedded (never liveness-checked).
    """
    live_rgb = _decode_rgb(selfie_raw)
    if live_rgb is None:
        return {
            "requestId": request_id,
            "decodeFailed": True,
            "faceDetected": False,
            "iqaPassed": False,
            "verificationResult": {"isMatch": False, "score": None, "threshold": decision_threshold},
        }

    ml_struct = None
    if run_liveness:
        ml_struct = iqa_svc.assess(live_rgb, _to_bgr_b64(live_rgb))

        if not ml_struct.spoof.is_live:
            return {
                "requestId": request_id,
                "imageQualityAssessment": ml_struct.model_dump(),
                "iqaPassed": False,
                "livenessFailed": True,
                "faceDetected": False,
                "verificationResult": {"isMatch": False, "score": None, "threshold": decision_threshold},
            }

        if not ml_struct.passed:
            return {
                "requestId": request_id,
                "imageQualityAssessment": ml_struct.model_dump(),
                "iqaPassed": False,
                "faceDetected": False,
                "verificationResult": {"isMatch": False, "score": None, "threshold": decision_threshold},
            }

    live_embed = embedding_svc.embed(live_rgb)
    if not live_embed.face_detected:
        return {
            "requestId": request_id,
            "imageQualityAssessment": ml_struct.model_dump() if ml_struct else None,
            "iqaPassed": run_liveness,
            "faceDetected": False,
            "verificationResult": {"isMatch": False, "score": None, "threshold": decision_threshold},
        }

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
        iqa_passed=run_liveness,
    ).model_dump(by_alias=True)


@router.post("/facematch/verify")
async def facematch_verify(
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    fetcher: S3FetcherDep,
    file: UploadFile | None = File(None),
    selfie_s3_url: str | None = Form(None),
    s3_url: str | None = Form(None),
    reference_file: UploadFile | None = File(None),
    threshold: float | None = Form(None),
) -> dict:
    """Face-to-face 1:1 verification: a selfie vs a reference face image.

    Selfie (exactly one): ``file`` (live upload — runs liveness/quality) or
    ``selfie_s3_url`` (stored image — pure image-to-image match, no liveness).
    Reference (exactly one): ``s3_url`` or ``reference_file`` (a face photo).
    """
    request_id = str(uuid.uuid4())
    if file is not None:
        selfie_raw = await file.read()
        _validate_image_bytes(selfie_raw)
        run_liveness = True
    else:
        selfie_raw = await _resolve_image_bytes(
            fetcher, upload=None, s3_url=selfie_s3_url, what="file or selfie_s3_url"
        )
        run_liveness = False

    ref_raw = await _resolve_image_bytes(
        fetcher, upload=reference_file, s3_url=s3_url, what="s3_url or reference_file"
    )
    decision_threshold = threshold if threshold is not None else settings.verify_threshold
    return _run_verify(
        request_id=request_id,
        selfie_raw=selfie_raw,
        ref_raw=ref_raw,
        iqa_svc=iqa_svc,
        embedding_svc=embedding_svc,
        decision_threshold=decision_threshold,
        run_liveness=run_liveness,
    )


@router.post("/docmatch/verify")
async def docmatch_verify(
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    fetcher: S3FetcherDep,
    file: UploadFile | None = File(None),
    selfie_s3_url: str | None = Form(None),
    document_file: UploadFile | None = File(None),
    s3_url: str | None = Form(None),
    threshold: float | None = Form(None),
) -> dict:
    """Doc-to-face 1:1 verification: a selfie vs the photo on an ID document.

    Selfie (exactly one): ``file`` (live upload — runs liveness/quality) or
    ``selfie_s3_url`` (stored image — pure image-to-image match, no liveness).
    Document (exactly one): ``document_file`` (uploaded Aadhaar/PAN) or ``s3_url``
    (document image URL). The face is auto-extracted from the document. Defaults
    to a more lenient threshold than face-match because printed ID photos embed weaker.
    """
    request_id = str(uuid.uuid4())
    if file is not None:
        selfie_raw = await file.read()
        _validate_image_bytes(selfie_raw)
        run_liveness = True
    else:
        selfie_raw = await _resolve_image_bytes(
            fetcher, upload=None, s3_url=selfie_s3_url, what="file or selfie_s3_url"
        )
        run_liveness = False

    ref_raw = await _resolve_image_bytes(
        fetcher, upload=document_file, s3_url=s3_url, what="s3_url or document_file"
    )
    decision_threshold = threshold if threshold is not None else settings.doc_verify_threshold
    return _run_verify(
        request_id=request_id,
        selfie_raw=selfie_raw,
        ref_raw=ref_raw,
        iqa_svc=iqa_svc,
        embedding_svc=embedding_svc,
        decision_threshold=decision_threshold,
        run_liveness=run_liveness,
    )


# Backward-compatible alias for the original single endpoint (= face match).
@router.post("/verify")
async def verify_face(
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    fetcher: S3FetcherDep,
    file: UploadFile | None = File(None),
    selfie_s3_url: str | None = Form(None),
    s3_url: str | None = Form(None),
    reference_file: UploadFile | None = File(None),
    threshold: float | None = Form(None),
) -> dict:
    """Deprecated alias of ``/facematch/verify`` (kept for existing callers)."""
    return await facematch_verify(
        settings=settings,
        iqa_svc=iqa_svc,
        embedding_svc=embedding_svc,
        fetcher=fetcher,
        file=file,
        selfie_s3_url=selfie_s3_url,
        s3_url=s3_url,
        reference_file=reference_file,
        threshold=threshold,
    )
