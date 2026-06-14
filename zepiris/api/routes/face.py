from __future__ import annotations

import base64
import binascii
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Form

from zepiris.deps import EmbeddingDep, IQADep, LearnerDep, S3FetcherDep, SettingsDep
from zepiris.exceptions import (
    DocumentTooBlurryError,
    EmptyUploadError,
    FeedbackValidationError,
    ImageEncodeError,
    ImageSourceError,
    ImageTooLargeError,
    ReferenceFaceNotFoundError,
    ReferenceImageDecodeError,
)
from zepiris.schemas.face import (
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_SIZE_MB,
    DocMatchRequest,
    FaceMatchRequest,
    VerificationResult,
    VerificationScores,
    VerifyResponse,
)
from zepiris.services.learning import FACE_KIND, GENERIC_DOC_TYPE
from zepiris.services.similarity import cosine

router = APIRouter()


def _resolve_threshold(
    explicit: float | None, learner, doc_type: str, default: float
) -> tuple[float, str]:
    """Resolve the decision threshold and report where it came from.

    Precedence: explicit per-request value > learned (calibrated from feedback
    for this doc_type) > configured default. The source string surfaces in the
    response so callers can see the adaptive learning being applied.
    """
    if explicit is not None:
        return explicit, "explicit"
    learned = learner.learned_threshold(doc_type)
    if learned is not None:
        return learned, "learned"
    return default, "default"


def _scores_dict(match_score: float | None, threshold: float, ml_struct) -> dict:
    """Flat numeric summary gathered from the match score + the IQA result.

    Quality scores are ``None`` when the liveness/IQA gate did not run
    (``ml_struct is None``, e.g. docmatch).
    """
    return {
        "matchScore": match_score,
        "threshold": threshold,
        "margin": (match_score - threshold) if match_score is not None else None,
        "livenessScore": ml_struct.spoof.probability if ml_struct else None,
        "blurScore": ml_struct.blur.probability if ml_struct else None,
        "nsfwSafeScore": ml_struct.nsfw.probability if ml_struct else None,
    }


def _record_outcome(learner, *, request_id: str, doc_type: str, body: dict) -> None:
    """Log a scored verification so operator feedback can calibrate thresholds.

    Only scores and the decision are logged — never images or personal data.
    Unscored results (decode/liveness/IQA failures) carry no signal for
    threshold fitting and are skipped.
    """
    result = body.get("verificationResult") or {}
    score = result.get("score")
    if score is None:
        return
    scores = body.get("scores") or {}
    learner.record_sample(
        request_id=request_id,
        doc_type=doc_type,
        score=float(score),
        threshold=float(result["threshold"]),
        is_match=bool(result["isMatch"]),
        liveness_score=scores.get("livenessScore"),
        blur_score=scores.get("blurScore"),
        nsfw_safe_score=scores.get("nsfwSafeScore"),
    )


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


def _decode_b64_image(value: str, *, field: str) -> bytes:
    """Decode an inline base64 image payload to raw, size-validated bytes.

    Accepts both a bare base64 string and a ``data:`` URI
    (e.g. ``data:image/jpeg;base64,<payload>``).
    """
    payload = value.strip()
    if payload.startswith("data:"):
        # Strip the "data:<mime>;base64," prefix, keep the payload.
        payload = payload.partition(",")[2]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageSourceError(field=field, reason="invalid_base64") from exc
    _validate_image_bytes(raw)
    return raw


def _resolve_image_source(
    *, b64: str | None, s3_url: str | None, fetcher, field: str
) -> bytes:
    """Resolve one image side (selfie or reference) to raw bytes.

    Each side is supplied via exactly one of two inputs — inline base64
    (``<field>_b64``) or an S3 URL (``<field>_s3``). Supplying both is
    ambiguous and rejected (400); supplying neither is missing (400).

    The selfie is liveness/quality-checked downstream regardless of which
    input it arrived through; the reference is only ever embedded.
    """
    has_b64 = bool(b64 and b64.strip())
    has_s3 = bool(s3_url and s3_url.strip())
    if has_b64 and has_s3:
        raise ImageSourceError(field=field, reason="ambiguous")
    if has_b64:
        return _decode_b64_image(b64, field=field)
    if has_s3:
        return fetcher.fetch(s3_url.strip())
    raise ImageSourceError(field=field, reason="missing")


# Margin added around the detected document face before cropping; the extra
# context lets the detector re-localize keypoints precisely on the crop.
DOC_FACE_CROP_MARGIN = 0.35
# Upscale tiny ID-card photos so the crop's shorter side reaches this many
# pixels — keypoint alignment (and thus the embedding) is far better at size.
DOC_FACE_MIN_SIDE = 320
# If the detected face already fills the frame, cropping buys nothing.
DOC_FACE_SKIP_AREA = 0.85


def _sharpness(image_rgb: np.ndarray) -> float:
    """Variance of the Laplacian — a cheap focus/blur metric (higher = sharper).

    Computed on the extracted face at its native resolution (before any
    upscaling, which would inflate the number). A crisp ID photo typically
    scores > 100; blurry phone captures of a card fall in the single/low-double
    digits.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _extract_doc_face(doc_rgb: np.ndarray, embedding_svc) -> tuple[np.ndarray | None, dict]:
    """Split the face photo out of a document image (Aadhaar/PAN/licence scan).

    Detects the primary face on the document, crops it with margin, and
    upscales small crops so the recognition model sees a proper-sized face
    instead of a tiny photo lost in the card layout.

    Returns ``(crop, diag)``. ``crop`` is ``None`` when no usable face region is
    found (caller falls back to embedding the full doc). ``diag`` always carries
    what was observed — detection score, area fraction, and the crop's native
    sharpness — so the response can explain a weak match.
    """
    diag: dict = {"faceDetected": False, "detScore": None, "sharpness": None}
    det = embedding_svc.detect_box(doc_rgb)
    diag["faceDetected"] = bool(det.face_detected)
    if not det.face_detected:
        return None, diag
    diag["detScore"] = round(float(det.score), 4)

    x1, y1, x2, y2 = det.bbox
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return None, diag
    if (x2 - x1) * (y2 - y1) >= DOC_FACE_SKIP_AREA:
        return None, diag  # already face-dominant; the normal path handles it best

    h, w = doc_rgb.shape[:2]
    mx = (x2 - x1) * w * DOC_FACE_CROP_MARGIN
    my = (y2 - y1) * h * DOC_FACE_CROP_MARGIN
    px1 = max(0, int(round(x1 * w - mx)))
    py1 = max(0, int(round(y1 * h - my)))
    px2 = min(w, int(round(x2 * w + mx)))
    py2 = min(h, int(round(y2 * h + my)))
    crop = doc_rgb[py1:py2, px1:px2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None, diag

    # Sharpness is measured on the native-resolution crop, before upscaling.
    diag["sharpness"] = round(_sharpness(crop), 2)

    ch, cw = crop.shape[:2]
    if min(ch, cw) < DOC_FACE_MIN_SIDE:
        scale = DOC_FACE_MIN_SIDE / min(ch, cw)
        crop = cv2.resize(
            crop,
            (int(round(cw * scale)), int(round(ch * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop, diag


def _embed_probe(probe_rgb: np.ndarray, embedding_svc, *, is_document: bool):
    """Embed the incoming probe image, returning ``(embedding, doc_diag)``.

    For a document, the face photo is first split out (crop + upscale) and
    embedded on its own, falling back to the full image when extraction fails.
    ``doc_diag`` is ``None`` for a plain face probe, else the extraction
    diagnostics (detection score, sharpness, whether the crop was used).
    """
    if not is_document:
        return embedding_svc.embed(probe_rgb), None

    face_crop, diag = _extract_doc_face(probe_rgb, embedding_svc)
    diag["usedCrop"] = False
    if face_crop is not None:
        crop_embed = embedding_svc.embed(face_crop)
        if crop_embed.face_detected:
            diag["usedCrop"] = True
            return crop_embed, diag
    return embedding_svc.embed(probe_rgb), diag


def _run_verify(
    *,
    request_id: str,
    probe_raw: bytes,
    reference_raw: bytes,
    iqa_svc,
    embedding_svc,
    decision_threshold: float,
    threshold_source: str,
    run_liveness: bool,
    probe_is_document: bool = False,
    min_sharpness: float = 0.0,
) -> dict:
    """Core 1:1 verification: embed the incoming probe and the stored reference
    selfie, then compare with cosine similarity.

    ``probe_raw`` is the image submitted *now* to be verified — a live face
    (facematch) or an ID document (docmatch). ``reference_raw`` is the user's
    enrolled selfie (the DB source of truth); it is only ever embedded.

    With ``run_liveness`` the probe is gated on liveness + image quality before
    embedding (facematch: the probe is a live capture). docmatch passes it
    ``False`` because the probe is a printed document, not a live face. With
    ``probe_is_document`` the face photo is split out of the probe before
    embedding, and its sharpness is reported in ``documentFace``; when
    ``min_sharpness`` > 0 a too-blurry extracted face is rejected.
    """
    def _verification_result(is_match: bool, score: float | None) -> dict:
        return {
            "isMatch": is_match,
            "score": score,
            "threshold": decision_threshold,
            "thresholdSource": threshold_source,
        }

    probe_rgb = _decode_rgb(probe_raw)
    if probe_rgb is None:
        return {
            "requestId": request_id,
            "decodeFailed": True,
            "faceDetected": False,
            "iqaPassed": False,
            "verificationResult": _verification_result(False, None),
            "scores": _scores_dict(None, decision_threshold, None),
        }

    ml_struct = None
    if run_liveness:
        ml_struct = iqa_svc.assess(probe_rgb, _to_bgr_b64(probe_rgb))

        if not ml_struct.spoof.is_live:
            return {
                "requestId": request_id,
                "imageQualityAssessment": ml_struct.model_dump(),
                "iqaPassed": False,
                "livenessFailed": True,
                "faceDetected": False,
                "verificationResult": _verification_result(False, None),
                "scores": _scores_dict(None, decision_threshold, ml_struct),
            }

        if not ml_struct.passed:
            return {
                "requestId": request_id,
                "imageQualityAssessment": ml_struct.model_dump(),
                "iqaPassed": False,
                "faceDetected": False,
                "verificationResult": _verification_result(False, None),
                "scores": _scores_dict(None, decision_threshold, ml_struct),
            }

    probe_embed, doc_diag = _embed_probe(
        probe_rgb, embedding_svc, is_document=probe_is_document
    )

    # Flag (and optionally reject) a too-blurry document face: a low-sharpness
    # crop embeds poorly and silently drags the match score down.
    if doc_diag is not None:
        sharp = doc_diag.get("sharpness")
        low = (
            min_sharpness > 0 and doc_diag.get("usedCrop") and sharp is not None
            and sharp < min_sharpness
        )
        doc_diag["lowQuality"] = bool(low)
        if low:
            raise DocumentTooBlurryError(sharpness=sharp, min_sharpness=min_sharpness)

    if not probe_embed.face_detected:
        return {
            "requestId": request_id,
            "imageQualityAssessment": ml_struct.model_dump() if ml_struct else None,
            "iqaPassed": True,
            "faceDetected": False,
            "verificationResult": _verification_result(False, None),
            "scores": _scores_dict(None, decision_threshold, ml_struct),
            "documentFace": doc_diag,
        }

    reference_rgb = _decode_rgb(reference_raw)
    if reference_rgb is None:
        raise ReferenceImageDecodeError()
    reference_embed = embedding_svc.embed(reference_rgb)
    if not reference_embed.face_detected:
        raise ReferenceFaceNotFoundError()

    score = cosine(probe_embed.embedding, reference_embed.embedding)
    is_match = score >= decision_threshold

    return VerifyResponse(
        request_id=request_id,
        image_quality_assessment=ml_struct,
        verification_result=VerificationResult(
            is_match=bool(is_match),
            score=score,
            threshold=decision_threshold,
            threshold_source=threshold_source,
        ),
        scores=VerificationScores(**_scores_dict(score, decision_threshold, ml_struct)),
        document_face=doc_diag,
        face_detected=True,
        iqa_passed=True,
    ).model_dump(by_alias=True)


@router.post("/facematch/verify")
async def facematch_verify(
    req: FaceMatchRequest,
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Face-to-face 1:1 verification: an incoming live face vs the enrolled selfie.

    JSON body. Incoming face being verified (``face_check``; the live capture,
    gated on liveness/quality): supply exactly one of ``face_check_b64`` (base64
    binary) or ``face_check_s3`` (S3 URL). Enrolled reference selfie / source of
    truth (``source_selfie``; only embedded, from the DB): supply exactly one of
    ``source_selfie_b64`` or ``source_selfie_s3``. Sending both inputs for a
    side is rejected.
    """
    request_id = str(uuid.uuid4())
    probe_raw = _resolve_image_source(
        b64=req.face_check_b64, s3_url=req.face_check_s3, fetcher=fetcher, field="face_check"
    )
    reference_raw = _resolve_image_source(
        b64=req.source_selfie_b64, s3_url=req.source_selfie_s3, fetcher=fetcher,
        field="source_selfie",
    )
    decision_threshold, threshold_source = _resolve_threshold(
        req.threshold, learner, FACE_KIND, settings.verify_threshold
    )
    body = _run_verify(
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        iqa_svc=iqa_svc,
        embedding_svc=embedding_svc,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
        run_liveness=True,
    )
    _record_outcome(learner, request_id=request_id, doc_type=FACE_KIND, body=body)
    return body


@router.post("/docmatch/verify")
async def docmatch_verify(
    req: DocMatchRequest,
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Doc-to-face 1:1 verification: the face on an uploaded ID document vs the
    enrolled selfie.

    JSON body. Incoming document being verified (``doc_check``; face
    auto-extracted, then embedded — no liveness gate, since a printed ID is not
    a live capture): supply exactly one of ``doc_check_b64`` (base64 binary) or
    ``doc_check_s3`` (S3 URL). Enrolled reference selfie / source of truth
    (``source_selfie``; only embedded, from the DB): supply exactly one of
    ``source_selfie_b64`` or ``source_selfie_s3``. Defaults to a more lenient
    threshold than face-match because printed ID photos embed weaker.

    ``doc_type`` (e.g. ``aadhaar``, ``pan``) buckets the request for adaptive
    threshold learning: each document type's threshold is calibrated separately
    from operator feedback, since Aadhaar and PAN photos score differently.
    """
    request_id = str(uuid.uuid4())
    probe_raw = _resolve_image_source(
        b64=req.doc_check_b64, s3_url=req.doc_check_s3, fetcher=fetcher, field="doc_check"
    )
    reference_raw = _resolve_image_source(
        b64=req.source_selfie_b64, s3_url=req.source_selfie_s3, fetcher=fetcher,
        field="source_selfie",
    )
    doc_kind = (req.doc_type or GENERIC_DOC_TYPE).strip().lower() or GENERIC_DOC_TYPE
    decision_threshold, threshold_source = _resolve_threshold(
        req.threshold, learner, doc_kind, settings.doc_verify_threshold
    )
    body = _run_verify(
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        iqa_svc=iqa_svc,
        embedding_svc=embedding_svc,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
        run_liveness=False,
        probe_is_document=True,
        min_sharpness=settings.doc_min_sharpness,
    )
    _record_outcome(learner, request_id=request_id, doc_type=doc_kind, body=body)
    return body


@router.post("/feedback")
async def verification_feedback(
    learner: LearnerDep,
    request_id: str | None = Form(None),
    genuine: bool | None = Form(None),
) -> dict:
    """Report the confirmed outcome of a past verification.

    ``genuine=true`` means the selfie and the document/reference really were the
    same person (per downstream KYC/manual review); ``genuine=false`` means an
    impostor. Feedback is joined with the logged match score and used to re-fit
    the decision threshold for that request's document type — this is how the
    system keeps improving on real Aadhaar/PAN traffic while it runs.
    """
    if not request_id or genuine is None:
        raise FeedbackValidationError("provide request_id and genuine")
    summary = learner.record_feedback(request_id=request_id, genuine=genuine)
    return {"requestId": request_id, **summary}


# Backward-compatible alias for the original single endpoint (= face match).
@router.post("/verify")
async def verify_face(
    req: FaceMatchRequest,
    settings: SettingsDep,
    iqa_svc: IQADep,
    embedding_svc: EmbeddingDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Deprecated alias of ``/facematch/verify`` (kept for existing callers)."""
    return await facematch_verify(
        req=req,
        settings=settings,
        iqa_svc=iqa_svc,
        embedding_svc=embedding_svc,
        fetcher=fetcher,
        learner=learner,
    )
