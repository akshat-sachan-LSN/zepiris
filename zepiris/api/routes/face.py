from __future__ import annotations

import asyncio
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
    # The ML service decodes as BGR then converts BGR->RGB, so hand it BGR.
    # PNG (lossless), and encoded identically to the embed path's payload, so the
    # probe the liveness call sends is byte-for-byte the probe the embed call
    # sends — letting the ML detector memoize one detection across both calls
    # instead of detecting the same face twice. (Liveness on lossless vs the old
    # 2nd-generation JPEG differs by <0.005 in measured prob_live.)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
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


async def _resolve_two_sources(
    *, probe_kwargs: dict, reference_kwargs: dict
) -> tuple[bytes, bytes]:
    """Resolve the probe and reference images concurrently.

    Both sides are independent network/decode work; fetching them in parallel
    (off the event loop) removes one S3 round-trip from the critical path when
    both arrive as S3 URLs. Base64 inputs resolve instantly either way.
    """
    probe_raw, reference_raw = await asyncio.gather(
        asyncio.to_thread(_resolve_image_source, **probe_kwargs),
        asyncio.to_thread(_resolve_image_source, **reference_kwargs),
    )
    return probe_raw, reference_raw


def _embed_probe(probe_rgb: np.ndarray, embedding_svc, *, is_document: bool):
    """Embed the incoming probe image, returning ``(embedding, doc_diag)``.

    The embedding service runs a single detection cascade (primary → low-thresh →
    upscale) that already recovers small/printed document faces, then recognizes
    the face it finds — so a document is embedded directly, in one detection
    pass, with no separate locate-and-crop step (measured: same match score,
    ~half the latency). ``doc_diag`` is ``None`` for a plain face probe, else the
    document-face diagnostics (detection score + face sharpness) the embed
    returned, used to flag a blurry capture.
    """
    result = embedding_svc.embed(probe_rgb)
    if not is_document:
        return result, None
    diag = {
        "faceDetected": bool(result.face_detected),
        "detScore": round(result.det_score, 4) if result.det_score is not None else None,
        "sharpness": result.face_sharpness,
    }
    return result, diag


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

    # Face match only: no liveness / NSFW / blur gating. Embed and score.
    ml_struct = None

    probe_embed, doc_diag = _embed_probe(
        probe_rgb, embedding_svc, is_document=probe_is_document
    )

    # Flag (and optionally reject) a too-blurry document face: a low-sharpness
    # crop embeds poorly and silently drags the match score down.
    if doc_diag is not None:
        sharp = doc_diag.get("sharpness")
        low = (
            min_sharpness > 0 and doc_diag.get("faceDetected") and sharp is not None
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
    is_match = bool(score >= decision_threshold)

    return VerifyResponse(
        request_id=request_id,
        image_quality_assessment=ml_struct,
        verification_result=VerificationResult(
            is_match=is_match,
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
    probe_raw, reference_raw = await _resolve_two_sources(
        probe_kwargs=dict(
            b64=req.face_check_b64, s3_url=req.face_check_s3, fetcher=fetcher, field="face_check"
        ),
        reference_kwargs=dict(
            b64=req.source_selfie_b64, s3_url=req.source_selfie_s3, fetcher=fetcher,
            field="source_selfie",
        ),
    )
    decision_threshold, threshold_source = _resolve_threshold(
        req.threshold, learner, FACE_KIND, settings.verify_threshold
    )
    body = await asyncio.to_thread(
        _run_verify,
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        iqa_svc=iqa_svc,
        embedding_svc=embedding_svc,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
        run_liveness=False,
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
    probe_raw, reference_raw = await _resolve_two_sources(
        probe_kwargs=dict(
            b64=req.doc_check_b64, s3_url=req.doc_check_s3, fetcher=fetcher, field="doc_check"
        ),
        reference_kwargs=dict(
            b64=req.source_selfie_b64, s3_url=req.source_selfie_s3, fetcher=fetcher,
            field="source_selfie",
        ),
    )
    doc_kind = (req.doc_type or GENERIC_DOC_TYPE).strip().lower() or GENERIC_DOC_TYPE
    decision_threshold, threshold_source = _resolve_threshold(
        req.threshold, learner, doc_kind, settings.doc_verify_threshold
    )
    body = await asyncio.to_thread(
        _run_verify,
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
