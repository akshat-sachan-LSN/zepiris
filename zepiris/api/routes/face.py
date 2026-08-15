from __future__ import annotations

import asyncio
import base64
import binascii
import uuid

from fastapi import APIRouter, Form

from zepiris.deps import LearnerDep, MatcherDep, S3FetcherDep, SettingsDep
from zepiris.exceptions import (
    DocumentTooBlurryError,
    EmptyUploadError,
    FeedbackValidationError,
    ImageSourceError,
    ImageTooLargeError,
    ReferenceFaceNotFoundError,
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
from zepiris.services.matching import ProbeImageDecodeError

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


async def _record_outcome(learner, *, request_id: str, doc_type: str, body: dict) -> None:
    """Log a scored verification so operator feedback can calibrate thresholds.

    Only scores and the decision are logged — never images or personal data.
    Unscored results (decode/no-face failures) carry no signal for threshold
    fitting and are skipped.

    The write goes to a worker thread: it appends to a file under a process-wide
    lock, which on the event loop would serialize every request in the process
    behind one another's disk I/O — a global bottleneck on a path that is
    otherwise fully concurrent.
    """
    result = body.get("verificationResult") or {}
    score = result.get("score")
    if score is None:
        return
    scores = body.get("scores") or {}
    await asyncio.to_thread(
        learner.record_sample,
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


#: Above this payload size, base64 decoding moves to a worker thread. Decoding a
#: few hundred KB takes single-digit milliseconds, which is nothing once — and a
#: hard throughput ceiling when 100 requests do it on the event loop at the same
#: time, since none of them can make progress while one decodes. Below the
#: threshold the thread hop costs more than the decode it avoids.
_B64_OFFLOAD_THRESHOLD_BYTES = 64 * 1024


def _decode_b64_sync(value: str, *, field: str) -> bytes:
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


async def _decode_b64_image(value: str, *, field: str) -> bytes:
    """Decode an inline base64 image payload to raw, size-validated bytes.

    Accepts both a bare base64 string and a ``data:`` URI
    (e.g. ``data:image/jpeg;base64,<payload>``). Large payloads decode off the
    event loop — see :data:`_B64_OFFLOAD_THRESHOLD_BYTES`.
    """
    if len(value) >= _B64_OFFLOAD_THRESHOLD_BYTES:
        return await asyncio.to_thread(_decode_b64_sync, value, field=field)
    return _decode_b64_sync(value, field=field)


async def _resolve_image_source(
    *, b64: str | None, s3_url: str | None, fetcher, field: str
) -> bytes:
    """Resolve one image side (selfie or reference) to raw bytes.

    Each side is supplied via exactly one of two inputs — inline base64
    (``<field>_b64``) or an S3 URL (``<field>_s3``). Supplying both is
    ambiguous and rejected (400); supplying neither is missing (400).

    Whichever way the bytes arrive, they are passed through untouched — the
    matcher decodes them exactly once, wherever the models live.
    """
    has_b64 = bool(b64 and b64.strip())
    has_s3 = bool(s3_url and s3_url.strip())
    if has_b64 and has_s3:
        raise ImageSourceError(field=field, reason="ambiguous")
    if has_b64:
        return await _decode_b64_image(b64, field=field)
    if has_s3:
        raw = await fetcher.fetch(s3_url.strip())
        _validate_image_bytes(raw)
        return raw
    raise ImageSourceError(field=field, reason="missing")


async def _run_match(
    *,
    request_id: str,
    probe_raw: bytes,
    reference_raw: bytes,
    matcher,
    decision_threshold: float,
    threshold_source: str,
    probe_is_document: bool = False,
    min_sharpness: float = 0.0,
) -> dict:
    """Core 1:1 verification, scored in a single call to the ML service.

    The API never decodes or re-encodes an image on this path: the bytes that
    arrived (from base64 or S3) are the bytes the ML service receives, and only
    a similarity score comes back. Everything the old path spent per side — a
    decode, a lossless PNG re-encode, base64 both ways, and its own HTTP round
    trip — is gone, along with the 512-float vectors that used to cross the wire
    just to be dot-producted here.

    ``probe_raw`` is the image submitted *now* to be verified — a live face
    (facematch) or an ID document (docmatch). ``reference_raw`` is the user's
    enrolled selfie; it is only ever embedded.

    With ``probe_is_document`` the extracted face's sharpness is reported in
    ``documentFace``; when ``min_sharpness`` > 0 a too-blurry face is rejected.
    """

    def _verification_result(is_match: bool, score: float | None) -> dict:
        return {
            "isMatch": is_match,
            "score": score,
            "threshold": decision_threshold,
            "thresholdSource": threshold_source,
        }

    try:
        result = await matcher.match(
            probe_raw, reference_raw, want_probe_sharpness=probe_is_document
        )
    except ProbeImageDecodeError:
        # An unreadable capture is an ordinary outcome, not a fault: report it in
        # the response body the same way the caller sees every other verdict.
        return {
            "requestId": request_id,
            "decodeFailed": True,
            "faceDetected": False,
            "iqaPassed": False,
            "verificationResult": _verification_result(False, None),
            "scores": _scores_dict(None, decision_threshold, None),
        }

    doc_diag = None
    if probe_is_document:
        doc_diag = {
            "faceDetected": bool(result.probe_face_detected),
            "detScore": (
                round(result.probe_det_score, 4)
                if result.probe_det_score is not None
                else None
            ),
            "sharpness": result.probe_face_sharpness,
        }
        sharp = result.probe_face_sharpness
        low = (
            min_sharpness > 0
            and result.probe_face_detected
            and sharp is not None
            and sharp < min_sharpness
        )
        doc_diag["lowQuality"] = bool(low)
        if low:
            raise DocumentTooBlurryError(sharpness=sharp, min_sharpness=min_sharpness)

    if not result.probe_face_detected:
        return {
            "requestId": request_id,
            "imageQualityAssessment": None,
            "iqaPassed": True,
            "faceDetected": False,
            "verificationResult": _verification_result(False, None),
            "scores": _scores_dict(None, decision_threshold, None),
            "documentFace": doc_diag,
        }

    if not result.reference_face_detected:
        raise ReferenceFaceNotFoundError()

    score = float(result.score)
    return VerifyResponse(
        request_id=request_id,
        image_quality_assessment=None,
        verification_result=VerificationResult(
            is_match=bool(score >= decision_threshold),
            score=score,
            threshold=decision_threshold,
            threshold_source=threshold_source,
        ),
        scores=VerificationScores(**_scores_dict(score, decision_threshold, None)),
        document_face=doc_diag,
        face_detected=True,
        iqa_passed=True,
    ).model_dump(by_alias=True)


async def _resolve_two_sources(
    *, probe_kwargs: dict, reference_kwargs: dict
) -> tuple[bytes, bytes]:
    """Resolve the probe and reference images concurrently.

    Both sides are independent network work, so fetching them in parallel removes
    one S3 round-trip from the critical path when both arrive as S3 URLs. Base64
    inputs resolve instantly either way. Both fetches are awaited directly rather
    than handed to threads: at high concurrency, parking a worker thread per
    in-flight fetch would make the thread pool the bottleneck instead of S3.
    """
    probe_raw, reference_raw = await asyncio.gather(
        _resolve_image_source(**probe_kwargs),
        _resolve_image_source(**reference_kwargs),
    )
    return probe_raw, reference_raw


@router.post("/facematch/verify")
async def facematch_verify(
    req: FaceMatchRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
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
    body = await _run_match(
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        matcher=matcher,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
    )
    await _record_outcome(learner, request_id=request_id, doc_type=FACE_KIND, body=body)
    return body


@router.post("/docmatch/verify")
async def docmatch_verify(
    req: DocMatchRequest,
    settings: SettingsDep,
    matcher: MatcherDep,
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
    body = await _run_match(
        request_id=request_id,
        probe_raw=probe_raw,
        reference_raw=reference_raw,
        matcher=matcher,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
        probe_is_document=True,
        min_sharpness=settings.doc_min_sharpness,
    )
    await _record_outcome(learner, request_id=request_id, doc_type=doc_kind, body=body)
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
    matcher: MatcherDep,
    fetcher: S3FetcherDep,
    learner: LearnerDep,
) -> dict:
    """Deprecated alias of ``/facematch/verify`` (kept for existing callers)."""
    return await facematch_verify(
        req=req,
        settings=settings,
        matcher=matcher,
        fetcher=fetcher,
        learner=learner,
    )
