from __future__ import annotations

from pydantic import BaseModel, Field

from zepiris.schemas.ml_inference import ImageQualityAssessmentResult

MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# 1:1 verification (stateless: live base64 selfie vs S3 reference)
# ---------------------------------------------------------------------------


class FaceMatchRequest(BaseModel):
    """JSON body for ``POST /v1/faces/facematch/verify`` (and the ``/verify`` alias).

    Supply exactly one input per side. ``source_selfie_*`` is the enrolled
    selfie (source of truth, only embedded); ``face_check_*`` is the incoming
    live face being verified (liveness-gated).
    """

    source_selfie_b64: str | None = None
    source_selfie_s3: str | None = None
    face_check_b64: str | None = None
    face_check_s3: str | None = None
    threshold: float | None = None


class DocMatchRequest(BaseModel):
    """JSON body for ``POST /v1/faces/docmatch/verify``.

    ``source_selfie_*`` is the enrolled selfie (source of truth, only embedded);
    ``doc_check_*`` is the incoming ID document being verified (face extracted,
    no liveness gate). ``doc_type`` buckets the request for adaptive learning.
    """

    source_selfie_b64: str | None = None
    source_selfie_s3: str | None = None
    doc_check_b64: str | None = None
    doc_check_s3: str | None = None
    threshold: float | None = None
    doc_type: str | None = None


class VerificationResult(BaseModel):
    """Outcome of a 1:1 verification."""

    is_match: bool = Field(..., alias="isMatch")
    score: float | None = None
    threshold: float
    #: Where ``threshold`` came from: "explicit" (per-request), "learned"
    #: (adaptively calibrated from feedback for this doc_type), or "default".
    threshold_source: str | None = Field(None, alias="thresholdSource")

    model_config = {"populate_by_name": True}


class VerificationScores(BaseModel):
    """Flat numeric summary of a verification, for easy downstream consumption.

    The same numbers also live in ``verificationResult`` and
    ``imageQualityAssessment``; this block gathers them in one place. Quality
    scores are ``null`` when the liveness/IQA gate did not run (e.g. docmatch).
    """

    #: Cosine similarity of the two embeddings in [0, 1] (null if not scored).
    match_score: float | None = Field(None, alias="matchScore")
    #: Decision threshold applied to ``match_score``.
    threshold: float
    #: ``match_score - threshold`` — how far over/under the line (null if unscored).
    margin: float | None = None
    #: Probability the probe is a live face in [0, 1] (spoof check).
    liveness_score: float | None = Field(None, alias="livenessScore")
    #: Probability the probe is sharp in [0, 1] (blur check).
    blur_score: float | None = Field(None, alias="blurScore")
    #: Probability the probe is safe in [0, 1] (NSFW check).
    nsfw_safe_score: float | None = Field(None, alias="nsfwSafeScore")

    model_config = {"populate_by_name": True}


class VerifyResponse(BaseModel):
    """Response for the stateless 1:1 verify endpoint.

    The selfie is always a live base64 capture, so ``image_quality_assessment``
    carries the liveness/quality result; it is null only on an early decode failure.
    """

    request_id: str = Field(..., alias="requestId")
    image_quality_assessment: ImageQualityAssessmentResult | None = Field(
        None, alias="imageQualityAssessment"
    )
    verification_result: VerificationResult = Field(..., alias="verificationResult")
    scores: VerificationScores | None = None
    #: docmatch only — diagnostics for the face extracted from the document
    #: (detection score, sharpness, whether the crop was used, low-quality flag).
    document_face: dict | None = Field(None, alias="documentFace")
    face_detected: bool = Field(..., alias="faceDetected")
    iqa_passed: bool = Field(..., alias="iqaPassed")

    model_config = {"populate_by_name": True}
