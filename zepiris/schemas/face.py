from __future__ import annotations

from pydantic import BaseModel, Field

from zepiris.schemas.ml_inference import ImageQualityAssessmentResult

MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# 1:1 verification (stateless: live photo vs S3 reference)
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Outcome of a 1:1 verification."""

    is_match: bool = Field(..., alias="isMatch")
    score: float | None = None
    threshold: float

    model_config = {"populate_by_name": True}


class VerifyResponse(BaseModel):
    """Response for the stateless 1:1 verify endpoint.

    ``image_quality_assessment`` is null for pure image-to-image (S3↔S3) matches,
    where liveness/quality gating is skipped because neither side is a live capture.
    """

    request_id: str = Field(..., alias="requestId")
    image_quality_assessment: ImageQualityAssessmentResult | None = Field(
        None, alias="imageQualityAssessment"
    )
    verification_result: VerificationResult = Field(..., alias="verificationResult")
    face_detected: bool = Field(..., alias="faceDetected")
    iqa_passed: bool = Field(..., alias="iqaPassed")

    model_config = {"populate_by_name": True}
