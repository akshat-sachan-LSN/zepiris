from zepiris.schemas.face import VerificationResult, VerifyResponse
from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)


def _iqa() -> ImageQualityAssessmentResult:
    return ImageQualityAssessmentResult(
        passed=True,
        nsfw=NSFWDetectionResult(is_safe=True, probability=0.01),
        spoof=SpoofDetectionResult(is_live=True, probability=0.99),
        blur=BlurDetectionResult(is_sharp=True, probability=0.95),
    )


def test_verify_response_camelcase_aliases() -> None:
    resp = VerifyResponse(
        request_id="abc",
        image_quality_assessment=_iqa(),
        verification_result=VerificationResult(is_match=True, score=0.91, threshold=0.5),
        face_detected=True,
        iqa_passed=True,
    )
    data = resp.model_dump(by_alias=True)
    assert data["requestId"] == "abc"
    assert data["verificationResult"]["isMatch"] is True
    assert data["verificationResult"]["score"] == 0.91
    assert data["faceDetected"] is True
    assert data["iqaPassed"] is True
