"""Pydantic schemas for ML inference service results."""

from pydantic import BaseModel, Field


class FaceEmbeddingResult(BaseModel):
    """Result of face embedding inference.

    Attributes:
        face_detected: Whether a face was found in the input image
        embedding: L2-normalized face embedding vector (zero vector when no face detected)
        embedding_dim: Dimension of the embedding vector
    """

    face_detected: bool
    embedding: list[float]
    embedding_dim: int


class FaceDetectionResult(BaseModel):
    """Result of a lightweight face-detection-only pass (no recognition).

    Attributes:
        face_detected: Whether a face was found
        bbox: [x1, y1, x2, y2] of the selected face, normalized to [0, 1]
              relative to the input image (all zeros when no face)
        score: Detector confidence for the selected face
    """

    face_detected: bool
    bbox: list[float]
    score: float = 0.0


class SpoofDetectionResult(BaseModel):
    """Result of spoof detection inference.

    Attributes:
        is_live: True if image is genuine/live, False if spoofed
        probability: Probability of image being live in [0, 1] range
    """

    is_live: bool
    probability: float = Field(ge=0.0, le=1.0)


class NSFWDetectionResult(BaseModel):
    """Result of NSFW detection inference.

    Attributes:
        is_safe: True if content is safe, False if NSFW detected
        probability: Probability of content being safe in [0, 1] range
    """

    is_safe: bool
    probability: float = Field(ge=0.0, le=1.0)


class BlurDetectionResult(BaseModel):
    """Result of blur detection inference.

    Attributes:
        is_sharp: True if image is sharp, False if blurry
        probability: Probability of image being sharp in [0, 1] range
    """

    is_sharp: bool
    probability: float = Field(ge=0.0, le=1.0)


class ImageQualityAssessmentResult(BaseModel):
    """Combined result of image quality assessment across multiple checks.

    Runs three quality checks on an image: NSFW detection, spoof detection,
    and blur detection. Returns aggregated result.

    Attributes:
        passed: True if all quality checks pass, False if any check fails
        nsfw: NSFW detection result
        spoof: Spoof detection result
        blur: Blur detection result
    """

    passed: bool
    nsfw: NSFWDetectionResult
    spoof: SpoofDetectionResult
    blur: BlurDetectionResult
