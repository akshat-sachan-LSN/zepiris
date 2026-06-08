"""Domain exceptions for the main API.

Raised from routes and services; translated to JSON by ``register_exception_handlers``
in :mod:`zepiris.exception_handlers`. Prefer these over raw ``HTTPException`` in
application code so status codes and payloads stay consistent.
"""

from __future__ import annotations

from typing import Any


class ZepirisServiceError(Exception):
    """Base class for expected API failures (4xx/5xx with a stable ``detail`` shape)."""

    default_status_code: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        detail: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code if status_code is not None else self.default_status_code
        self.detail: str | dict[str, Any] | list[Any] = (
            detail if detail is not None else (message or "error")
        )


# --- Client / upload validation ---


class EmptyUploadError(ZepirisServiceError):
    default_status_code = 400

    def __init__(self) -> None:
        super().__init__("empty_upload", detail="empty_upload")


class ImageTooLargeError(ZepirisServiceError):
    default_status_code = 422

    def __init__(self, mb: float, max_mb: float) -> None:
        d = f"image_too_large_{mb:.2f}MB_max_{max_mb}MB"
        super().__init__(d, detail=d)


class ImageEncodeError(ZepirisServiceError):
    default_status_code = 400

    def __init__(self) -> None:
        super().__init__("image_encode_failed", detail="image_encode_failed")


# --- ML inference (HTTP client) ---


class MLInferenceUpstreamError(ZepirisServiceError):
    """ML microservice returned a non-success HTTP status."""

    default_status_code = 502

    def __init__(
        self,
        *,
        status_code: int,
        detail: dict[str, Any],
    ) -> None:
        super().__init__(
            str(detail.get("message", "ml_inference_request_failed")),
            status_code=status_code,
            detail=detail,
        )


class MLInferenceTimeoutError(ZepirisServiceError):
    default_status_code = 503

    def __init__(self) -> None:
        super().__init__(
            "ml_inference_timeout",
            detail="ml_inference_request_timeout",
        )


class MLInferenceTransportError(ZepirisServiceError):
    """Network / connection error talking to the ML microservice."""

    default_status_code = 503

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            detail={"message": "ml_inference_unreachable", "reason": message},
        )


# --- Face / quality ---


class ImageQualityCheckFailedError(ZepirisServiceError):
    default_status_code = 422

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__("image_quality_check_failed", detail=detail)


class LivenessCheckFailedError(ZepirisServiceError):
    """Raised when the image is a spoof (photo/screen replay), not a live face.

    Distinct from :class:`ImageQualityCheckFailedError` so callers can tell a
    liveness rejection apart from blur/NSFW quality failures.
    """

    default_status_code = 422

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__("liveness_failed", detail=detail)


# --- Reference image (S3) ---


class ReferenceImageFetchError(ZepirisServiceError):
    """Could not retrieve the reference image from the supplied S3 URL."""

    default_status_code = 400

    def __init__(self, reason: str, detail_msg: str) -> None:
        super().__init__(
            "reference_image_fetch_failed",
            detail={
                "message": "reference_image_fetch_failed",
                "reason": reason,
                "detail": detail_msg,
            },
        )


class ReferenceImageDecodeError(ZepirisServiceError):
    """Fetched reference bytes are not a decodable image."""

    default_status_code = 400

    def __init__(self) -> None:
        super().__init__(
            "reference_image_decode_failed", detail="reference_image_decode_failed"
        )


class ReferenceFaceNotFoundError(ZepirisServiceError):
    """No face detected in the reference image."""

    default_status_code = 400

    def __init__(self) -> None:
        super().__init__(
            "reference_face_not_detected", detail="reference_face_not_detected"
        )
