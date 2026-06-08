"""Domain exception defaults."""

from zepiris.exceptions import (
    EmptyUploadError,
    ReferenceFaceNotFoundError,
    ReferenceImageDecodeError,
    ReferenceImageFetchError,
    ZepirisServiceError,
)


def test_zepiris_service_error_base() -> None:
    e = ZepirisServiceError("x", detail="y")
    assert e.status_code == 500
    assert e.detail == "y"


def test_empty_upload_error() -> None:
    e = EmptyUploadError()
    assert e.status_code == 400
    assert e.detail == "empty_upload"


def test_reference_image_fetch_error_is_400_with_reason() -> None:
    exc = ReferenceImageFetchError(reason="timeout", detail_msg="slow")
    assert exc.status_code == 400
    assert exc.detail["message"] == "reference_image_fetch_failed"
    assert exc.detail["reason"] == "timeout"


def test_reference_image_decode_error_is_400() -> None:
    exc = ReferenceImageDecodeError()
    assert exc.status_code == 400
    assert exc.detail == "reference_image_decode_failed"


def test_reference_face_not_found_error_is_400() -> None:
    exc = ReferenceFaceNotFoundError()
    assert exc.status_code == 400
    assert exc.detail == "reference_face_not_detected"
