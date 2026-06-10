import io

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.api.routes import face as face_routes
from zepiris.deps import EmbeddingDep, IQADep, S3FetcherDep, SettingsDep
from zepiris.exception_handlers import register_exception_handlers
from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    FaceEmbeddingResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)


def _jpeg_bytes() -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class _Settings:
    verify_threshold = 0.5
    doc_verify_threshold = 0.4


class _IQA:
    def __init__(self, *, live=True, passed=True) -> None:
        self._live, self._passed = live, passed

    def assess(self, image_rgb, image_b64) -> ImageQualityAssessmentResult:
        return ImageQualityAssessmentResult(
            passed=self._passed,
            nsfw=NSFWDetectionResult(is_safe=True, probability=0.01),
            spoof=SpoofDetectionResult(is_live=self._live, probability=0.99),
            blur=BlurDetectionResult(is_sharp=True, probability=0.95),
        )


class _Embedding:
    def __init__(self, *, live_vec, ref_vec, face=True) -> None:
        self._vecs = [live_vec, ref_vec]
        self._face = face

    def embed(self, image_rgb) -> FaceEmbeddingResult:
        vec = self._vecs.pop(0)
        return FaceEmbeddingResult(
            face_detected=self._face if vec is not None else False,
            embedding=vec or [],
            embedding_dim=len(vec or []),
        )


class _Fetcher:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def fetch(self, url: str) -> bytes:
        return self._data


def _client(iqa, embedding, fetcher) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(face_routes.router, prefix="/v1/faces")
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: _Settings()
    app.dependency_overrides[IQADep.__metadata__[0].dependency] = lambda: iqa
    app.dependency_overrides[EmbeddingDep.__metadata__[0].dependency] = lambda: embedding
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: fetcher
    return TestClient(app)


def _post(client, *, s3_url="https://s3/ref.jpg", path="/v1/faces/facematch/verify"):
    return client.post(
        path,
        data={"s3_url": s3_url},
        files={"file": ("live.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )


def test_match_when_vectors_identical() -> None:
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = _post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True
    assert body["verificationResult"]["score"] == pytest.approx(1.0)


def test_facematch_reference_via_uploaded_file() -> None:
    """facematch reference supplied as an uploaded photo instead of s3_url."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(b"unused"))
    r = client.post(
        "/v1/faces/facematch/verify",
        files={
            "file": ("selfie.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg"),
            "reference_file": ("photo.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg"),
        },
    )
    assert r.status_code == 200
    assert r.json()["verificationResult"]["isMatch"] is True


def test_docmatch_via_uploaded_document() -> None:
    """docmatch: selfie vs an uploaded Aadhaar/PAN document, lenient default threshold."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(b"unused"))
    r = client.post(
        "/v1/faces/docmatch/verify",
        files={
            "file": ("selfie.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg"),
            "document_file": ("aadhaar.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg"),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True
    assert body["verificationResult"]["threshold"] == pytest.approx(0.4)  # doc default


def test_facematch_s3_to_s3_skips_liveness() -> None:
    """Selfie + reference both via S3 URL: pure image match, no liveness/IQA gate."""
    vec = [1.0, 0.0, 0.0]
    # _IQA would report a result, but it must NOT be consulted in the S3->S3 path.
    client = _client(_IQA(live=False), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        data={"selfie_s3_url": "https://s3/selfie.jpg", "s3_url": "https://s3/ref.jpg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True          # matched despite live=False
    assert body["iqaPassed"] is False                              # liveness skipped
    assert body["imageQualityAssessment"] is None                 # no IQA run


def test_docmatch_s3_to_s3_selfie_and_document() -> None:
    """Selfie S3 link + document S3 link: extract doc face, compare, no liveness."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(live=False), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/docmatch/verify",
        data={"selfie_s3_url": "https://s3/selfie.jpg", "s3_url": "https://s3/aadhaar.jpg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True
    assert body["verificationResult"]["threshold"] == pytest.approx(0.4)
    assert body["imageQualityAssessment"] is None


def test_error_when_no_reference_provided() -> None:
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        files={"file": ("live.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )
    assert r.status_code == 400


def test_no_match_when_vectors_orthogonal() -> None:
    client = _client(
        _IQA(),
        _Embedding(live_vec=[1.0, 0.0], ref_vec=[0.0, 1.0]),
        _Fetcher(_jpeg_bytes()),
    )
    body = _post(client).json()
    assert body["verificationResult"]["isMatch"] is False


def test_liveness_failure_short_circuits() -> None:
    client = _client(
        _IQA(live=False),
        _Embedding(live_vec=[1.0], ref_vec=[1.0]),
        _Fetcher(_jpeg_bytes()),
    )
    body = _post(client).json()
    assert body["iqaPassed"] is False
    assert body["livenessFailed"] is True
    assert body["verificationResult"]["isMatch"] is False


def test_no_face_in_live_photo() -> None:
    client = _client(
        _IQA(),
        _Embedding(live_vec=None, ref_vec=[1.0], face=False),
        _Fetcher(_jpeg_bytes()),
    )
    body = _post(client).json()
    assert body["faceDetected"] is False
    assert body["verificationResult"]["isMatch"] is False
