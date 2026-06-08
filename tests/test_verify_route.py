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


def _post(client, *, s3_url="https://s3/ref.jpg"):
    return client.post(
        "/v1/faces/verify",
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
