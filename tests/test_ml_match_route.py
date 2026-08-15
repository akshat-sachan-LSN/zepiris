"""The ML service's binary match endpoint: raw bytes in, one score out."""

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.ml_inference.concurrency import InferenceLimiter
from zepiris.ml_inference.deps import FaceEmbeddingDep
from zepiris.ml_inference.routes import router
from zepiris.schemas.ml_inference import FaceMatchResult


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _Service:
    def __init__(self, result: FaceMatchResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def match_pair(self, probe_rgb, reference_rgb, *, want_probe_sharpness=False):
        self.calls.append(
            {
                "probe_shape": probe_rgb.shape,
                "reference_shape": reference_rgb.shape,
                "want_probe_sharpness": want_probe_sharpness,
            }
        )
        return self._result


def _client(service, *, limit=4) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.inference_limiter = InferenceLimiter(limit=limit, queue_timeout=5.0)
    app.dependency_overrides[FaceEmbeddingDep.__metadata__[0].dependency] = lambda: service
    return TestClient(app)


def _post(client, probe=None, reference=None, **params):
    # `is None` rather than a falsy check — an empty body is a case under test.
    return client.post(
        "/v1/face/match",
        files={
            "probe": ("p", _jpeg() if probe is None else probe, "application/octet-stream"),
            "reference": (
                "r",
                _jpeg() if reference is None else reference,
                "application/octet-stream",
            ),
        },
        params=params,
    )


def test_returns_score_for_a_matching_pair() -> None:
    service = _Service(
        FaceMatchResult(score=0.77, probe_face_detected=True, reference_face_detected=True)
    )
    body = _post(_client(service)).json()
    assert body["score"] == pytest.approx(0.77)
    assert body["probe_face_detected"] is True


def test_decodes_both_images_from_raw_bytes() -> None:
    """No base64 anywhere: the endpoint takes the upload as it stands."""
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    assert _post(_client(service)).status_code == 200
    assert service.calls[0]["probe_shape"] == (64, 64, 3)
    assert service.calls[0]["reference_shape"] == (64, 64, 3)


def test_sharpness_is_opt_in() -> None:
    """Face match must not pay for a metric only the document path reads."""
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    client = _client(service)
    _post(client)
    assert service.calls[-1]["want_probe_sharpness"] is False
    _post(client, want_probe_sharpness="true")
    assert service.calls[-1]["want_probe_sharpness"] is True


def test_reports_missing_face_without_failing() -> None:
    service = _Service(
        FaceMatchResult(score=None, probe_face_detected=False, reference_face_detected=False)
    )
    r = _post(_client(service))
    assert r.status_code == 200
    assert r.json()["score"] is None


def test_rejects_undecodable_probe_naming_the_side() -> None:
    """The caller maps this onto a probe-specific outcome, so the side must be named."""
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    r = _post(_client(service), probe=b"not an image")
    assert r.status_code == 400
    assert "probe" in r.json()["detail"]


def test_rejects_undecodable_reference_naming_the_side() -> None:
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    r = _post(_client(service), reference=b"not an image")
    assert r.status_code == 400
    assert "reference" in r.json()["detail"]


def test_rejects_empty_upload() -> None:
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    r = _post(_client(service), probe=b"")
    assert r.status_code == 400
