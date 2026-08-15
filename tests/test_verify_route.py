import base64

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.api.routes import face as face_routes
from zepiris.deps import (
    EmbeddingDep,
    IQADep,
    LearnerDep,
    MatcherDep,
    S3FetcherDep,
    SettingsDep,
)
from zepiris.exception_handlers import register_exception_handlers
from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    FaceDetectionResult,
    FaceEmbeddingResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)
from zepiris.services.matching import LocalFaceMatcher


def _jpeg_bytes() -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _jpeg_b64() -> str:
    return base64.b64encode(_jpeg_bytes()).decode()


class _Settings:
    verify_threshold = 0.5
    doc_verify_threshold = 0.4
    doc_min_sharpness = 0.0


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
    def __init__(
        self, *, live_vec, ref_vec, face=True, extra_vecs=None, det_score=0.9, sharpness=50.0
    ) -> None:
        self._vecs = [live_vec, ref_vec, *(extra_vecs or [])]
        self._face = face
        self._det_score = det_score
        self._sharpness = sharpness
        self.embedded_shapes: list[tuple] = []

    def embed(self, image_rgb) -> FaceEmbeddingResult:
        self.embedded_shapes.append(image_rgb.shape)
        vec = self._vecs.pop(0)
        detected = self._face if vec is not None else False
        return FaceEmbeddingResult(
            face_detected=detected,
            embedding=vec or [],
            embedding_dim=len(vec or []),
            det_score=self._det_score if detected else None,
            face_sharpness=self._sharpness if detected else None,
        )

    def detect_box(self, image_rgb) -> FaceDetectionResult:
        return FaceDetectionResult(face_detected=True, bbox=[0.0, 0.0, 1.0, 1.0], score=0.9)


class _Fetcher:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def fetch(self, url: str) -> bytes:
        return self._data


class _Learner:
    """Stub adaptive learner: records calls, serves canned learned thresholds."""

    def __init__(self, learned: dict[str, float] | None = None) -> None:
        self._learned = learned or {}
        self.samples: list[dict] = []
        self.feedback: list[dict] = []

    def learned_threshold(self, doc_type: str) -> float | None:
        return self._learned.get(doc_type)

    def record_sample(self, **kwargs) -> None:
        self.samples.append(kwargs)

    def record_feedback(self, *, request_id: str, genuine: bool) -> dict:
        self.feedback.append({"request_id": request_id, "genuine": genuine})
        return {"recorded": True, "matched_sample": False, "doc_type": None, "thresholds": {}}


def _client(iqa, embedding, fetcher, learner=None) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(face_routes.router, prefix="/v1/faces")
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: _Settings()
    app.dependency_overrides[IQADep.__metadata__[0].dependency] = lambda: iqa
    app.dependency_overrides[EmbeddingDep.__metadata__[0].dependency] = lambda: embedding
    # The routes score through a matcher; LocalFaceMatcher is the in-process one,
    # so the stub embedding provider still drives the assertions below.
    app.dependency_overrides[MatcherDep.__metadata__[0].dependency] = lambda: LocalFaceMatcher(
        embedding
    )
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: fetcher
    app.dependency_overrides[LearnerDep.__metadata__[0].dependency] = lambda: learner or _Learner()
    return TestClient(app)


def _probe_s3_field(path: str) -> str:
    """The probe-side (incoming image) S3 param name differs per endpoint."""
    return "doc_check_s3" if "docmatch" in path else "face_check_s3"


def _post(client, *, s3_url="https://s3/ref.jpg", path="/v1/faces/facematch/verify"):
    """Canonical request (JSON body): probe from the S3 URL, base64 source selfie."""
    return client.post(
        path, json={"source_selfie_b64": _jpeg_b64(), _probe_s3_field(path): s3_url}
    )


def test_match_when_vectors_identical() -> None:
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = _post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True
    assert body["verificationResult"]["score"] == pytest.approx(1.0)
    assert body["iqaPassed"] is True
    assert body["imageQualityAssessment"] is None          # facematch runs no liveness/IQA


def test_docmatch_document_vs_source_selfie() -> None:
    """docmatch: uploaded document (doc_check) vs enrolled selfie, lenient default.

    docmatch runs NO liveness/IQA gate (the probe is a printed document, not a
    live face), so imageQualityAssessment is absent.
    """
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = _post(client, path="/v1/faces/docmatch/verify", s3_url="https://s3/aadhaar.jpg")
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True
    assert body["iqaPassed"] is True
    assert body["imageQualityAssessment"] is None  # no liveness gate on documents
    assert body["verificationResult"]["threshold"] == pytest.approx(0.4)  # doc default


def test_facematch_returns_score_block() -> None:
    """The response exposes a flat scores block: match + margin. Liveness/blur/nsfw
    are null because facematch runs no IQA/liveness gate (pure 1:1 match)."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    body = _post(client).json()
    scores = body["scores"]
    assert scores["matchScore"] == pytest.approx(1.0)
    assert scores["threshold"] == pytest.approx(0.5)
    assert scores["margin"] == pytest.approx(0.5)
    assert scores["livenessScore"] is None
    assert scores["blurScore"] is None
    assert scores["nsfwSafeScore"] is None
    assert body["verificationResult"]["thresholdSource"] == "default"


def test_docmatch_scores_have_no_liveness_numbers() -> None:
    """docmatch runs no IQA, so quality scores are null but the match score is present."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    body = _post(client, path="/v1/faces/docmatch/verify", s3_url="https://s3/aadhaar.jpg").json()
    scores = body["scores"]
    assert scores["matchScore"] == pytest.approx(1.0)
    assert scores["livenessScore"] is None
    assert scores["blurScore"] is None
    assert scores["nsfwSafeScore"] is None


def test_threshold_source_reports_learned() -> None:
    """When a learned threshold is applied, the response says so."""
    vec = [1.0, 0.0, 0.0]
    learner = _Learner(learned={"aadhaar": 0.55})
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()), learner)
    r = client.post(
        "/v1/faces/docmatch/verify",
        json={
            "source_selfie_b64": _jpeg_b64(),
            "doc_check_s3": "https://s3/aadhaar.jpg",
            "doc_type": "aadhaar",
        },
    )
    assert r.json()["verificationResult"]["thresholdSource"] == "learned"


def test_recorded_sample_has_no_quality_scores() -> None:
    """facematch logs the match score for learning; quality scores are null since
    no IQA/liveness gate runs (pure 1:1 match)."""
    vec = [1.0, 0.0, 0.0]
    learner = _Learner()
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()), learner)
    _post(client)
    assert len(learner.samples) == 1
    sample = learner.samples[0]
    assert sample["score"] == pytest.approx(1.0)
    assert sample["liveness_score"] is None
    assert sample["blur_score"] is None
    assert sample["nsfw_safe_score"] is None


def test_docmatch_reports_document_face_diagnostics() -> None:
    """docmatch surfaces a documentFace block (detection score, sharpness) from the embed."""
    vec = [1.0, 0.0, 0.0]
    embedding = _Embedding(live_vec=vec, ref_vec=vec, det_score=0.87, sharpness=42.0)
    client = _client(_IQA(), embedding, _Fetcher(_jpeg_bytes()))
    body = _post(client, path="/v1/faces/docmatch/verify", s3_url="https://s3/aadhaar.jpg").json()
    doc = body["documentFace"]
    assert doc["faceDetected"] is True
    assert doc["detScore"] == pytest.approx(0.87)
    assert doc["sharpness"] == pytest.approx(42.0)
    assert doc["lowQuality"] is False  # min_sharpness disabled by default


def test_facematch_has_no_document_face_block() -> None:
    """facematch is not a document flow, so documentFace is null."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    assert _post(client).json()["documentFace"] is None


def test_docmatch_rejects_too_blurry_document_when_gate_enabled() -> None:
    """With doc_min_sharpness set, a sub-threshold document face is rejected (422)."""

    class _StrictSettings(_Settings):
        doc_min_sharpness = 25.0

    vec = [1.0, 0.0, 0.0]
    embedding = _Embedding(live_vec=vec, ref_vec=vec, sharpness=6.8)  # below the gate
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(face_routes.router, prefix="/v1/faces")
    app.dependency_overrides[SettingsDep.__metadata__[0].dependency] = lambda: _StrictSettings()
    app.dependency_overrides[IQADep.__metadata__[0].dependency] = lambda: _IQA()
    app.dependency_overrides[EmbeddingDep.__metadata__[0].dependency] = lambda: embedding
    # The routes score through a matcher; LocalFaceMatcher is the in-process one,
    # so the stub embedding provider still drives the assertions below.
    app.dependency_overrides[MatcherDep.__metadata__[0].dependency] = lambda: LocalFaceMatcher(
        embedding
    )
    app.dependency_overrides[S3FetcherDep.__metadata__[0].dependency] = lambda: _Fetcher(
        _jpeg_bytes()
    )
    app.dependency_overrides[LearnerDep.__metadata__[0].dependency] = lambda: _Learner()
    client = TestClient(app)
    r = client.post(
        "/v1/faces/docmatch/verify",
        json={"source_selfie_b64": _jpeg_b64(), "doc_check_s3": "https://s3/aadhaar.jpg"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["message"] == "document_too_blurry"


def test_docmatch_does_not_run_liveness() -> None:
    """A spoof-flagged document still verifies: docmatch never runs the liveness gate."""
    vec = [1.0, 0.0, 0.0]
    # _IQA(live=False) would short-circuit facematch, but docmatch never calls it.
    client = _client(
        _IQA(live=False), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes())
    )
    r = _post(client, path="/v1/faces/docmatch/verify", s3_url="https://s3/aadhaar.jpg")
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True
    assert "livenessFailed" not in body


def test_docmatch_embeds_document_in_one_pass() -> None:
    """docmatch embeds the document directly (no separate locate+crop detection)."""
    vec = [1.0, 0.0, 0.0]
    embedding = _Embedding(live_vec=vec, ref_vec=vec)
    client = _client(_IQA(), embedding, _Fetcher(_jpeg_bytes()))
    r = _post(client, path="/v1/faces/docmatch/verify", s3_url="https://s3/aadhaar.jpg")
    assert r.status_code == 200
    assert r.json()["verificationResult"]["isMatch"] is True
    # exactly two embeds: the document probe, then the source selfie — no extra
    # detection/crop pass.
    assert len(embedding.embedded_shapes) == 2


def test_docmatch_uses_learned_threshold_for_doc_type() -> None:
    """With no explicit threshold, the learned per-doc-type threshold wins over the default."""
    vec = [1.0, 0.0, 0.0]
    learner = _Learner(learned={"aadhaar": 0.55})
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()), learner)
    r = client.post(
        "/v1/faces/docmatch/verify",
        json={
            "source_selfie_b64": _jpeg_b64(),
            "doc_check_s3": "https://s3/aadhaar.jpg",
            "doc_type": "Aadhaar",
        },
    )
    assert r.status_code == 200
    assert r.json()["verificationResult"]["threshold"] == pytest.approx(0.55)


def test_explicit_threshold_beats_learned() -> None:
    vec = [1.0, 0.0, 0.0]
    learner = _Learner(learned={"pan": 0.55})
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()), learner)
    r = client.post(
        "/v1/faces/docmatch/verify",
        json={
            "source_selfie_b64": _jpeg_b64(),
            "doc_check_s3": "https://s3/pan.jpg",
            "doc_type": "pan",
            "threshold": 0.6,
        },
    )
    assert r.json()["verificationResult"]["threshold"] == pytest.approx(0.6)


def test_scored_verification_is_recorded_for_learning() -> None:
    vec = [1.0, 0.0, 0.0]
    learner = _Learner()
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()), learner)
    r = client.post(
        "/v1/faces/docmatch/verify",
        json={
            "source_selfie_b64": _jpeg_b64(),
            "doc_check_s3": "https://s3/aadhaar.jpg",
            "doc_type": "aadhaar",
        },
    )
    assert r.status_code == 200
    assert len(learner.samples) == 1
    sample = learner.samples[0]
    assert sample["doc_type"] == "aadhaar"
    assert sample["request_id"] == r.json()["requestId"]
    assert sample["score"] == pytest.approx(1.0)


def test_unscored_verification_is_not_recorded() -> None:
    """No face in the probe carries no match score, so nothing is logged for learning."""
    learner = _Learner()
    client = _client(
        _IQA(), _Embedding(live_vec=None, ref_vec=[1.0], face=False), _Fetcher(_jpeg_bytes()), learner
    )
    _post(client)
    assert learner.samples == []


def test_feedback_endpoint_records_outcome() -> None:
    learner = _Learner()
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()), learner)
    r = client.post("/v1/faces/feedback", data={"request_id": "abc-123", "genuine": "true"})
    assert r.status_code == 200
    assert r.json()["recorded"] is True
    assert learner.feedback == [{"request_id": "abc-123", "genuine": True}]


def test_feedback_endpoint_requires_fields() -> None:
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()))
    r = client.post("/v1/faces/feedback", data={"request_id": "abc-123"})
    assert r.status_code == 400


def test_selfie_b64_accepts_data_uri() -> None:
    """A data: URI payload is accepted for the selfie."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        json={
            "source_selfie_b64": f"data:image/jpeg;base64,{_jpeg_b64()}",
            "face_check_s3": "https://s3/ref.jpg",
        },
    )
    assert r.status_code == 200
    assert r.json()["verificationResult"]["isMatch"] is True


def test_invalid_base64_selfie_rejected() -> None:
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        json={"source_selfie_b64": "!!!not-valid-base64!!!", "face_check_s3": "https://s3/ref.jpg"},
    )
    assert r.status_code == 400


def test_facematch_probe_and_source_from_s3() -> None:
    """Both sides can be S3 URLs; facematch is a pure 1:1 match (no liveness gate)."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        json={"face_check_s3": "https://s3/live.jpg", "source_selfie_s3": "https://s3/db.jpg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verificationResult"]["isMatch"] is True
    assert body["iqaPassed"] is True
    assert body["imageQualityAssessment"] is None  # no liveness gate


def test_source_selfie_from_b64_matches() -> None:
    """The enrolled source selfie can arrive as inline base64 instead of an S3 URL."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        json={"source_selfie_b64": _jpeg_b64(), "face_check_b64": _jpeg_b64()},
    )
    assert r.status_code == 200
    assert r.json()["verificationResult"]["isMatch"] is True


def test_docmatch_reference_from_b64() -> None:
    """docmatch also accepts the document as inline base64 (doc_check_b64)."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/docmatch/verify",
        json={
            "source_selfie_b64": _jpeg_b64(),
            "doc_check_b64": _jpeg_b64(),
            "doc_type": "aadhaar",
        },
    )
    assert r.status_code == 200
    assert r.json()["verificationResult"]["isMatch"] is True


def test_error_when_no_selfie_provided() -> None:
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()))
    r = client.post("/v1/faces/facematch/verify", json={"face_check_s3": "https://s3/ref.jpg"})
    assert r.status_code == 400


def test_error_when_no_reference_provided() -> None:
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()))
    r = client.post("/v1/faces/facematch/verify", json={"source_selfie_b64": _jpeg_b64()})
    assert r.status_code == 400


def test_error_when_both_selfie_sources_provided() -> None:
    """Both base64 and S3 for the selfie is ambiguous -> rejected."""
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        json={
            "source_selfie_b64": _jpeg_b64(),
            "source_selfie_s3": "https://s3/selfie.jpg",
            "face_check_s3": "https://s3/ref.jpg",
        },
    )
    assert r.status_code == 400


def test_error_when_both_reference_sources_provided() -> None:
    """Both base64 and S3 for the reference is ambiguous -> rejected."""
    client = _client(_IQA(), _Embedding(live_vec=[1.0], ref_vec=[1.0]), _Fetcher(_jpeg_bytes()))
    r = client.post(
        "/v1/faces/facematch/verify",
        json={
            "source_selfie_b64": _jpeg_b64(),
            "face_check_b64": _jpeg_b64(),
            "face_check_s3": "https://s3/ref.jpg",
        },
    )
    assert r.status_code == 400


def test_facematch_is_pure_match_no_liveness() -> None:
    """facematch is a pure 1:1 match: no IQA/liveness gate runs, so a
    spoof-flagged probe still verifies on match score alone."""
    vec = [1.0, 0.0, 0.0]
    client = _client(_IQA(live=False), _Embedding(live_vec=vec, ref_vec=vec), _Fetcher(_jpeg_bytes()))
    body = _post(client).json()
    assert body["verificationResult"]["isMatch"] is True     # matched on score alone
    assert body["imageQualityAssessment"] is None            # liveness gate did not run
    assert "livenessFailed" not in body                      # spoof flag ignored
    assert body["scores"]["livenessScore"] is None


def test_no_match_when_vectors_orthogonal() -> None:
    client = _client(
        _IQA(),
        _Embedding(live_vec=[1.0, 0.0], ref_vec=[0.0, 1.0]),
        _Fetcher(_jpeg_bytes()),
    )
    body = _post(client).json()
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
