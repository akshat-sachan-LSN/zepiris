"""The ML service's binary match endpoint: raw bytes in, one score out."""

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zepiris.framing import encode_pair_frame
from zepiris.ml_inference.concurrency import InferenceLimiter
from zepiris.ml_inference.deps import FaceEmbeddingDep
from zepiris.ml_inference.embedding_cache import (
    ReferenceEmbedding,
    ReferenceEmbeddingCache,
    reference_digest,
)
from zepiris.ml_inference.routes import router
from zepiris.schemas.ml_inference import FaceMatchResult


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _embedding() -> ReferenceEmbedding:
    return ReferenceEmbedding(
        vector=np.zeros(512, dtype=np.float32), face_detected=True, det_score=0.9
    )


class _Service:
    """Stands in for FaceEmbeddingService, mirroring how it treats the reference.

    The route hands the reference over as a **callable** so a cache hit never
    decodes those bytes. That only holds if the service resolves it lazily, so
    this fake resolves it exactly where the real one does — on a cache miss — and
    counts the resolutions, which is what the laziness test asserts on.
    """

    def __init__(self, result: FaceMatchResult, *, cache_size: int = 0) -> None:
        self._result = result
        self.calls: list[dict] = []
        self.resolved = 0
        self.reference_cache = ReferenceEmbeddingCache(max_entries=cache_size)

    def match_pair(
        self,
        probe_rgb,
        reference_rgb,
        *,
        want_probe_sharpness=False,
        reference_key=None,
        parallel=False,
    ):
        if self.reference_cache.get(reference_key) is None:
            reference = reference_rgb() if callable(reference_rgb) else reference_rgb
            self.resolved += 1
            self.reference_cache.put(reference_key, _embedding())
        else:
            reference = None
        self.calls.append(
            {
                "probe_shape": probe_rgb.shape,
                "reference_shape": None if reference is None else reference.shape,
                "want_probe_sharpness": want_probe_sharpness,
                "reference_key": reference_key,
                "parallel": parallel,
            }
        )
        return self._result


def _client(service, *, limit=4, parallel=False) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.inference_limiter = InferenceLimiter(limit=limit, queue_timeout=5.0)
    app.state.parallel_pair_embed = parallel
    app.state.face_embedding_service = service
    app.dependency_overrides[FaceEmbeddingDep.__metadata__[0].dependency] = lambda: service
    return TestClient(app)


def _post(client, probe=None, reference=None, **params):
    # `is None` rather than a falsy check — an empty image is a case under test.
    return client.post(
        "/v1/face/match",
        content=encode_pair_frame(
            _jpeg() if probe is None else probe,
            _jpeg() if reference is None else reference,
        ),
        headers={"Content-Type": "application/octet-stream"},
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


def test_splits_the_frame_at_the_declared_boundary() -> None:
    """Differently sized images must not bleed into each other."""
    ok, buf = cv2.imencode(".jpg", np.full((32, 48, 3), 200, dtype=np.uint8))
    assert ok
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    _post(_client(service), probe=buf.tobytes(), reference=_jpeg())
    assert service.calls[0]["probe_shape"] == (32, 48, 3)
    assert service.calls[0]["reference_shape"] == (64, 64, 3)


def test_rejects_a_truncated_frame() -> None:
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    r = _client(service).post(
        "/v1/face/match",
        content=b"\x00\x00",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 400
    assert "malformed_frame" in r.json()["detail"]


def test_rejects_a_frame_declaring_more_than_it_carries() -> None:
    """A length prefix longer than the body must fail, not read past the end."""
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    r = _client(service).post(
        "/v1/face/match",
        content=(999_999).to_bytes(4, "big") + b"short",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 400
    assert "malformed_frame" in r.json()["detail"]


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


# --- reference cache wiring -------------------------------------------------


def test_reference_is_keyed_by_its_own_bytes_when_caching_is_on() -> None:
    """The key must come from the bytes, so identical selfies share an entry."""
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True),
        cache_size=16,
    )
    ref = _jpeg()
    _post(_client(service), reference=ref)
    assert service.calls[0]["reference_key"] == reference_digest(ref)


def test_no_key_is_computed_when_the_cache_is_disabled() -> None:
    """Hashing every reference for a cache that cannot store it is pure cost."""
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True),
        cache_size=0,
    )
    _post(_client(service))
    assert service.calls[0]["reference_key"] is None


def test_parallel_embedding_is_requested_only_when_cores_are_idle() -> None:
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    _post(_client(service, limit=4, parallel=True))
    assert service.calls[0]["parallel"] is True


def test_parallel_embedding_is_off_when_the_service_is_saturated() -> None:
    """One free slot is this request's own; fanning out would steal throughput."""
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    _post(_client(service, limit=1, parallel=True))
    assert service.calls[0]["parallel"] is False


def test_parallel_embedding_respects_the_setting() -> None:
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    _post(_client(service, limit=8, parallel=False))
    assert service.calls[0]["parallel"] is False


def test_metrics_reports_the_cache() -> None:
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True),
        cache_size=16,
    )
    body = _client(service).get("/metrics").json()
    assert body["reference_cache"]["enabled"] is True
    assert body["reference_cache"]["max_entries"] == 16


# --- the parallel gate, at occupancies a TestClient cannot produce ------------


class _State:
    parallel_pair_embed = True


def test_gate_allows_fan_out_only_while_doubling_still_fits() -> None:
    """Each parallel request wants two cores, so `active * 2` must fit the limit."""
    from zepiris.ml_inference.routes import _may_embed_in_parallel

    limiter = InferenceLimiter(limit=8, queue_timeout=1.0)
    for active, expected in ((1, True), (4, True), (5, False), (8, False)):
        limiter._active = active
        assert _may_embed_in_parallel(_State(), limiter) is expected, f"active={active}"


def test_gate_closes_as_soon_as_anything_queues() -> None:
    """A backlog means the CPU is oversubscribed whatever the active count says."""
    from zepiris.ml_inference.routes import _may_embed_in_parallel

    limiter = InferenceLimiter(limit=8, queue_timeout=1.0)
    limiter._active = 1
    limiter._waiting = 1
    assert _may_embed_in_parallel(_State(), limiter) is False


def test_gate_is_off_when_the_setting_is_off() -> None:
    from zepiris.ml_inference.routes import _may_embed_in_parallel

    class _Off:
        parallel_pair_embed = False

    limiter = InferenceLimiter(limit=8, queue_timeout=1.0)
    limiter._active = 1
    assert _may_embed_in_parallel(_Off(), limiter) is False


def test_a_cached_reference_is_never_decoded() -> None:
    """The saving is the decode, not just the embed.

    The reference of a repeat verification is the same bytes every time, so once
    its embedding is cached there is nothing to learn from decoding the JPEG
    again — several milliseconds of CPU per request, on the majority of requests.
    """
    service = _Service(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True),
        cache_size=16,
    )
    client = _client(service)
    ref = _jpeg()
    _post(client, reference=ref)
    _post(client, reference=ref)
    assert service.resolved == 1
    assert service.calls[0]["reference_shape"] == (64, 64, 3)
    assert service.calls[1]["reference_shape"] is None


def test_the_reference_is_handed_over_lazily() -> None:
    """A callable, not an array — peeking the cache in the route would race.

    An entry can be evicted between a peek and the service's own lookup, and a
    caller that had already decided not to decode would have nothing to embed.
    """
    captured: list[object] = []

    class _Capture(_Service):
        def match_pair(self, probe_rgb, reference_rgb, **kw):
            captured.append(reference_rgb)
            return super().match_pair(probe_rgb, reference_rgb, **kw)

    service = _Capture(
        FaceMatchResult(score=0.5, probe_face_detected=True, reference_face_detected=True)
    )
    assert _post(_client(service)).status_code == 200
    assert callable(captured[0])
