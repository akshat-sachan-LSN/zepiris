"""The matcher contract: one pair in, one score out — over HTTP or in-process."""

import asyncio

import cv2
import httpx
import numpy as np
import pytest

from zepiris.exceptions import (
    MLInferenceTimeoutError,
    MLInferenceUpstreamError,
    ReferenceImageDecodeError,
)
from zepiris.framing import HEADER_SIZE, decode_pair_frame
from zepiris.schemas.ml_inference import FaceEmbeddingResult
from zepiris.services.matching import (
    LocalFaceMatcher,
    ProbeImageDecodeError,
    RemoteFaceMatcher,
)
from zepiris.services.ml_client import AsyncMLInferenceClient


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


class _Embedding:
    """Stub provider returning canned vectors in call order."""

    def __init__(self, *vecs, sharpness=42.0) -> None:
        self._vecs = list(vecs)
        self._sharpness = sharpness
        self.calls = 0

    def embed(self, image_rgb, **kwargs) -> FaceEmbeddingResult:
        self.calls += 1
        vec = self._vecs.pop(0)
        return FaceEmbeddingResult(
            face_detected=vec is not None,
            embedding=vec or [],
            embedding_dim=len(vec or []),
            det_score=0.9 if vec else None,
            face_sharpness=self._sharpness if vec else None,
        )


def _remote(handler) -> RemoteFaceMatcher:
    client = AsyncMLInferenceClient("http://ml")
    client.client = httpx.AsyncClient(
        base_url="http://ml", transport=httpx.MockTransport(handler)
    )
    return RemoteFaceMatcher(client)


# -- local -----------------------------------------------------------------


def test_local_scores_identical_vectors_as_one() -> None:
    matcher = LocalFaceMatcher(_Embedding([1.0, 0.0], [1.0, 0.0]))
    result = asyncio.run(matcher.match(_jpeg(), _jpeg()))
    assert result.score == pytest.approx(1.0)
    assert result.probe_face_detected and result.reference_face_detected


def test_local_scores_orthogonal_vectors_as_zero() -> None:
    matcher = LocalFaceMatcher(_Embedding([1.0, 0.0], [0.0, 1.0]))
    assert asyncio.run(matcher.match(_jpeg(), _jpeg())).score == pytest.approx(0.0)


def test_local_skips_reference_embed_when_probe_has_no_face() -> None:
    """Nothing to compare against — the reference embed would be wasted work."""
    embedding = _Embedding(None, [1.0, 0.0])
    result = asyncio.run(LocalFaceMatcher(embedding).match(_jpeg(), _jpeg()))
    assert result.probe_face_detected is False
    assert result.score is None
    assert embedding.calls == 1


def test_local_reports_missing_reference_face() -> None:
    matcher = LocalFaceMatcher(_Embedding([1.0, 0.0], None))
    result = asyncio.run(matcher.match(_jpeg(), _jpeg()))
    assert result.probe_face_detected is True
    assert result.reference_face_detected is False
    assert result.score is None


def test_local_sharpness_only_when_requested() -> None:
    """Face match never reads sharpness, so it must not be reported there."""
    matcher = LocalFaceMatcher(_Embedding([1.0], [1.0], sharpness=17.5))
    assert asyncio.run(matcher.match(_jpeg(), _jpeg())).probe_face_sharpness is None

    matcher = LocalFaceMatcher(_Embedding([1.0], [1.0], sharpness=17.5))
    doc = asyncio.run(matcher.match(_jpeg(), _jpeg(), want_probe_sharpness=True))
    assert doc.probe_face_sharpness == pytest.approx(17.5)


def test_local_raises_on_undecodable_probe() -> None:
    matcher = LocalFaceMatcher(_Embedding([1.0], [1.0]))
    with pytest.raises(ProbeImageDecodeError):
        asyncio.run(matcher.match(b"not an image", _jpeg()))


def test_local_raises_on_undecodable_reference() -> None:
    matcher = LocalFaceMatcher(_Embedding([1.0], [1.0]))
    with pytest.raises(ReferenceImageDecodeError):
        asyncio.run(matcher.match(_jpeg(), b"not an image"))


# -- remote ----------------------------------------------------------------


def test_remote_sends_both_images_raw_in_one_call() -> None:
    """The whole point of the path: original bytes, one round trip, no base64."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["count"] = seen.get("count", 0) + 1
        seen["body"] = request.content
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "score": 0.81,
                "probe_face_detected": True,
                "reference_face_detected": True,
            },
        )

    probe, reference = _jpeg(), _jpeg()
    result = asyncio.run(_remote(handler).match(probe, reference))

    assert result.score == pytest.approx(0.81)
    assert seen["count"] == 1
    assert "/v1/face/match" in seen["url"]
    # Both originals travel verbatim, and the body is exactly the two images plus
    # the 4-byte length prefix — no encoding expansion, no multipart scaffolding.
    assert decode_pair_frame(seen["body"]) == (probe, reference)
    assert len(seen["body"]) == len(probe) + len(reference) + HEADER_SIZE


def test_remote_maps_probe_decode_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "failed_to_decode_image: probe"})

    with pytest.raises(ProbeImageDecodeError):
        asyncio.run(_remote(handler).match(_jpeg(), _jpeg()))


def test_remote_maps_reference_decode_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "failed_to_decode_image: reference"})

    with pytest.raises(ReferenceImageDecodeError):
        asyncio.run(_remote(handler).match(_jpeg(), _jpeg()))


def test_remote_maps_overload_to_service_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": {"message": "inference_overloaded"}})

    with pytest.raises(MLInferenceUpstreamError) as exc:
        asyncio.run(_remote(handler).match(_jpeg(), _jpeg()))
    assert exc.value.status_code == 503


def test_remote_maps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    with pytest.raises(MLInferenceTimeoutError):
        asyncio.run(_remote(handler).match(_jpeg(), _jpeg()))
