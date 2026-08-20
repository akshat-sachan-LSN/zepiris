"""match_pair's caching and parallel paths, with the models stubbed out.

The point of these tests is the *work avoided*, not the score: a cache hit must
embed one side instead of two, and must produce exactly the score the uncached
path would have produced.
"""

import numpy as np
import pytest

from zepiris.ml_inference.embedding_cache import reference_digest
from zepiris.ml_inference.face_embedding import FaceEmbeddingService


class _StubbedService(FaceEmbeddingService):
    """A real service with the two model calls replaced by counted fakes.

    Subclassing rather than mocking keeps the caching logic under test exactly as
    it ships — only detection and recognition are stand-ins.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.probe_embeds = 0
        self.reference_embeds = 0
        self.no_face_for: set[int] = set()

    def _vector(self, image_rgb) -> np.ndarray:
        # Deterministic per image, so a cached vector is checkable against a fresh one.
        rng = np.random.default_rng(int(image_rgb[0, 0, 0]))
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def _embed_probe(self, probe_rgb, want_sharpness):
        self.probe_embeds += 1
        if int(probe_rgb[0, 0, 0]) in self.no_face_for:
            return np.zeros(512, dtype=np.float32), False, None, None
        return self._vector(probe_rgb), True, 0.91, (7.5 if want_sharpness else None)

    def embed_vector(self, image_rgb):
        self.reference_embeds += 1
        if int(image_rgb[0, 0, 0]) in self.no_face_for:
            return np.zeros(512, dtype=np.float32), False, None
        return self._vector(image_rgb), True, 0.88


def _image(tag: int) -> np.ndarray:
    return np.full((64, 64, 3), tag, dtype=np.uint8)


def _service(**kwargs) -> _StubbedService:
    return _StubbedService(device="cpu", **kwargs)


def test_repeat_reference_is_embedded_once() -> None:
    svc = _service(reference_cache_size=8)
    ref, key = _image(10), reference_digest(b"ref-10")

    first = svc.match_pair(_image(1), ref, reference_key=key)
    second = svc.match_pair(_image(2), ref, reference_key=key)

    assert svc.reference_embeds == 1, "the second verification re-embedded the reference"
    assert svc.probe_embeds == 2, "each probe must still be embedded — it is new pixels"
    assert first.reference_face_detected and second.reference_face_detected


def test_a_cache_hit_scores_identically_to_a_miss() -> None:
    """Caching must be invisible in the answer, only in the time taken."""
    cached = _service(reference_cache_size=8)
    uncached = _service(reference_cache_size=0)
    probe, ref, key = _image(3), _image(10), reference_digest(b"ref-10")

    cached.match_pair(probe, ref, reference_key=key)  # populate
    hit = cached.match_pair(probe, ref, reference_key=key)
    miss = uncached.match_pair(probe, ref, reference_key=key)

    assert hit.score == pytest.approx(miss.score)
    assert hit.reference_det_score == miss.reference_det_score


def test_different_references_do_not_share_an_entry() -> None:
    svc = _service(reference_cache_size=8)
    a = svc.match_pair(_image(1), _image(10), reference_key=reference_digest(b"a"))
    b = svc.match_pair(_image(1), _image(20), reference_key=reference_digest(b"b"))

    assert svc.reference_embeds == 2
    assert a.score != pytest.approx(b.score)


def test_no_key_bypasses_the_cache_entirely() -> None:
    svc = _service(reference_cache_size=8)
    svc.match_pair(_image(1), _image(10))
    svc.match_pair(_image(2), _image(10))
    assert svc.reference_embeds == 2
    assert svc.reference_cache.snapshot()["size"] == 0


def test_disabled_cache_embeds_every_time() -> None:
    svc = _service(reference_cache_size=0)
    key = reference_digest(b"ref")
    svc.match_pair(_image(1), _image(10), reference_key=key)
    svc.match_pair(_image(2), _image(10), reference_key=key)
    assert svc.reference_embeds == 2


def test_a_faceless_probe_still_skips_the_reference() -> None:
    """The short-circuit is the cheapest path there is; caching must not lose it."""
    svc = _service(reference_cache_size=8)
    svc.no_face_for.add(1)
    result = svc.match_pair(_image(1), _image(10), reference_key=reference_digest(b"r"))

    assert result.probe_face_detected is False
    assert result.score is None
    assert svc.reference_embeds == 0


def test_a_faceless_reference_is_cached_as_such() -> None:
    svc = _service(reference_cache_size=8)
    svc.no_face_for.add(10)
    key = reference_digest(b"bad-ref")

    first = svc.match_pair(_image(1), _image(10), reference_key=key)
    second = svc.match_pair(_image(2), _image(10), reference_key=key)

    assert first.reference_face_detected is False
    assert second.reference_face_detected is False
    assert svc.reference_embeds == 1, "an unusable reference was re-embedded on retry"


def test_sharpness_still_only_computed_on_request() -> None:
    svc = _service(reference_cache_size=8)
    key = reference_digest(b"r")
    assert svc.match_pair(_image(1), _image(10), reference_key=key).probe_face_sharpness is None
    got = svc.match_pair(_image(1), _image(10), reference_key=key, want_probe_sharpness=True)
    assert got.probe_face_sharpness == pytest.approx(7.5)


# --- parallel path ----------------------------------------------------------


def test_parallel_embedding_gives_the_same_score() -> None:
    svc = _service(reference_cache_size=0)
    probe, ref = _image(4), _image(10)
    sequential = svc.match_pair(probe, ref)
    parallel = svc.match_pair(probe, ref, parallel=True)
    assert parallel.score == pytest.approx(sequential.score)


def test_parallel_embedding_populates_the_cache() -> None:
    svc = _service(reference_cache_size=8)
    key = reference_digest(b"r")
    svc.match_pair(_image(1), _image(10), reference_key=key, parallel=True)
    svc.match_pair(_image(2), _image(10), reference_key=key, parallel=True)
    assert svc.reference_embeds == 1


def test_a_cache_hit_does_not_bother_going_parallel() -> None:
    """With the reference already known there is no second embed to overlap."""
    svc = _service(reference_cache_size=8)
    key = reference_digest(b"r")
    svc.match_pair(_image(1), _image(10), reference_key=key)
    svc.match_pair(_image(2), _image(10), reference_key=key, parallel=True)
    assert svc.reference_embeds == 1


def test_parallel_faceless_probe_reports_the_same_verdict() -> None:
    """Going parallel trades the short-circuit away; the answer must not change."""
    svc = _service(reference_cache_size=8)
    svc.no_face_for.add(1)
    result = svc.match_pair(
        _image(1), _image(10), reference_key=reference_digest(b"r"), parallel=True
    )
    assert result.probe_face_detected is False
    assert result.score is None
