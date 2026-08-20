"""The reference-embedding cache: keying, eviction, and the disabled case."""

import numpy as np

from zepiris.ml_inference.embedding_cache import (
    ReferenceEmbedding,
    ReferenceEmbeddingCache,
    reference_digest,
)


def _emb(seed: int = 0) -> ReferenceEmbedding:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return ReferenceEmbedding(vector=v / np.linalg.norm(v), face_detected=True, det_score=0.9)


def test_digest_is_content_addressed() -> None:
    assert reference_digest(b"same") == reference_digest(b"same")
    assert reference_digest(b"same") != reference_digest(b"different")
    # Length is fixed regardless of input size — it is a key, not a summary.
    assert len(reference_digest(b"x")) == len(reference_digest(b"x" * 1_000_000)) == 32


def test_hit_returns_the_same_embedding() -> None:
    cache = ReferenceEmbeddingCache(max_entries=8)
    value = _emb()
    cache.put("k", value)
    got = cache.get("k")
    assert got is not None
    assert np.array_equal(got.vector, value.vector)
    assert got.det_score == 0.9


def test_miss_then_hit_is_reflected_in_the_snapshot() -> None:
    cache = ReferenceEmbeddingCache(max_entries=8)
    assert cache.get("k") is None
    cache.put("k", _emb())
    assert cache.get("k") is not None

    snap = cache.snapshot()
    assert snap["hits"] == 1
    assert snap["misses"] == 1
    assert snap["hit_rate"] == 0.5
    assert snap["size"] == 1


def test_a_none_key_is_neither_hit_nor_miss() -> None:
    """Paths that do not key their reference must not distort the hit rate."""
    cache = ReferenceEmbeddingCache(max_entries=8)
    assert cache.get(None) is None
    cache.put(None, _emb())
    assert cache.snapshot() == {
        "enabled": True,
        "size": 0,
        "max_entries": 8,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "hit_rate": 0.0,
    }


def test_eviction_is_least_recently_used() -> None:
    cache = ReferenceEmbeddingCache(max_entries=2)
    cache.put("a", _emb(1))
    cache.put("b", _emb(2))
    assert cache.get("a") is not None  # 'a' is now the most recent, so 'b' is next out
    cache.put("c", _emb(3))

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None
    assert cache.snapshot()["evictions"] == 1
    assert cache.snapshot()["size"] == 2


def test_reinserting_a_key_does_not_grow_the_cache() -> None:
    cache = ReferenceEmbeddingCache(max_entries=2)
    cache.put("a", _emb(1))
    cache.put("a", _emb(2))
    assert cache.snapshot()["size"] == 1
    assert cache.snapshot()["evictions"] == 0


def test_a_no_face_reference_is_cached_too() -> None:
    """Otherwise an unusable enrolled image pays full detection on every retry."""
    cache = ReferenceEmbeddingCache(max_entries=4)
    cache.put("k", ReferenceEmbedding(vector=None, face_detected=False))
    got = cache.get("k")
    assert got is not None
    assert got.face_detected is False
    assert got.vector is None


def test_disabled_cache_never_stores() -> None:
    cache = ReferenceEmbeddingCache(max_entries=0)
    assert cache.enabled is False
    cache.put("k", _emb())
    assert cache.get("k") is None
    snap = cache.snapshot()
    assert snap["enabled"] is False
    assert snap["size"] == 0
    # A disabled cache reports no traffic rather than 100% misses.
    assert snap["misses"] == 0


def test_clear_drops_entries_but_keeps_counters() -> None:
    cache = ReferenceEmbeddingCache(max_entries=4)
    cache.put("k", _emb())
    assert cache.get("k") is not None
    cache.clear()
    assert cache.get("k") is None
    snap = cache.snapshot()
    assert snap["size"] == 0
    assert snap["hits"] == 1


def test_concurrent_access_keeps_the_counters_consistent() -> None:
    """The match path runs in a worker thread; several requests touch this at once."""
    from concurrent.futures import ThreadPoolExecutor

    cache = ReferenceEmbeddingCache(max_entries=64)
    keys = [f"k{i}" for i in range(32)]
    for k in keys:
        cache.put(k, _emb())

    def hammer() -> None:
        for _ in range(50):
            for k in keys:
                cache.get(k)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: hammer(), range(8)))

    snap = cache.snapshot()
    assert snap["hits"] == 8 * 50 * 32
    assert snap["misses"] == 0
    assert snap["size"] == 32
