"""Content-addressed cache for reference embeddings.

A 1:1 verification embeds two faces, and half that work is work the service has
already done. The probe is a fresh capture every time — new pixels, nothing to
reuse. The reference is the *enrolled* selfie: the same bytes come back on every
verification of that person, and re-embedding them costs ~165 ms of a ~350 ms
request (measured, CPU ``balanced`` tier) to reproduce a vector that cannot have
changed. Caching it halves the latency of a repeat verification and doubles what
one instance can serve.

Keys are a digest of the reference bytes, which makes staleness impossible
rather than merely unlikely: different bytes are a different key, and identical
bytes through the same model give an identical embedding. There is no TTL to
tune and no invalidation to get wrong — a re-enrolled selfie is new bytes, so it
simply misses. The one thing the key does *not* cover is the model itself, so a
process serving a new tier must not inherit an old process's entries; since the
cache lives in memory and dies with the process, it cannot.

A miss is also cached when the reference has no detectable face. Without that, a
caller retrying against an unusable enrolled image pays full detection every
time — the expensive path, repeated, for an answer that will not change.

512 float32s is 2 KB, so the default 4096 entries costs ~8 MB — a rounding error
against the 1–2 GB the models occupy.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

#: Digest length in bytes. 16 bytes (128 bits) makes an accidental collision
#: unreachable at any cache size this process will hold, at half the key memory
#: of a full-length hash. This is a cache key, not a security boundary.
_DIGEST_BYTES = 16


def reference_digest(raw: bytes) -> str:
    """Key for a reference image's bytes.

    BLAKE2b rather than SHA-256 because it is faster on the sizes involved (a
    ~150 KB JPEG hashes in well under a millisecond, against the ~165 ms embed
    the hit avoids) and the digest length is a parameter rather than a truncation.
    """
    return hashlib.blake2b(raw, digest_size=_DIGEST_BYTES).hexdigest()


@dataclass(frozen=True)
class ReferenceEmbedding:
    """An embedded reference side, ready to score a probe against.

    Attributes:
        vector: L2-normalized 512-d embedding, or None when no face was found.
        face_detected: Whether a face was found in the reference image.
        det_score: Detector confidence for the reference face, when there was one.
    """

    vector: np.ndarray | None
    face_detected: bool
    det_score: float | None = None


class ReferenceEmbeddingCache:
    """Bounded LRU of reference embeddings, keyed by image-content digest.

    Thread-safe: the match path runs in a worker thread, so several requests
    touch this concurrently. The lock is held only for dict bookkeeping, never
    across an inference, so it is not a contention point.

    ``max_entries <= 0`` disables the cache while keeping every call site
    branch-free — ``get`` always misses and ``put`` does nothing.
    """

    def __init__(self, max_entries: int) -> None:
        self._max = max(0, int(max_entries))
        self._entries: OrderedDict[str, ReferenceEmbedding] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self._max > 0

    @property
    def max_entries(self) -> int:
        return self._max

    def get(self, key: str | None) -> ReferenceEmbedding | None:
        """Return the cached embedding for ``key``, or None on a miss.

        A ``None`` key means the caller has nothing to look up (cache disabled,
        or a path that does not key its reference); it counts as neither a hit
        nor a miss, so the hit rate stays a statement about cacheable traffic.
        """
        if not self._max or key is None:
            return None
        with self._lock:
            found = self._entries.get(key)
            if found is None:
                self._misses += 1
                return None
            # Refresh recency: this key is now the newest, so eviction takes the
            # references nobody is verifying against any more.
            self._entries.move_to_end(key)
            self._hits += 1
            return found

    def put(self, key: str | None, value: ReferenceEmbedding) -> None:
        """Store an embedding, evicting the least recently used entry if full."""
        if not self._max or key is None:
            return
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = value
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """Drop every entry, keeping the counters (they describe the process)."""
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> dict[str, float | int | bool]:
        """Hit rate and occupancy, for /metrics.

        The hit rate is the number worth watching: the cache only pays off when
        the same reference recurs, so a rate near zero means this traffic is
        first-time verifications and the memory is buying nothing.
        """
        with self._lock:
            looked_up = self._hits + self._misses
            return {
                "enabled": self._max > 0,
                "size": len(self._entries),
                "max_entries": self._max,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / looked_up, 3) if looked_up else 0.0,
            }
