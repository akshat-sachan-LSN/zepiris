from __future__ import annotations

import time
from collections import OrderedDict

import httpx

from zepiris.exceptions import ReferenceImageFetchError


class _ReferenceBytesCache:
    """Byte-budget LRU of fetched reference images, keyed by URL.

    The enrolled selfie is fetched from S3 on *every* verification of that
    person, and it is the same bytes every time: production reference URLs are
    stable, unsigned object URLs whose keys are content-unique (uuid /
    user+timestamp), so a URL cannot quietly start serving different pixels.
    Caching the bytes here removes one of the two S3 round trips per request —
    and with it S3's latency tail, which otherwise lands directly in the
    verification path — while the probe (a fresh capture behind a fresh URL on
    every request) rightly never hits.

    Bounded by **bytes**, not entries, because the values are whole images:
    entry-count budgets look small until 4096 x 300 KB turns out to be 1.2 GB.
    The TTL is a safety valve for the one assumption above — an overwritten key
    — bounding how long a stale body could be served if a deployment breaks the
    content-unique naming.

    No lock: the fetcher is awaited only on the event loop, and neither ``get``
    nor ``put`` yields between check and mutate, so single-threaded execution
    is the synchronization.
    """

    def __init__(self, max_bytes: int, ttl_seconds: float) -> None:
        self._max_bytes = max(0, int(max_bytes))
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._total_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    @property
    def enabled(self) -> bool:
        return self._max_bytes > 0

    def get(self, url: str) -> bytes | None:
        if not self._max_bytes:
            return None
        entry = self._entries.get(url)
        if entry is None:
            self._misses += 1
            return None
        data, deadline = entry
        if time.monotonic() >= deadline:
            del self._entries[url]
            self._total_bytes -= len(data)
            self._expired += 1
            self._misses += 1
            return None
        # Refresh recency: eviction takes the references nobody is verifying
        # against any more.
        self._entries.move_to_end(url)
        self._hits += 1
        return data

    def put(self, url: str, data: bytes) -> None:
        if not self._max_bytes or len(data) > self._max_bytes:
            return
        old = self._entries.pop(url, None)
        if old is not None:
            self._total_bytes -= len(old[0])
        self._entries[url] = (data, time.monotonic() + self._ttl)
        self._total_bytes += len(data)
        while self._total_bytes > self._max_bytes:
            _, (evicted, _) = self._entries.popitem(last=False)
            self._total_bytes -= len(evicted)
            self._evictions += 1

    def snapshot(self) -> dict[str, float | int | bool]:
        """Hit rate and occupancy, for /metrics.

        The hit rate is the number worth watching: near zero means this traffic
        is first-time verifications and the memory is buying nothing — near one
        means half the S3 traffic (and its tail) is gone.
        """
        looked_up = self._hits + self._misses
        return {
            "enabled": self._max_bytes > 0,
            "entries": len(self._entries),
            "bytes": self._total_bytes,
            "max_bytes": self._max_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "expired": self._expired,
            "hit_rate": round(self._hits / looked_up, 3) if looked_up else 0.0,
        }


class S3ImageFetcher:
    """Fetch a reference image from a presigned/public URL via a guarded HTTP GET.

    No AWS credentials: the URL must be directly retrievable. Guards against
    slow responses (timeout on the client) and oversized payloads (max_bytes).

    Async because both images are fetched on every request: at high concurrency
    a blocking fetch would hold a worker thread per in-flight request purely to
    wait on a socket, and the thread pool — not S3 — would become the limit.

    ``cache_max_bytes`` > 0 enables an in-process bytes cache for URLs fetched
    with ``cacheable=True`` — see :class:`_ReferenceBytesCache` for why only the
    reference side qualifies. 0 (the default) keeps every fetch a real GET.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        max_bytes: int,
        *,
        cache_max_bytes: int = 0,
        cache_ttl_seconds: float = 900.0,
    ) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._cache = _ReferenceBytesCache(cache_max_bytes, cache_ttl_seconds)

    async def fetch(self, url: str, *, cacheable: bool = False) -> bytes:
        if cacheable:
            cached = self._cache.get(url)
            if cached is not None:
                return cached

        try:
            response = await self._client.get(url)
        except httpx.TimeoutException as exc:
            raise ReferenceImageFetchError(reason="timeout", detail_msg=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ReferenceImageFetchError(reason="transport_error", detail_msg=str(exc)) from exc

        if response.status_code != 200:
            raise ReferenceImageFetchError(
                reason="bad_status", detail_msg=f"status_{response.status_code}"
            )

        data = response.content
        if len(data) > self._max_bytes:
            raise ReferenceImageFetchError(
                reason="too_large",
                detail_msg=f"{len(data)}_bytes_max_{self._max_bytes}",
            )
        if not data:
            raise ReferenceImageFetchError(reason="empty", detail_msg="empty_body")

        if cacheable:
            self._cache.put(url, data)
        return data

    def cache_snapshot(self) -> dict[str, float | int | bool]:
        """Cache saturation for the API's /metrics endpoint."""
        return self._cache.snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
