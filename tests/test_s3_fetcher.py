import asyncio

import httpx
import pytest

from zepiris.exceptions import ReferenceImageFetchError
from zepiris.services.s3_fetcher import S3ImageFetcher


def _fetcher(handler, *, max_bytes: int = 1024) -> S3ImageFetcher:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return S3ImageFetcher(client=client, max_bytes=max_bytes)


def _fetch(fetcher: S3ImageFetcher, url: str) -> bytes:
    """Drive the async fetch from a sync test."""
    return asyncio.run(fetcher.fetch(url))


def test_fetch_returns_bytes_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"imagebytes")

    assert _fetch(_fetcher(handler), "https://s3/ref.jpg") == b"imagebytes"


def test_fetch_raises_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"nope")

    with pytest.raises(ReferenceImageFetchError):
        _fetch(_fetcher(handler), "https://s3/missing.jpg")


def test_fetch_raises_on_oversize() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000)

    with pytest.raises(ReferenceImageFetchError):
        _fetch(_fetcher(handler, max_bytes=1024), "https://s3/big.jpg")


def test_fetch_raises_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    with pytest.raises(ReferenceImageFetchError):
        _fetch(_fetcher(handler), "https://s3/slow.jpg")


def test_fetches_are_concurrent() -> None:
    """Both image fetches must overlap — they are the only network wait on the
    request path, and serializing them would double it."""
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, content=b"img")

    fetcher = _fetcher(handler)

    async def both() -> None:
        await asyncio.gather(
            fetcher.fetch("https://s3/a.jpg"), fetcher.fetch("https://s3/b.jpg")
        )

    asyncio.run(both())
    assert peak == 2


def test_cacheable_fetch_hits_after_first_get() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"refbytes")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    fetcher = S3ImageFetcher(client=client, max_bytes=1024, cache_max_bytes=64 * 1024)

    async def twice() -> tuple[bytes, bytes]:
        first = await fetcher.fetch("https://s3/ref.jpg", cacheable=True)
        second = await fetcher.fetch("https://s3/ref.jpg", cacheable=True)
        return first, second

    first, second = asyncio.run(twice())
    assert first == second == b"refbytes"
    assert calls == 1
    snap = fetcher.cache_snapshot()
    assert snap["hits"] == 1 and snap["misses"] == 1


def test_uncacheable_fetch_never_caches() -> None:
    """The probe side is a fresh capture every time; it must not occupy the
    budget or be served from a previous request."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"probe")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    fetcher = S3ImageFetcher(client=client, max_bytes=1024, cache_max_bytes=64 * 1024)

    async def twice() -> None:
        await fetcher.fetch("https://s3/probe.jpg")
        await fetcher.fetch("https://s3/probe.jpg")

    asyncio.run(twice())
    assert calls == 2
    assert fetcher.cache_snapshot()["entries"] == 0


def test_cache_disabled_by_default() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"refbytes")

    fetcher = _fetcher(handler)
    _fetch(fetcher, "https://s3/ref.jpg")
    assert _fetch(fetcher, "https://s3/ref.jpg") == b"refbytes"
    assert calls == 2


def test_cache_evicts_by_byte_budget_lru_first() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=request.url.path.encode().ljust(40, b"x"))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    # Budget fits two 40-byte bodies, not three.
    fetcher = S3ImageFetcher(client=client, max_bytes=1024, cache_max_bytes=100)

    async def run() -> None:
        await fetcher.fetch("https://s3/a", cacheable=True)
        await fetcher.fetch("https://s3/b", cacheable=True)
        await fetcher.fetch("https://s3/a", cacheable=True)  # refresh a's recency
        await fetcher.fetch("https://s3/c", cacheable=True)  # evicts b, the LRU
        # a and c should hit; b should miss and refetch.
        before = fetcher.cache_snapshot()["misses"]
        await fetcher.fetch("https://s3/a", cacheable=True)
        await fetcher.fetch("https://s3/c", cacheable=True)
        await fetcher.fetch("https://s3/b", cacheable=True)
        after = fetcher.cache_snapshot()["misses"]
        assert after - before == 1

    asyncio.run(run())
    snap = fetcher.cache_snapshot()
    assert snap["evictions"] >= 1
    assert snap["bytes"] <= 100


def test_cache_expires_after_ttl() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"refbytes")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    fetcher = S3ImageFetcher(
        client=client, max_bytes=1024, cache_max_bytes=64 * 1024, cache_ttl_seconds=0.0
    )

    async def twice() -> None:
        await fetcher.fetch("https://s3/ref.jpg", cacheable=True)
        await fetcher.fetch("https://s3/ref.jpg", cacheable=True)

    asyncio.run(twice())
    assert calls == 2
    assert fetcher.cache_snapshot()["expired"] == 1


def test_failed_fetch_is_not_cached() -> None:
    responses = [httpx.Response(500), httpx.Response(200, content=b"good")]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    fetcher = S3ImageFetcher(client=client, max_bytes=1024, cache_max_bytes=64 * 1024)

    with pytest.raises(ReferenceImageFetchError):
        asyncio.run(fetcher.fetch("https://s3/ref.jpg", cacheable=True))
    assert asyncio.run(fetcher.fetch("https://s3/ref.jpg", cacheable=True)) == b"good"
