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
