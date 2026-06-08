import httpx
import pytest

from zepiris.exceptions import ReferenceImageFetchError
from zepiris.services.s3_fetcher import S3ImageFetcher


def _fetcher(handler, *, max_bytes: int = 1024) -> S3ImageFetcher:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, timeout=5.0)
    return S3ImageFetcher(client=client, max_bytes=max_bytes)


def test_fetch_returns_bytes_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"imagebytes")

    fetcher = _fetcher(handler)
    assert fetcher.fetch("https://s3/ref.jpg") == b"imagebytes"


def test_fetch_raises_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"nope")

    fetcher = _fetcher(handler)
    with pytest.raises(ReferenceImageFetchError):
        fetcher.fetch("https://s3/missing.jpg")


def test_fetch_raises_on_oversize() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000)

    fetcher = _fetcher(handler, max_bytes=1024)
    with pytest.raises(ReferenceImageFetchError):
        fetcher.fetch("https://s3/big.jpg")


def test_fetch_raises_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    fetcher = _fetcher(handler)
    with pytest.raises(ReferenceImageFetchError):
        fetcher.fetch("https://s3/slow.jpg")
