from __future__ import annotations

import httpx

from zepiris.exceptions import ReferenceImageFetchError


class S3ImageFetcher:
    """Fetch a reference image from a presigned/public URL via a guarded HTTP GET.

    No AWS credentials: the URL must be directly retrievable. Guards against
    slow responses (timeout on the client) and oversized payloads (max_bytes).

    Async because both images are fetched on every request: at high concurrency
    a blocking fetch would hold a worker thread per in-flight request purely to
    wait on a socket, and the thread pool — not S3 — would become the limit.
    """

    def __init__(self, client: httpx.AsyncClient, max_bytes: int) -> None:
        self._client = client
        self._max_bytes = max_bytes

    async def fetch(self, url: str) -> bytes:
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
        return data

    async def aclose(self) -> None:
        await self._client.aclose()
