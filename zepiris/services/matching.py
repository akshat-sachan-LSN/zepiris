"""1:1 face matching, from raw image bytes to a similarity score.

The verification hot path has one job: turn two images into one number. Framing
it that way — rather than as "embed, embed, then compare" — is what lets the
expensive part happen in exactly one place, with nothing but bytes going in and
a float coming out.

Two implementations:

``RemoteFaceMatcher``
    Production. One HTTP call carrying both images as raw multipart bytes;
    the ML service decodes, embeds and scores, and returns the similarity.

``LocalFaceMatcher``
    Embeds both sides through a :class:`FaceEmbeddingProvider` in this process.
    Used when the API and the models are co-located (no HTTP hop at all), and by
    the test suite, where a stub provider stands in for the models.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import httpx
import numpy as np

from zepiris.exceptions import (
    MLInferenceTimeoutError,
    MLInferenceTransportError,
    MLInferenceUpstreamError,
    ReferenceImageDecodeError,
)
from zepiris.schemas.ml_inference import FaceMatchResult

if TYPE_CHECKING:
    from zepiris.services.embedding import FaceEmbeddingProvider
    from zepiris.services.ml_client import AsyncMLInferenceClient


class ProbeImageDecodeError(Exception):
    """The submitted image could not be decoded.

    Not a service error: the caller gets a normal 200 response reporting
    ``decodeFailed``, because an unreadable capture is an ordinary outcome of
    the verification flow rather than a fault.
    """


class FaceMatcher(ABC):
    """Scores one probe image against one reference image."""

    @abstractmethod
    async def match(
        self,
        probe_raw: bytes,
        reference_raw: bytes,
        *,
        want_probe_sharpness: bool = False,
    ) -> FaceMatchResult:
        """Return the similarity of the two images and what was detected in each.

        Raises:
            ProbeImageDecodeError: the probe image could not be decoded.
            ReferenceImageDecodeError: the reference image could not be decoded.
        """
        raise NotImplementedError


class RemoteFaceMatcher(FaceMatcher):
    """Scores a pair in one call to the ML inference service."""

    def __init__(self, client: AsyncMLInferenceClient) -> None:
        self._client = client

    async def match(
        self,
        probe_raw: bytes,
        reference_raw: bytes,
        *,
        want_probe_sharpness: bool = False,
    ) -> FaceMatchResult:
        try:
            return await self._client.match_faces(
                probe_raw, reference_raw, want_probe_sharpness=want_probe_sharpness
            )
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)
        except httpx.TimeoutException as e:
            raise MLInferenceTimeoutError() from e
        except httpx.HTTPError as e:
            raise MLInferenceTransportError(str(e)) from e

    @staticmethod
    def _raise_for_status(e: httpx.HTTPStatusError) -> None:
        """Translate an ML-service error into the matching ZepIris error.

        An undecodable image is reported per side (400 naming ``probe`` or
        ``reference``) so each maps onto the outcome the caller already expects,
        rather than collapsing into a generic upstream failure.
        """
        detail: dict = {
            "message": "ml_inference_request_failed",
            "upstream_status": e.response.status_code,
        }
        try:
            upstream = e.response.json()
        except Exception:
            upstream = e.response.text[:2000]
        detail["upstream"] = upstream

        if e.response.status_code == 400 and isinstance(upstream, dict):
            reason = str(upstream.get("detail", ""))
            if "probe" in reason:
                raise ProbeImageDecodeError(reason) from e
            if "reference" in reason:
                raise ReferenceImageDecodeError() from e

        status = 503 if e.response.status_code == 503 else 502
        raise MLInferenceUpstreamError(status_code=status, detail=detail) from e


class LocalFaceMatcher(FaceMatcher):
    """Scores a pair in-process through a :class:`FaceEmbeddingProvider`.

    Skips the HTTP hop entirely, so it is the lowest-latency arrangement when
    the API and the models run in the same container. The two embeds run
    sequentially: under real concurrency every core is already occupied by other
    requests, so splitting one request across threads costs contention without
    buying throughput.
    """

    def __init__(self, embedding: FaceEmbeddingProvider) -> None:
        self._embedding = embedding

    async def match(
        self,
        probe_raw: bytes,
        reference_raw: bytes,
        *,
        want_probe_sharpness: bool = False,
    ) -> FaceMatchResult:
        return await asyncio.to_thread(
            self._match_sync, probe_raw, reference_raw, want_probe_sharpness
        )

    def _match_sync(
        self, probe_raw: bytes, reference_raw: bytes, want_probe_sharpness: bool
    ) -> FaceMatchResult:
        probe_rgb = _decode_rgb(probe_raw)
        if probe_rgb is None:
            raise ProbeImageDecodeError("failed_to_decode_image: probe")

        probe = self._embedding.embed(probe_rgb)
        if not probe.face_detected:
            return FaceMatchResult(
                score=None,
                probe_face_detected=False,
                reference_face_detected=False,
                probe_face_sharpness=probe.face_sharpness if want_probe_sharpness else None,
            )

        reference_rgb = _decode_rgb(reference_raw)
        if reference_rgb is None:
            raise ReferenceImageDecodeError()

        reference = self._embedding.embed(reference_rgb)
        probe_sharp = probe.face_sharpness if want_probe_sharpness else None
        if not reference.face_detected:
            return FaceMatchResult(
                score=None,
                probe_face_detected=True,
                reference_face_detected=False,
                probe_det_score=probe.det_score,
                probe_face_sharpness=probe_sharp,
            )

        return FaceMatchResult(
            score=_cosine(probe.embedding, reference.embedding),
            probe_face_detected=True,
            reference_face_detected=True,
            probe_det_score=probe.det_score,
            reference_det_score=reference.det_score,
            probe_face_sharpness=probe_sharp,
        )


def _decode_rgb(raw: bytes) -> np.ndarray | None:
    image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero-length."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
