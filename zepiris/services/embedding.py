from __future__ import annotations

import base64
import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import httpx
import numpy as np

from zepiris.exceptions import (
    ImageEncodeError,
    MLInferenceTimeoutError,
    MLInferenceTransportError,
    MLInferenceUpstreamError,
)
from zepiris.schemas.ml_inference import FaceDetectionResult, FaceEmbeddingResult


def _wrap_ml_errors(call):
    """Run an ML-client call, converting httpx errors into ZepIris service errors.

    Without this, a failing ML request (e.g. the recognition model not loaded ->
    503) raises a raw ``httpx.HTTPStatusError`` that escapes as a bare HTTP 500
    "Internal Server Error". This surfaces a clear, structured message instead.
    """
    try:
        return call()
    except httpx.HTTPStatusError as e:
        detail: dict = {"message": "ml_inference_request_failed", "upstream_status": e.response.status_code}
        try:
            detail["upstream"] = e.response.json()
        except Exception:
            detail["upstream"] = e.response.text[:2000]
        status = 503 if e.response.status_code == 503 else 502
        raise MLInferenceUpstreamError(status_code=status, detail=detail) from e
    except httpx.TimeoutException as e:
        raise MLInferenceTimeoutError() from e
    except httpx.HTTPError as e:
        raise MLInferenceTransportError(str(e)) from e

if TYPE_CHECKING:
    from zepiris.services.ml_client import MLInferenceClient


class FaceEmbeddingProvider(ABC):
    """Produces a FaceEmbeddingResult containing the embedding vector and face-detection flag.

    The interface mirrors the ML inference microservice contract so swapping
    from the stub to the real service (or MLInferenceClient) is seamless.
    """

    @abstractmethod
    def embed(self, image_rgb: np.ndarray) -> FaceEmbeddingResult:
        raise NotImplementedError

    def detect_box(self, image_rgb: np.ndarray) -> FaceDetectionResult:
        """Detect the primary face and return its normalized bounding box.

        Default implementation derives the box from :meth:`embed` (no box info,
        so it reports a full-frame box when a face is found). Real providers
        override this with a cheaper detection-only call.
        """
        result = self.embed(image_rgb)
        return FaceDetectionResult(
            face_detected=result.face_detected,
            bbox=[0.0, 0.0, 1.0, 1.0] if result.face_detected else [0.0, 0.0, 0.0, 0.0],
            score=1.0 if result.face_detected else 0.0,
        )


class StubFaceEmbeddingService(FaceEmbeddingProvider):
    """Deterministic pseudo-embedding from image pixels (L2-normalized).

    Always reports face_detected=True since we cannot actually detect faces
    without a real model.  Replace with the ML inference service for production.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed(self, image_rgb: np.ndarray) -> FaceEmbeddingResult:
        payload = image_rgb.tobytes()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embedding = vec.astype(np.float32).tolist()
        return FaceEmbeddingResult(
            face_detected=True,
            embedding=embedding,
            embedding_dim=len(embedding),
        )


class MLInferenceEmbeddingService(FaceEmbeddingProvider):
    """Calls the ML inference microservice /v1/face/embed with a base64 PNG.

    Callers pass RGB arrays. The ML service decodes with OpenCV (BGR) and
    converts BGR->RGB, so the payload must be encoded from BGR for the colors
    to survive the round trip — encoding the RGB array directly would hand the
    recognition model channel-swapped images and degrade match scores.

    PNG (lossless) rather than JPEG: the input already survived one JPEG
    compression at capture, and a second lossy generation measurably blurs the
    high-frequency detail the recognition model keys on.
    """

    def __init__(self, client: MLInferenceClient) -> None:
        self._client = client

    @staticmethod
    def _rgb_to_png_b64(image_rgb: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ImageEncodeError()
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def embed(self, image_rgb: np.ndarray) -> FaceEmbeddingResult:
        image_b64 = self._rgb_to_png_b64(image_rgb)
        return _wrap_ml_errors(lambda: self._client.embed_face(image_b64))

    def detect_box(self, image_rgb: np.ndarray) -> FaceDetectionResult:
        image_b64 = self._rgb_to_png_b64(image_rgb)
        return _wrap_ml_errors(lambda: self._client.detect_face(image_b64))
