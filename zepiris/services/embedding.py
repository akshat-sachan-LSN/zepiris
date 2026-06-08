from __future__ import annotations

import base64
import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import numpy as np

from zepiris.exceptions import ImageEncodeError
from zepiris.schemas.ml_inference import FaceDetectionResult, FaceEmbeddingResult

if TYPE_CHECKING:
    from zepiris.services.ml_client import MLInferenceClient


class FaceEmbeddingProvider(ABC):
    """Produces a FaceEmbeddingResult containing the embedding vector and face-detection flag.

    The interface mirrors the ML inference microservice contract so swapping
    from the stub to the real service (or MLInferenceClient) is seamless.
    """

    @abstractmethod
    def embed(self, image_bgr: np.ndarray) -> FaceEmbeddingResult:
        raise NotImplementedError

    def detect_box(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        """Detect the primary face and return its normalized bounding box.

        Default implementation derives the box from :meth:`embed` (no box info,
        so it reports a full-frame box when a face is found). Real providers
        override this with a cheaper detection-only call.
        """
        result = self.embed(image_bgr)
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

    def embed(self, image_bgr: np.ndarray) -> FaceEmbeddingResult:
        payload = image_bgr.tobytes()
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
    """Calls the ML inference microservice /v1/face/embed with a base64 JPEG derived from BGR."""

    def __init__(self, client: MLInferenceClient) -> None:
        self._client = client

    def embed(self, image_bgr: np.ndarray) -> FaceEmbeddingResult:
        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise ImageEncodeError()
        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return self._client.embed_face(image_b64)

    def detect_box(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise ImageEncodeError()
        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return self._client.detect_face(image_b64)
