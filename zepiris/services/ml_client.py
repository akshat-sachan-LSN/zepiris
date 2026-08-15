"""HTTP client for calling the ML inference microservice.

The main application uses this client to send requests to the separate
ML inference container (running on port 8001 by default).

Example:
    import base64
    import cv2

    client = MLInferenceClient("http://localhost:8001")

    # Load image and convert to base64
    image_bgr = cv2.imread("photo.jpg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_bytes = cv2.imencode(".jpg", image_rgb)[1].tobytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    nsfw_result = client.detect_nsfw(image_b64)
    client.close()
"""

from __future__ import annotations

from typing import Any

import httpx

from zepiris.framing import encode_pair_frame
from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    FaceDetectionResult,
    FaceEmbeddingResult,
    FaceMatchResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)


def _upstream_error_detail(response: httpx.Response) -> dict[str, Any]:
    try:
        upstream: Any = response.json()
    except Exception:
        upstream = response.text[:2000]
    return {
        "message": "ml_inference_request_failed",
        "upstream_status": response.status_code,
        "upstream": upstream,
    }


class MLInferenceClient:
    """HTTP client for calling the remote ML inference service.

    Accepts images in base64 format and sends them to the inference service
    as JSON for simpler HTTP communication.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 60.0) -> None:
        """Initialize ML inference client.

        Args:
            base_url: Base URL of the ML inference service, e.g. "http://localhost:8001"
            timeout_seconds: Per-request timeout. CPU embedding with the detection
                fallback cascade can far exceed httpx's 5s default.
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)

    def _prepare_image_json(self, image_b64: str) -> dict:
        """Prepare JSON payload for an image in base64 format.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            dict: JSON payload with base64-encoded image
        """
        return {"image_b64": image_b64}

    def detect_nsfw(self, image_b64: str) -> NSFWDetectionResult:
        """Run NSFW detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            NSFWDetectionResult: Detection result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/nsfw_check", json=payload)
        response.raise_for_status()
        return NSFWDetectionResult(**response.json())

    def detect_spoof(self, image_b64: str) -> SpoofDetectionResult:
        """Run spoof detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            SpoofDetectionResult: Detection result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/spoof_check", json=payload)
        response.raise_for_status()
        return SpoofDetectionResult(**response.json())

    def detect_blur(self, image_b64: str) -> BlurDetectionResult:
        """Run blur detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            BlurDetectionResult: Detection result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/blur_check", json=payload)
        response.raise_for_status()
        return BlurDetectionResult(**response.json())

    def embed_face(self, image_b64: str) -> FaceEmbeddingResult:
        """Generate face embedding from an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            FaceEmbeddingResult: Embedding result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/face/embed", json=payload)
        response.raise_for_status()
        return FaceEmbeddingResult(**response.json())

    def detect_face(self, image_b64: str) -> FaceDetectionResult:
        """Detect the primary face and return its normalized bounding box.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            FaceDetectionResult: detection flag + normalized bbox + score
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/face/detect", json=payload)
        response.raise_for_status()
        return FaceDetectionResult(**response.json())

    def assess_image_quality(self, image_b64: str) -> ImageQualityAssessmentResult:
        """Run combined image quality assessment (NSFW + spoof + blur).

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            ImageQualityAssessmentResult: Combined assessment result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/assess", json=payload)
        response.raise_for_status()
        return ImageQualityAssessmentResult(**response.json())

    def healthz(self) -> dict[str, str]:
        """Check service health.

        Returns:
            dict: Health status response
        """
        return self._get_json("/healthz")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> MLInferenceClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()


class AsyncMLInferenceClient:
    """Async client for the hot verification path.

    The API process does no image work of its own — it fetches bytes and waits on
    the ML service — so it should hold requests on the event loop rather than
    parking a worker thread per in-flight call. At 100 concurrent verifications
    the sync client would need 100 threads to do nothing but block on a socket.

    Connection-pool limits are set explicitly: httpx defaults to 20 keep-alive
    connections, so beyond that every request pays a fresh TCP handshake and
    connections churn exactly when load is highest.
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 60.0,
        max_connections: int = 200,
        connect_timeout_seconds: float = 5.0,
    ) -> None:
        """
        Args:
            base_url: Base URL of the ML inference service.
            timeout_seconds: Read/write/pool timeout for a request.
            max_connections: Pool ceiling; keep-alive is held at the same value so
                a steady concurrent load reuses connections instead of churning.
            connect_timeout_seconds: Separate, shorter budget for establishing a
                connection — a dead upstream should fail fast, not consume the
                full inference timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                timeout_seconds, connect=connect_timeout_seconds
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
                keepalive_expiry=60.0,
            ),
        )

    async def match_faces(
        self,
        probe: bytes,
        reference: bytes,
        *,
        want_probe_sharpness: bool = False,
    ) -> FaceMatchResult:
        """Score a 1:1 pair, sending both images as one framed binary body.

        The images go over the wire exactly as they arrived — no base64, no
        re-encode — and only the similarity comes back. Framing rather than
        multipart keeps both sides in memory: multipart spools any part over 1 MB
        to a temporary file, which a typical phone photo exceeds.
        """
        response = await self.client.post(
            "/v1/face/match",
            content=encode_pair_frame(probe, reference),
            headers={"Content-Type": "application/octet-stream"},
            params={"want_probe_sharpness": str(want_probe_sharpness).lower()},
        )
        response.raise_for_status()
        return FaceMatchResult(**response.json())

    async def healthz(self) -> dict[str, str]:
        """Check service health."""
        response = await self.client.get("/healthz")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()
