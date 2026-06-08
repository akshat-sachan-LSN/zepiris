"""Face embedding service using InsightFace (buffalo_l)."""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.schemas.ml_inference import FaceDetectionResult, FaceEmbeddingResult


class FaceEmbeddingService(ModelService):
    """Face embedding service using InsightFace buffalo_l model.

    Uses InsightFace FaceAnalysis for detection + recognition.
    Selects the most central face that exceeds a minimum area threshold.
    Returns a 512-d normalized embedding vector.
    InsightFace handles model downloading via its own default mechanism.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        detection_size: tuple[int, int] = (640, 640),
        facial_area_threshold: float = 0.01,
        device: str = "cpu",
        enable_padding_retry: bool = True,
        padding_fraction: float = 0.25,
    ) -> None:
        """Initialize face embedding service.

        Args:
            embedding_dim: Expected embedding dimension (default 512 for buffalo_l)
            detection_size: Face detection size as (width, height) tuple
            facial_area_threshold: Minimum face area as fraction of image area
            device: Inference device ("cpu" or "cuda")
            enable_padding_retry: If True, retry detection on a reflect-padded copy
                of the image when no face is found on the first pass. Tightly cropped
                faces that fill the frame or touch a border are frequently missed by
                the detector because it has no margin/context to work with; padding
                restores that margin.
            padding_fraction: Border width added on each side during the retry,
                expressed as a fraction of the image's larger dimension.
        """
        self._embedding_dim = embedding_dim
        self._detection_size = detection_size
        self._facial_area_threshold = facial_area_threshold
        self._enable_padding_retry = enable_padding_retry
        self._padding_fraction = padding_fraction
        self._face_app: FaceAnalysis | None = None
        config = ModelServiceConfig(
            model_name="face_embedding",
            device=device,
        )
        super().__init__(config)

    def load_model(self) -> FaceAnalysis:
        """Initialize InsightFace FaceAnalysis with buffalo_l model.

        Returns:
            FaceAnalysis: Prepared model with detection + recognition only
        """
        if self._face_app is not None:
            return self._face_app

        ctx_id = 0 if self.config.device != "cpu" else -1
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=self._detection_size)
        app.models = {k: v for k, v in app.models.items() if k in ("detection", "recognition")}

        self._face_app = app
        return self._face_app

    def _select_face(self, image_rgb: np.ndarray) -> Face | None:
        """Detect faces in an image and select the most central qualifying one.

        Returns the chosen ``Face`` (detection + keypoints) or ``None`` when the
        detector finds nothing.
        """
        app = self.load_model()

        bboxes, kpss = app.det_model.detect(image_rgb, max_num=0, metric="default")

        if len(bboxes) == 0:
            return None

        h, w = image_rgb.shape[:2]
        img_area = h * w
        img_center = np.array([w / 2.0, h / 2.0])

        filtered_indices = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[:4]
            face_area = (x2 - x1) * (y2 - y1)
            if face_area / img_area > self._facial_area_threshold:
                filtered_indices.append(i)

        candidates = filtered_indices if filtered_indices else list(range(len(bboxes)))

        selected_idx = 0
        min_dist = float("inf")
        for i in candidates:
            x1, y1, x2, y2 = bboxes[i][:4]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            dist = np.linalg.norm(center - img_center)
            if dist < min_dist:
                min_dist = dist
                selected_idx = i

        return Face(
            bbox=bboxes[selected_idx, :4],
            kps=kpss[selected_idx],
            det_score=bboxes[selected_idx, 4],
        )

    def _pad_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Add a reflective border so a frame-filling face regains surrounding margin."""
        h, w = image_rgb.shape[:2]
        pad = int(round(max(h, w) * self._padding_fraction))
        if pad <= 0:
            return image_rgb
        return np.pad(
            image_rgb,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="reflect",
        )

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Detect faces and select the most central face above area threshold.

        If the first detection pass finds nothing and ``enable_padding_retry`` is
        set, the image is reflect-padded and detection is retried. Recognition then
        runs on the padded image so the embedding keeps full facial context.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            dict: {"image": image used for recognition, "face": Face or None}
        """
        face = self._select_face(image_rgb)
        if face is not None:
            return {"image": image_rgb, "face": face}

        if self._enable_padding_retry:
            padded = self._pad_image(image_rgb)
            if padded is not image_rgb:
                face = self._select_face(padded)
                if face is not None:
                    return {"image": padded, "face": face}

        return {"image": image_rgb, "face": None}

    def predict(self, preprocessed_data: dict) -> tuple[np.ndarray, bool]:
        """Extract embedding for the detected face using the recognition model.

        Args:
            preprocessed_data: dict with "image" and "face" keys from preprocess()

        Returns:
            tuple[np.ndarray, bool]: Face embedding (shape (512,), dtype float32)
                                     and face_detected flag (True if face was found)
        """
        face = preprocessed_data["face"]
        if face is None:
            return np.zeros(self._embedding_dim, dtype=np.float32), False

        app = self.load_model()
        embedding = app.models["recognition"].get(preprocessed_data["image"], face)

        return np.asarray(embedding, dtype=np.float32), True

    def postprocess(self, output: tuple[np.ndarray, bool]) -> FaceEmbeddingResult:
        """L2-normalize embedding and wrap in result schema.

        Args:
            output: Tuple of (embedding, face_detected) from predict()

        Returns:
            FaceEmbeddingResult: Normalized embedding with face detection status and metadata
        """
        embedding, face_detected = output

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        embedding_list = embedding.astype(np.float32).tolist()
        return FaceEmbeddingResult(
            face_detected=face_detected,
            embedding=embedding_list,
            embedding_dim=len(embedding_list),
        )

    def detect_box(self, image_rgb: np.ndarray) -> FaceDetectionResult:
        """Detect the primary face and return its normalized bounding box.

        Detection only (no recognition), so it is cheap enough to poll. Runs on
        the original frame (no padding retry) so the returned box maps directly
        to the image shown to the user. Used by the UI readiness ring to decide
        whether the face is positioned inside the on-screen circle.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            FaceDetectionResult: detection flag, normalized [x1, y1, x2, y2], score
        """
        face = self._select_face(image_rgb)
        if face is None:
            return FaceDetectionResult(face_detected=False, bbox=[0.0, 0.0, 0.0, 0.0])

        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])
        bbox = [
            max(0.0, min(1.0, x1 / w)),
            max(0.0, min(1.0, y1 / h)),
            max(0.0, min(1.0, x2 / w)),
            max(0.0, min(1.0, y2 / h)),
        ]
        score = float(getattr(face, "det_score", 0.0) or 0.0)
        return FaceDetectionResult(face_detected=True, bbox=bbox, score=score)

    def embed(self, image_rgb: np.ndarray) -> FaceEmbeddingResult:
        """Generate face embedding from image.

        Convenience alias for ``forward()``.

        Args:
            image_rgb: Input face image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            FaceEmbeddingResult: L2-normalized embedding vector with face detection status and metadata
        """
        return self.forward(image_rgb)
