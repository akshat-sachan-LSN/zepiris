"""Face embedding service using InsightFace (antelopev2 / buffalo_l)."""

from __future__ import annotations

import logging
import os
import shutil

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.schemas.ml_inference import FaceDetectionResult, FaceEmbeddingResult

logger = logging.getLogger(__name__)


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
        model_name: str = "buffalo_l",
        det_thresh: float = 0.5,
        low_det_thresh: float = 0.3,
        enable_upscale_retry: bool = True,
        upscale_factor: float = 2.0,
        upscale_max_side: int = 2000,
        enable_flip_tta: bool = True,
    ) -> None:
        """Initialize face embedding service.

        Args:
            embedding_dim: Expected embedding dimension (512 for buffalo_l/antelopev2)
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
            model_name: InsightFace model pack. "antelopev2" (glintr100/ResNet100,
                Glint360K) is more accurate than "buffalo_l" (w600k_r50/ResNet50);
                both produce 512-d embeddings. Slower per call on CPU.
            det_thresh: Primary detector confidence (0.5 = clean, well-aligned crops).
            low_det_thresh: Fallback confidence used only when the primary pass finds
                no face — recovers small/printed/low-contrast faces (e.g. the photo on
                an Aadhaar/PAN document) at the cost of accepting weaker detections.
            enable_upscale_retry: If True, when no face is found, retry on an upscaled
                copy of the image. Small document faces detect far better when enlarged.
            upscale_factor: How much to enlarge on the upscale retry.
            upscale_max_side: Cap the longer side after upscaling (avoid huge images).
            enable_flip_tta: If True, average the embedding of the aligned face with
                that of its horizontal mirror (test-time augmentation). A standard
                ArcFace trick that measurably improves robustness on low-quality /
                blurry inputs (e.g. printed document photos) at the cost of one extra
                recognition pass; impostor scores are unaffected.
        """
        self._embedding_dim = embedding_dim
        self._detection_size = detection_size
        self._facial_area_threshold = facial_area_threshold
        self._enable_padding_retry = enable_padding_retry
        self._padding_fraction = padding_fraction
        self._model_name = model_name
        self._det_thresh = det_thresh
        self._low_det_thresh = low_det_thresh
        self._enable_upscale_retry = enable_upscale_retry
        self._upscale_factor = upscale_factor
        self._upscale_max_side = upscale_max_side
        self._enable_flip_tta = enable_flip_tta
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

        def _prepare(name: str) -> FaceAnalysis:
            app = FaceAnalysis(name=name)
            app.prepare(ctx_id=ctx_id, det_size=self._detection_size, det_thresh=self._det_thresh)
            app.models = {k: v for k, v in app.models.items() if k in ("detection", "recognition")}
            return app

        try:
            app = _prepare(self._model_name)
        except Exception:
            if self._model_name == "buffalo_l":
                raise
            # A partial/corrupt download (e.g. an interrupted antelopev2) leaves the
            # model folder present but incomplete, so InsightFace skips re-downloading
            # and then asserts 'detection' missing. Wipe the cache and retry a CLEAN
            # download once; only fall back to buffalo_l if that also fails.
            logger.warning(
                "Face model %r failed to load; clearing its cache and re-downloading",
                self._model_name,
                exc_info=True,
            )
            self._clear_model_cache(self._model_name)
            try:
                app = _prepare(self._model_name)
            except Exception:
                logger.warning(
                    "Face model %r still failed after a clean download; falling back to buffalo_l",
                    self._model_name,
                    exc_info=True,
                )
                app = _prepare("buffalo_l")

        self._face_app = app
        return self._face_app

    @staticmethod
    def _clear_model_cache(name: str) -> None:
        """Remove a model pack's cached folder + zip so it re-downloads cleanly.

        Mirrors InsightFace's cache location (``$INSIGHTFACE_HOME`` or
        ``~/.insightface``); safe to call even if nothing is there.
        """
        root = os.environ.get(
            "INSIGHTFACE_HOME", os.path.join(os.path.expanduser("~"), ".insightface")
        )
        models_dir = os.path.join(root, "models")
        shutil.rmtree(os.path.join(models_dir, name), ignore_errors=True)
        try:
            os.remove(os.path.join(models_dir, f"{name}.zip"))
        except OSError:
            pass

    def _select_face(self, image_rgb: np.ndarray, det_thresh: float | None = None) -> Face | None:
        """Detect faces and select the best one for recognition.

        Picks the **largest** qualifying face (most pixels → best alignment and
        embedding quality), breaking ties by detector confidence. This beats a
        "most central" rule: the dominant face is almost always the subject, and
        a bigger crop yields a more discriminative embedding (higher match scores).

        ``det_thresh`` temporarily overrides the detector confidence for this call
        (used by the fallback cascade to recover hard/document faces).

        Returns the chosen ``Face`` (detection + keypoints) or ``None`` when the
        detector finds nothing.
        """
        app = self.load_model()

        if det_thresh is not None and hasattr(app.det_model, "det_thresh"):
            prev = app.det_model.det_thresh
            app.det_model.det_thresh = det_thresh
            try:
                bboxes, kpss = app.det_model.detect(image_rgb, max_num=0, metric="default")
            finally:
                app.det_model.det_thresh = prev
        else:
            bboxes, kpss = app.det_model.detect(image_rgb, max_num=0, metric="default")

        if len(bboxes) == 0:
            return None

        h, w = image_rgb.shape[:2]
        img_area = h * w

        filtered_indices = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[:4]
            face_area = (x2 - x1) * (y2 - y1)
            if face_area / img_area > self._facial_area_threshold:
                filtered_indices.append(i)

        candidates = filtered_indices if filtered_indices else list(range(len(bboxes)))

        def _rank(i: int) -> tuple[float, float]:
            x1, y1, x2, y2 = bboxes[i][:4]
            area = float((x2 - x1) * (y2 - y1))
            score = float(bboxes[i][4]) if bboxes.shape[1] > 4 else 0.0
            return (area, score)

        selected_idx = 0
        best_rank = (-1.0, -1.0)
        for i in candidates:
            rank = _rank(i)
            if rank > best_rank:
                best_rank = rank
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

    def _upscale_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Enlarge the image so small/printed faces (e.g. on documents) detect better."""
        h, w = image_rgb.shape[:2]
        factor = self._upscale_factor
        longer = max(h, w) * factor
        if longer > self._upscale_max_side:  # don't blow up huge scans
            factor = self._upscale_max_side / float(max(h, w))
        if factor <= 1.0:
            return image_rgb
        return cv2.resize(
            image_rgb, (int(round(w * factor)), int(round(h * factor))), interpolation=cv2.INTER_CUBIC
        )

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Detect a face with a fallback cascade, returning the image to embed on.

        Tries progressively harder so small/printed faces (documents) are recovered
        while clean selfies still resolve on the first, highest-quality pass:

          1. original image @ primary det_thresh        (best alignment, selfies)
          2. original image @ low det_thresh             (weak/low-contrast faces)
          3. reflect-padded image @ low det_thresh       (frame-filling/edge faces)
          4. upscaled image @ low det_thresh             (small document photos)

        Recognition runs on whichever image produced the detection, so the face
        bbox/keypoints stay in the right coordinate space.

        Returns:
            dict: {"image": image used for recognition, "face": Face or None}
        """
        # 1. primary pass — clean, well-aligned (covers normal selfies/photos)
        face = self._select_face(image_rgb)
        if face is not None:
            return {"image": image_rgb, "face": face}

        # 2. lower the confidence threshold on the original
        if self._low_det_thresh < self._det_thresh:
            face = self._select_face(image_rgb, det_thresh=self._low_det_thresh)
            if face is not None:
                return {"image": image_rgb, "face": face}

        # 3. reflect-pad (restores margin for frame-filling / border-touching faces)
        if self._enable_padding_retry:
            padded = self._pad_image(image_rgb)
            if padded is not image_rgb:
                face = self._select_face(padded, det_thresh=self._low_det_thresh)
                if face is not None:
                    return {"image": padded, "face": face}

        # 4. upscale (small printed document faces detect far better enlarged)
        if self._enable_upscale_retry:
            upscaled = self._upscale_image(image_rgb)
            if upscaled is not image_rgb:
                face = self._select_face(upscaled, det_thresh=self._low_det_thresh)
                if face is not None:
                    return {"image": upscaled, "face": face}

        return {"image": image_rgb, "face": None}

    @staticmethod
    def _face_region_sharpness(image_rgb: np.ndarray, face: Face) -> float | None:
        """Variance-of-Laplacian over the detected face box — a cheap focus metric.

        Measured on the native-resolution face region the recognizer saw, so the
        document path can flag a blurry capture without a separate detection pass.
        """
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in face.bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        region = image_rgb[y1:y2, x1:x2]
        if region.size == 0:
            return None
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    def predict(self, preprocessed_data: dict) -> tuple:
        """Extract embedding for the detected face using the recognition model.

        Args:
            preprocessed_data: dict with "image" and "face" keys from preprocess()

        Returns:
            tuple: (embedding (512,), face_detected, det_score, face_sharpness).
        """
        face = preprocessed_data["face"]
        if face is None:
            return np.zeros(self._embedding_dim, dtype=np.float32), False, None, None

        app = self.load_model()
        rec = app.models["recognition"]
        image = preprocessed_data["image"]

        if not self._enable_flip_tta:
            embedding = rec.get(image, face)
        else:
            # Flip test-time augmentation: align the face once (keypoint-driven warp),
            # then average the embedding of the aligned crop and its horizontal mirror.
            # Summing here is fine — postprocess() L2-normalizes the result.
            aligned = face_align.norm_crop(image, landmark=face.kps, image_size=rec.input_size[0])
            embedding = rec.get_feat(aligned).flatten() + rec.get_feat(cv2.flip(aligned, 1)).flatten()

        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        sharpness = self._face_region_sharpness(image, face)
        return np.asarray(embedding, dtype=np.float32), True, det_score, sharpness

    def postprocess(self, output: tuple) -> FaceEmbeddingResult:
        """L2-normalize embedding and wrap in result schema.

        Args:
            output: (embedding, face_detected, det_score, face_sharpness) from predict()

        Returns:
            FaceEmbeddingResult: Normalized embedding with detection status and metadata
        """
        embedding, face_detected, det_score, sharpness = output

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        embedding_list = embedding.astype(np.float32).tolist()
        return FaceEmbeddingResult(
            face_detected=face_detected,
            embedding=embedding_list,
            embedding_dim=len(embedding_list),
            det_score=det_score,
            face_sharpness=sharpness,
        )

    def detect_box(self, image_rgb: np.ndarray) -> FaceDetectionResult:
        """Detect the primary face and return its normalized bounding box.

        Detection only (no recognition), so it is cheap enough to poll. Used by
        the UI readiness ring and by document face extraction.

        Falls back to a lower confidence threshold and then an upscaled copy
        when the primary pass finds nothing — small printed document faces need
        both. Uniform upscaling preserves normalized coordinates, so the
        returned box always maps directly onto the original frame. The reflect
        padding retry used by :meth:`preprocess` is deliberately NOT used here:
        padding shifts coordinates and the box would no longer line up.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            FaceDetectionResult: detection flag, normalized [x1, y1, x2, y2], score
        """
        detected = image_rgb
        face = self._select_face(image_rgb)
        if face is None and self._low_det_thresh < self._det_thresh:
            face = self._select_face(image_rgb, det_thresh=self._low_det_thresh)
        if face is None and self._enable_upscale_retry:
            upscaled = self._upscale_image(image_rgb)
            if upscaled is not image_rgb:
                face = self._select_face(upscaled, det_thresh=self._low_det_thresh)
                detected = upscaled
        if face is None:
            return FaceDetectionResult(face_detected=False, bbox=[0.0, 0.0, 0.0, 0.0])

        h, w = detected.shape[:2]
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
