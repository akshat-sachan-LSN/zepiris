"""Face embedding service using InsightFace (antelopev2 / buffalo_l)."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.ml_inference.embedding_cache import (
    ReferenceEmbedding,
    ReferenceEmbeddingCache,
)
from zepiris.ml_inference.face_engine import DEFAULT_TIER, EngineConfig, build_engine
from zepiris.schemas.ml_inference import (
    FaceDetectionResult,
    FaceEmbeddingResult,
    FaceMatchResult,
)

logger = logging.getLogger(__name__)


def _resolve_reference(
    reference: np.ndarray | Callable[[], np.ndarray] | None,
) -> np.ndarray:
    """Materialize a reference image that may have been supplied lazily.

    The match route passes a callable so the reference is decoded only when it is
    actually needed — a cache hit never decodes those bytes at all. Resolving
    here rather than at the call site means the decode happens on whichever
    thread does the embedding, so the parallel path overlaps it with the probe.
    """
    if reference is None:
        raise ValueError(
            "a reference image is required on a cache miss; pass an array or a "
            "callable that decodes one"
        )
    return reference() if callable(reference) else reference


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
        detection_size: tuple[int, int] = (512, 512),
        facial_area_threshold: float = 0.01,
        device: str = "cpu",
        enable_padding_retry: bool = True,
        padding_fraction: float = 0.25,
        model_name: str = "buffalo_l",
        det_thresh: float = 0.5,
        low_det_thresh: float = 0.3,
        bbox_bounds_tolerance: float = 0.05,
        enable_upscale_retry: bool = True,
        upscale_factor: float = 2.0,
        upscale_max_side: int = 2000,
        enable_flip_tta: bool = False,
        tier: str = DEFAULT_TIER,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        max_input_side: int = 0,
        enable_det_cache: bool = False,
        reference_cache_size: int = 0,
        parallel_embed_workers: int = 4,
        det_model_path: str | None = None,
        rec_model_path: str | None = None,
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
            bbox_bounds_tolerance: How far a detection may extend past the image edge
                before it is rejected as invalid, as a fraction of the image
                width/height. A fully-visible face is bounded by the frame, so a box
                spilling well past an edge is a hallucinated/partial detection; the
                low-confidence fallback occasionally returns such a box (e.g. latching
                onto a chin/neck region on a frame-filling selfie) which embeds as
                garbage and silently tanks the match score. Rejecting it lets the
                padding/upscale retries recover the real face instead.
            enable_upscale_retry: If True, when no face is found, retry on an upscaled
                copy of the image. Small document faces detect far better when enlarged.
            upscale_factor: How much to enlarge on the upscale retry.
            upscale_max_side: Cap the longer side after upscaling (avoid huge images).
            enable_flip_tta: If True, average the embedding of the aligned face with
                that of its horizontal mirror (test-time augmentation). A standard
                ArcFace trick that measurably improves robustness on low-quality /
                blurry inputs (e.g. printed document photos) at the cost of one extra
                recognition pass; impostor scores are unaffected.
            tier: Detector/recognizer pairing — see
                :mod:`zepiris.ml_inference.face_engine`. ``"accurate"`` reproduces
                stock buffalo_l; ``"balanced"`` keeps the same recognition weights
                behind a much cheaper detector; ``"fast"`` also swaps in
                MobileFaceNet recognition (needs threshold recalibration).
            intra_op_threads: ONNX Runtime threads within a single inference. 1 —
                the default — serves each request on one core and takes
                parallelism from request concurrency instead, which is what
                scales. See :class:`~zepiris.ml_inference.face_engine.EngineConfig`.
            inter_op_threads: ONNX Runtime threads across graph branches.
            max_input_side: Downscale inputs whose longer side exceeds this before
                detection (0 = never). The detector resizes to ``detection_size``
                internally regardless, so feeding it a 12 MP phone capture only
                pays for a bigger resize; the recognition crop is unaffected at
                any sane cap. 1600 is a safe production value.
            enable_det_cache: Memoize detector output keyed by image content. Only
                pays off when the same pixels are detected twice in one request,
                which was true when the liveness gate ran before the embed. With
                liveness off there is no second pass, so the cache is off by
                default — hashing a multi-megapixel frame costs more than it saves.
            reference_cache_size: Entries in the reference-embedding cache
                (0 = off). The enrolled selfie is the same bytes on every
                verification of that person, so its embedding — roughly half the
                cost of a match — is recomputed for nothing. Keyed by image
                content, so it cannot go stale. See
                :mod:`zepiris.ml_inference.embedding_cache`.
            parallel_embed_workers: Size of the helper pool used when a caller
                asks for the two sides to be embedded concurrently. Only used on
                a cache miss, and only when the caller knows there is spare CPU
                — see ``match_pair(parallel=...)``.
            det_model_path: Explicit path to a detector ``.onnx`` file. When set
                together with ``rec_model_path`` it overrides the tier's default
                detector, allowing FP16-converted models to be dropped in without
                changing the tier. Both must be provided or neither takes effect.
            rec_model_path: Explicit path to a recognizer ``.onnx`` file. See
                ``det_model_path``.
        """
        self._embedding_dim = embedding_dim
        self._detection_size = detection_size
        self._facial_area_threshold = facial_area_threshold
        self._enable_padding_retry = enable_padding_retry
        self._padding_fraction = padding_fraction
        self._model_name = model_name
        self._det_thresh = det_thresh
        self._low_det_thresh = low_det_thresh
        self._bbox_bounds_tolerance = bbox_bounds_tolerance
        self._enable_upscale_retry = enable_upscale_retry
        self._upscale_factor = upscale_factor
        self._upscale_max_side = upscale_max_side
        self._enable_flip_tta = enable_flip_tta
        self._tier = tier
        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._max_input_side = max_input_side
        self._reference_cache = ReferenceEmbeddingCache(reference_cache_size)
        self._parallel_embed_workers = max(1, int(parallel_embed_workers))
        self._det_model_path = det_model_path or None
        self._rec_model_path = rec_model_path or None
        self._embed_pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._face_app: FaceAnalysis | None = None
        self._load_lock = threading.Lock()
        # The detector's confidence threshold is instance state on the shared
        # det_model, so a per-call override has to be set and restored around the
        # call. Serialize that window: concurrent requests would otherwise
        # interleave set/restore and detect at each other's thresholds.
        self._det_thresh_lock = threading.Lock()
        # Short-lived memo of the detector's raw output, keyed by image content +
        # threshold. Worth it only when one request detects the same pixels twice,
        # which was the case while the liveness gate cropped the probe before the
        # embed ran. That gate is off, so this defaults to disabled: hashing a
        # multi-megapixel frame costs more than the detection it would save.
        # Bounded LRU; nothing is persisted beyond the last few requests.
        self._enable_det_cache = enable_det_cache
        self._det_cache: OrderedDict[bytes, tuple] = OrderedDict()
        self._det_cache_lock = threading.Lock()
        self._det_cache_max = 16
        config = ModelServiceConfig(
            model_name="face_embedding",
            device=device,
        )
        super().__init__(config)

    def load_model(self):
        """Build (once) the detector + recognizer pair this service runs on.

        Prefers :func:`~zepiris.ml_inference.face_engine.build_engine`, which
        pins the ONNX Runtime thread pools and can pair a cheap detector with
        the accurate recognizer. Falls back to stock ``FaceAnalysis`` if that
        fails for any reason, so a bad tier or a missing pack degrades to the
        previous behaviour instead of taking the service down.

        Returns:
            An object exposing ``.det_model`` and ``.models["recognition"]``.
        """
        if self._face_app is not None:
            return self._face_app

        with self._load_lock:
            if self._face_app is not None:
                return self._face_app
            try:
                self._face_app = build_engine(
                    EngineConfig(
                        tier=self._tier,
                        det_size=self._detection_size,
                        det_thresh=self._det_thresh,
                        intra_op_threads=self._intra_op_threads,
                        inter_op_threads=self._inter_op_threads,
                        device=self.config.device,
                        det_model_path=self._det_model_path,
                        rec_model_path=self._rec_model_path,
                    )
                )
                return self._face_app
            except Exception:
                logger.warning(
                    "Tuned face engine (tier=%r) failed to build; falling back to "
                    "stock FaceAnalysis(%r)",
                    self._tier,
                    self._model_name,
                    exc_info=True,
                )
            self._face_app = self._load_face_analysis()
            return self._face_app

    def _load_face_analysis(self) -> FaceAnalysis:
        """Stock InsightFace loading path (fallback)."""
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

        return app

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

    def _run_detect(self, app, image_rgb: np.ndarray, det_thresh: float | None) -> tuple:
        """Invoke the detector, honouring a one-call threshold override.

        The override is applied by mutating ``det_model.det_thresh`` around the
        call — InsightFace exposes no per-call threshold — so it is serialized
        under a lock. Without it, two concurrent requests can interleave the
        set/restore and one of them silently detects at the other's threshold.
        """
        det_model = app.det_model
        if det_thresh is None or not hasattr(det_model, "det_thresh"):
            return det_model.detect(image_rgb, max_num=0, metric="default")
        with self._det_thresh_lock:
            prev = det_model.det_thresh
            det_model.det_thresh = det_thresh
            try:
                return det_model.detect(image_rgb, max_num=0, metric="default")
            finally:
                det_model.det_thresh = prev

    def _detect(self, image_rgb: np.ndarray, det_thresh: float | None) -> tuple:
        """Run the detector, optionally memoizing per (image content, threshold).

        ``app.det_model.detect`` is deterministic in (image, det_thresh, det_size),
        so when one request detects the same pixels more than once the memo saves
        a full pass. It is off by default — see ``enable_det_cache``; hashing a
        multi-megapixel frame costs more than it saves on the single-pass path.
        Returns ``(bboxes, kpss)``.
        """
        app = self.load_model()
        if not self._enable_det_cache:
            return self._run_detect(app, image_rgb, det_thresh)

        eff_thresh = (
            det_thresh if det_thresh is not None else getattr(app.det_model, "det_thresh", None)
        )
        digest = hashlib.blake2b(np.ascontiguousarray(image_rgb), digest_size=16).digest()
        key = b"%s|%r|%r" % (digest, eff_thresh, image_rgb.shape)

        with self._det_cache_lock:
            cached = self._det_cache.get(key)
            if cached is not None:
                self._det_cache.move_to_end(key)
                return cached

        result = self._run_detect(app, image_rgb, det_thresh)

        with self._det_cache_lock:
            self._det_cache[key] = result
            self._det_cache.move_to_end(key)
            while len(self._det_cache) > self._det_cache_max:
                self._det_cache.popitem(last=False)
        return result

    def _cap_input(self, image_rgb: np.ndarray) -> np.ndarray:
        """Downscale an oversized frame before detection.

        The detector letterboxes to ``detection_size`` internally, so a 12 MP
        phone capture buys no extra detection accuracy — it only makes that
        resize more expensive, and every later retry pass too. Capping the longer
        side keeps the face far above the 112 px the recognizer needs.
        """
        cap = self._max_input_side
        if cap <= 0:
            return image_rgb
        h, w = image_rgb.shape[:2]
        longer = max(h, w)
        if longer <= cap:
            return image_rgb
        scale = cap / float(longer)
        return cv2.resize(
            image_rgb,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

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
        bboxes, kpss = self._detect(image_rgb, det_thresh)

        if len(bboxes) == 0:
            return None

        h, w = image_rgb.shape[:2]
        img_area = h * w

        # Reject detections that extend substantially beyond the frame. A fully
        # visible face is bounded by the image, so a box spilling well past an edge
        # is a hallucinated/partial detection (typically a weak low-threshold hit on
        # a chin/neck region of a frame-filling selfie). Embedding it yields a
        # near-random vector that silently drives the match score negative; dropping
        # it returns None so preprocess() falls through to the padding/upscale
        # retries that recover the real face.
        tol_x = w * self._bbox_bounds_tolerance
        tol_y = h * self._bbox_bounds_tolerance
        in_bounds = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[:4]
            if x1 >= -tol_x and y1 >= -tol_y and x2 <= w + tol_x and y2 <= h + tol_y:
                in_bounds.append(i)

        if not in_bounds:
            return None

        filtered_indices = []
        for i in in_bounds:
            x1, y1, x2, y2 = bboxes[i][:4]
            face_area = (x2 - x1) * (y2 - y1)
            if face_area / img_area > self._facial_area_threshold:
                filtered_indices.append(i)

        candidates = filtered_indices if filtered_indices else in_bounds

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
            image_rgb,
            (int(round(w * factor)), int(round(h * factor))),
            interpolation=cv2.INTER_CUBIC,
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
        image_rgb = self._cap_input(image_rgb)

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
        want_sharpness = preprocessed_data.get("want_sharpness", True)

        if not self._enable_flip_tta:
            embedding = rec.get(image, face)
        else:
            # Flip test-time augmentation: align the face once (keypoint-driven warp),
            # then average the embedding of the aligned crop and its horizontal mirror.
            # Summing here is fine — postprocess() L2-normalizes the result.
            aligned = face_align.norm_crop(image, landmark=face.kps, image_size=rec.input_size[0])
            embedding = (
                rec.get_feat(aligned).flatten() + rec.get_feat(cv2.flip(aligned, 1)).flatten()
            )

        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        # Only the document path reads sharpness; a face match would pay for a
        # full-resolution Laplacian it never looks at.
        sharpness = self._face_region_sharpness(image, face) if want_sharpness else None
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
        image_rgb = self._cap_input(image_rgb)
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

    def embed(self, image_rgb: np.ndarray, *, want_sharpness: bool = True) -> FaceEmbeddingResult:
        """Generate face embedding from image.

        Args:
            image_rgb: Input face image in RGB format, shape (H, W, 3), dtype uint8
            want_sharpness: Compute the face-region focus metric. Only the
                document path uses it; face match passes False to skip a
                full-resolution Laplacian it would not read.

        Returns:
            FaceEmbeddingResult: L2-normalized embedding vector with face detection status and metadata
        """
        preprocessed = self.preprocess(image_rgb)
        preprocessed["want_sharpness"] = want_sharpness
        return self.postprocess(self.predict(preprocessed))

    @property
    def reference_cache(self) -> ReferenceEmbeddingCache:
        """The reference-embedding cache, for keying at the call site and /metrics."""
        return self._reference_cache

    def _pool(self) -> ThreadPoolExecutor:
        """Helper threads for concurrent embedding, created on first use.

        Built lazily so a deployment that never asks for parallel embedding never
        carries the threads. ONNX Runtime releases the GIL for the duration of a
        Run call, so these threads genuinely execute on separate cores.
        """
        if self._embed_pool is None:
            with self._pool_lock:
                if self._embed_pool is None:
                    self._embed_pool = ThreadPoolExecutor(
                        max_workers=self._parallel_embed_workers,
                        thread_name_prefix="ref-embed",
                    )
        return self._embed_pool

    def embed_reference(self, reference_rgb: np.ndarray) -> ReferenceEmbedding:
        """Embed the reference side into a cacheable value."""
        vector, found, det_score = self.embed_vector(reference_rgb)
        if not found:
            return ReferenceEmbedding(vector=None, face_detected=False)
        return ReferenceEmbedding(vector=vector, face_detected=True, det_score=det_score)

    def match_pair(
        self,
        probe_rgb: np.ndarray,
        reference_rgb: np.ndarray | Callable[[], np.ndarray] | None,
        *,
        want_probe_sharpness: bool = False,
        reference_key: str | None = None,
        parallel: bool = False,
    ) -> FaceMatchResult:
        """Embed both sides and score them, all within this process.

        Doing the comparison here rather than in the caller keeps two 512-float
        vectors off the wire and out of JSON on the hot path.

        ``reference_key`` — a digest of the reference *bytes* from
        :func:`~zepiris.ml_inference.embedding_cache.reference_digest` — turns the
        reference side into a cache lookup. The enrolled selfie does not change
        between verifications, so on a hit this method does half the work: one
        embed instead of two. Pass None to bypass the cache.

        ``reference_rgb`` accepts either a decoded array or a **callable that
        returns one**. The callable form is what the match route passes, so the
        reference bytes are decoded only on a miss — on a hit that decode (a full
        JPEG, several milliseconds) never happens. It has to be a callable rather
        than the caller peeking at the cache first: between a peek and this
        method's own lookup the entry can be evicted, and a caller that had
        already decided not to decode would then have nothing to embed.

        ``parallel`` embeds the two sides concurrently instead of one after the
        other, which roughly halves the latency of a cache miss (measured 349 ms
        -> 177 ms on 8 cores). It is opt-in per call, and the caller is expected
        to ask only when there is spare CPU: at high concurrency every core is
        already busy with other requests, so splitting one request across threads
        buys latency for one caller by taking throughput from everyone. It also
        gives up the no-face short-circuit below — the reference is embedded
        before the probe's verdict is known — which is free only while cores are
        idle.

        A probe with no detectable face short-circuits: there is nothing to
        compare it against, so the reference is never embedded.
        """
        cached = self._reference_cache.get(reference_key)

        if cached is None and parallel:
            # Start the reference before the probe's verdict is known: with idle
            # cores that overlap is free, and it is the only way the two embeds
            # can run at once.
            pending = self._pool().submit(
                lambda: self.embed_reference(_resolve_reference(reference_rgb))
            )
            probe_vec, probe_found, probe_det, probe_sharp = self._embed_probe(
                probe_rgb, want_probe_sharpness
            )
            reference = pending.result()
            self._reference_cache.put(reference_key, reference)
            if not probe_found:
                return FaceMatchResult(
                    score=None,
                    probe_face_detected=False,
                    reference_face_detected=False,
                    probe_face_sharpness=probe_sharp,
                )
        else:
            probe_vec, probe_found, probe_det, probe_sharp = self._embed_probe(
                probe_rgb, want_probe_sharpness
            )
            if not probe_found:
                return FaceMatchResult(
                    score=None,
                    probe_face_detected=False,
                    reference_face_detected=False,
                    probe_face_sharpness=probe_sharp,
                )
            reference = cached
            if reference is None:
                reference = self.embed_reference(_resolve_reference(reference_rgb))
                self._reference_cache.put(reference_key, reference)

        if not reference.face_detected:
            return FaceMatchResult(
                score=None,
                probe_face_detected=True,
                reference_face_detected=False,
                probe_det_score=probe_det,
                probe_face_sharpness=probe_sharp,
            )

        probe_norm = float(np.linalg.norm(probe_vec))
        if probe_norm > 0:
            probe_vec = probe_vec / probe_norm

        return FaceMatchResult(
            score=float(np.dot(probe_vec, reference.vector)),
            probe_face_detected=True,
            reference_face_detected=True,
            probe_det_score=probe_det,
            reference_det_score=reference.det_score,
            probe_face_sharpness=probe_sharp,
        )

    def _embed_probe(
        self, probe_rgb: np.ndarray, want_sharpness: bool
    ) -> tuple[np.ndarray, bool, float | None, float | None]:
        """Detect and embed the probe side, keeping its sharpness if asked for."""
        preprocessed = self.preprocess(probe_rgb)
        preprocessed["want_sharpness"] = want_sharpness
        return self.predict(preprocessed)

    def embed_vector(self, image_rgb: np.ndarray) -> tuple[np.ndarray, bool, float | None]:
        """Embed and return the raw vector, skipping the JSON-facing result model.

        The match path compares two embeddings numerically and never serializes
        them, so building a 512-element Python float list per side (and
        validating it through Pydantic) is pure overhead. Returns
        ``(l2_normalized_vector, face_detected, det_score)``.
        """
        preprocessed = self.preprocess(image_rgb)
        preprocessed["want_sharpness"] = False
        embedding, face_detected, det_score, _ = self.predict(preprocessed)
        if not face_detected:
            return embedding, False, None
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding, True, det_score
