"""Anti-spoofing / liveness via MiniFASNet-V2 (Silent-Face) ONNX model.

This is the stronger, purpose-built replacement for the MobileNetV3 spoof
classifier. It runs the proven MiniFASNet-V2 ``2.7_80x80`` model exported to
ONNX (loaded with ``onnxruntime`` — no PyTorch/TensorFlow at inference, and no
pickle, so it is safe to load).

Pipeline:
    1. Detect the primary face -> bounding box.
    2. Crop with a 2.7x scale margin around the bbox center.
    3. Resize to 80x80, BGR.
    4. HWC -> NCHW, run ONNX, softmax over 3 classes.
    5. prob_live = output[1] (class 1 = real, per MiniFASNet convention).

NOTE: this specific ONNX export folds the /255 scaling into its graph, so the
input must be RAW pixels in [0, 255] (NOT pre-divided). Feeding /255 saturates
the model to a constant output that rejects real faces. The class order is also
[attack, real, attack] (index 1 = live), not the [live, print, replay] the
model card states — both were verified empirically against real faces.

Model: garciafido/minifasnet-v2-anti-spoofing-onnx (Apache-2.0), an ONNX export
of minivision-ai/Silent-Face-Anti-Spoofing. See THIRD-PARTY-NOTICES.md.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import onnxruntime as ort

from zepiris.ml_inference.moire_detection import ScreenReplayDetector
from zepiris.schemas.ml_inference import FaceDetectionResult, SpoofDetectionResult

_INPUT_SIZE = 80
_CROP_SCALE = 2.7

# Callable that returns the primary face box for an RGB image.
FaceDetector = Callable[[np.ndarray], FaceDetectionResult]


class OnnxSpoofDetectionService:
    """MiniFASNet ONNX liveness detector (ensemble) with optional screen gate.

    Runs one or more MiniFASNet ONNX models and averages their live probability
    — the canonical Silent-Face approach, where V2 (2.7x crop) and V1SE (4.0x
    crop) are complementary. More models => lower variance, fewer single-model
    misclassifications.

    Args:
        models: List of ``(onnx_path, crop_scale)`` pairs to ensemble. Each
            model sees a crop at its own scale (V2: 2.7, V1SE: 4.0).
        face_detector: Callable returning a :class:`FaceDetectionResult` (with a
            normalized ``bbox``) for an RGB image. Reused from the face
            embedding service so we don't load a second detector.
        live_threshold: averaged prob_live must exceed this to be considered live.
        screen_replay_detector: Optional passive moiré/glare gate; when it flags
            a screen the result is forced not-live regardless of the models.
    """

    def __init__(
        self,
        models: list[tuple[str, float]],
        face_detector: FaceDetector,
        live_threshold: float = 0.5,
        screen_replay_detector: ScreenReplayDetector | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one ONNX model is required")
        self._sessions: list[tuple[ort.InferenceSession, str, float]] = []
        for path, scale in models:
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._sessions.append((session, session.get_inputs()[0].name, scale))
        self._face_detector = face_detector
        self._live_threshold = live_threshold
        self._screen = screen_replay_detector

    def load_model(self) -> None:
        """No-op; the ONNX sessions load in __init__. Kept for API symmetry."""
        return None

    def forward(self, image_rgb: np.ndarray) -> SpoofDetectionResult:
        """Detect the face, ensemble MiniFASNet over its crops, AND-gate with moiré.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8.

        Returns:
            SpoofDetectionResult: ``is_live`` reflects ensemble + screen gate;
                ``probability`` is the averaged live probability.
        """
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        bbox_px = self._face_box_px(image_rgb)

        probs = [
            self._infer_prob_live(session, input_name, self._scaled_crop(image_bgr, bbox_px, scale))
            for session, input_name, scale in self._sessions
        ]
        prob_live = float(np.mean(probs))
        is_live = prob_live > self._live_threshold

        if is_live and self._screen is not None and self._screen.analyze(image_rgb).is_screen:
            is_live = False

        return SpoofDetectionResult(is_live=is_live, probability=prob_live)

    # -- internals ----------------------------------------------------------

    def _face_box_px(self, image_rgb: np.ndarray) -> tuple[int, int, int, int]:
        """Return the primary face box in pixels, or the full frame if none."""
        h, w = image_rgb.shape[:2]
        result = self._face_detector(image_rgb)
        if not result.face_detected:
            return 0, 0, w, h
        x1, y1, x2, y2 = result.bbox  # normalized [0, 1]
        return (
            int(round(x1 * w)),
            int(round(y1 * h)),
            int(round(x2 * w)),
            int(round(y2 * h)),
        )

    @staticmethod
    def _scaled_crop(
        image_bgr: np.ndarray,
        bbox_px: tuple[int, int, int, int],
        crop_scale: float = _CROP_SCALE,
    ) -> np.ndarray:
        """Crop a ``crop_scale``x square margin around the face center -> 80x80.

        Mirrors Silent-Face's ``CropImage.get_new_box``: the box is enlarged by
        the scale (capped so it stays in-frame) and shifted to stay within image
        bounds, then resized to the model input size.
        """
        src_h, src_w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_px
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, crop_scale)
        new_w = box_w * scale
        new_h = box_h * scale
        cx = x1 + box_w / 2.0
        cy = y1 + box_h / 2.0

        left = cx - new_w / 2.0
        top = cy - new_h / 2.0
        right = cx + new_w / 2.0
        bottom = cy + new_h / 2.0

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > src_w - 1:
            left -= right - (src_w - 1)
            right = src_w - 1
        if bottom > src_h - 1:
            top -= bottom - (src_h - 1)
            bottom = src_h - 1

        crop = image_bgr[int(top) : int(bottom) + 1, int(left) : int(right) + 1]
        if crop.size == 0:
            crop = image_bgr
        return cv2.resize(crop, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _infer_prob_live(session, input_name: str, crop_bgr: np.ndarray) -> float:  # noqa: ANN001
        """Run one ONNX model on an 80x80 BGR crop and return prob_live.

        Input is RAW pixels [0, 255] (these exports bake in the /255 scaling).
        """
        blob = crop_bgr.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # HWC -> NCHW
        logits = session.run(None, {input_name: blob})[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        return float(probs[1])  # index 1 = real/live
