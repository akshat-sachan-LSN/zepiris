"""MiniFASNet-V2 ONNX liveness service — preprocessing regression guards.

These lock in two hard-won, counter-intuitive facts about the
garciafido/minifasnet-v2 ONNX export (the model card documents both wrongly):
  * Input must be RAW pixels [0, 255] — the /255 scaling is baked into the
    graph; pre-dividing saturates the model and rejects real faces.
  * prob_live is class index 1 (real), not index 0.

The test runs the real ONNX model against a real face image (gallery/person-5)
so a regression in normalization or class index makes a genuine face read as a
spoof — exactly the failure we are guarding against.
"""

from pathlib import Path

import cv2
import pytest

from zepiris.ml_inference.onnx_spoof_detection import OnnxSpoofDetectionService
from zepiris.schemas.ml_inference import FaceDetectionResult

_MODEL = Path("models/minifasnet_v2_spoof.onnx")
_FACE = Path("gallery/person-5.jpg")

pytestmark = pytest.mark.skipif(
    not (_MODEL.exists() and _FACE.exists()),
    reason="requires local ONNX model and a sample face image",
)


def _centered_detector(_image_rgb):
    # Face occupies the central region of the gallery thumbnails.
    return FaceDetectionResult(face_detected=True, bbox=[0.2, 0.15, 0.8, 0.9], score=0.9)


def _service(threshold=0.5, screen=None):
    return OnnxSpoofDetectionService(
        models=[(str(_MODEL), 2.7)],
        face_detector=_centered_detector,
        live_threshold=threshold,
        screen_replay_detector=screen,
    )


def _real_face_rgb():
    bgr = cv2.imread(str(_FACE), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_real_face_scores_live():
    """A genuine face must read as live with high probability."""
    result = _service().forward(_real_face_rgb())
    assert result.is_live is True
    assert result.probability > 0.5


def test_crop_is_model_input_size():
    bgr = cv2.imread(str(_FACE), cv2.IMREAD_COLOR)
    crop = OnnxSpoofDetectionService._scaled_crop(bgr, (20, 15, 100, 130))
    assert crop.shape == (80, 80, 3)


_V2 = Path("models/minifasnet_v2_yakhyo.onnx")
_V1SE = Path("models/minifasnet_v1se_yakhyo.onnx")


@pytest.mark.skipif(
    not (_V2.exists() and _V1SE.exists() and _FACE.exists()),
    reason="requires both ensemble ONNX models",
)
def test_ensemble_keeps_real_face_live():
    """The two-model ensemble (V2 @2.7x + V1SE @4.0x) must keep a real face live."""
    svc = OnnxSpoofDetectionService(
        models=[(str(_V2), 2.7), (str(_V1SE), 4.0)],
        face_detector=_centered_detector,
        live_threshold=0.5,
    )
    result = svc.forward(_real_face_rgb())
    assert result.is_live is True
    assert result.probability > 0.5


def test_screen_gate_forces_not_live():
    """Even a high prob_live is overridden when the screen detector flags."""

    class _AlwaysScreen:
        def analyze(self, _img):
            from zepiris.ml_inference.moire_detection import ScreenReplayResult

            return ScreenReplayResult(True, 0.99, 0.0, 0.0, 0.0)

    result = _service(screen=_AlwaysScreen()).forward(_real_face_rgb())
    assert result.is_live is False
