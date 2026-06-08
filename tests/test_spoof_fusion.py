"""SpoofDetectionService fuses the learned model with screen-replay detection.

The model's verdict is AND-gated with the passive detector: either signal can
mark an image as not-live.
"""

import numpy as np

from zepiris.ml_inference.moire_detection import ScreenReplayResult
from zepiris.ml_inference.spoof_detection import SpoofDetectionService


class _FakeScreenDetector:
    def __init__(self, is_screen: bool) -> None:
        self._is_screen = is_screen

    def analyze(self, image_rgb):  # noqa: ANN001
        return ScreenReplayResult(
            is_screen=self._is_screen,
            replay_score=0.9 if self._is_screen else 0.1,
            high_freq_ratio=0.0,
            peak_score=0.0,
            glare_ratio=0.0,
        )


def _service(detector) -> SpoofDetectionService:
    svc = SpoofDetectionService(
        huggingface_repo_id="unused",
        spoof_threshold=0.7,
        screen_replay_detector=detector,
    )
    # Stub the learned model: always reports highly "live" (prob_live=0.99).
    svc.load_model = lambda: None  # type: ignore[method-assign]
    svc.predict = lambda _pre: np.array([0.99, 0.01], dtype=np.float32)  # type: ignore[method-assign]
    return svc


def _img() -> np.ndarray:
    return np.full((64, 64, 3), 120, np.uint8)


def test_screen_forces_not_live_even_when_model_says_live():
    svc = _service(_FakeScreenDetector(is_screen=True))
    result = svc.forward(_img())
    assert result.is_live is False
    # Probability still reflects the model's live score.
    assert result.probability > 0.9


def test_genuine_passes_when_detector_clears():
    svc = _service(_FakeScreenDetector(is_screen=False))
    result = svc.forward(_img())
    assert result.is_live is True


def test_no_detector_falls_back_to_model_only():
    svc = _service(None)
    result = svc.forward(_img())
    assert result.is_live is True
