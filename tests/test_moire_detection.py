"""Screen-replay (moiré/glare) detector behavior.

Two properties matter most:
  * It must NOT flag natural / smooth / textured real-face inputs (false
    positives lock out genuine users).
  * It SHOULD flag strong periodic screen patterns.
"""

import cv2
import numpy as np

from zepiris.ml_inference.moire_detection import ScreenReplayDetector


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _grid() -> np.ndarray:
    xx = np.meshgrid(np.arange(512), np.arange(512))[0]
    return (127 + 127 * np.cos(xx * 2.6)).astype(np.uint8)


def test_smooth_gradient_not_flagged():
    det = ScreenReplayDetector(threshold=0.5)
    g = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    assert det.analyze(_rgb(g)).is_screen is False


def test_flat_image_not_flagged():
    det = ScreenReplayDetector(threshold=0.5)
    flat = np.full((512, 512, 3), 120, np.uint8)
    assert det.analyze(flat).is_screen is False


def test_textured_noise_not_flagged():
    """A detailed (high-frequency but non-periodic) real photo must pass."""
    det = ScreenReplayDetector(threshold=0.5)
    rng = np.random.default_rng(0)
    n = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    assert det.analyze(_rgb(n)).is_screen is False


def test_blurred_face_not_flagged():
    det = ScreenReplayDetector(threshold=0.5)
    rng = np.random.default_rng(1)
    soft = cv2.GaussianBlur(rng.integers(0, 256, (512, 512), dtype=np.uint8), (0, 0), 5)
    assert det.analyze(_rgb(soft)).is_screen is False


def test_strong_periodic_pattern_flagged():
    det = ScreenReplayDetector(threshold=0.5)
    result = det.analyze(_rgb(_grid()))
    assert result.is_screen is True
    assert result.replay_score > 0.5


def test_threshold_is_tunable():
    """A stricter threshold flags borderline content; a lax one does not."""
    img = _rgb(_grid())
    strict = ScreenReplayDetector(threshold=0.1).analyze(img)
    lax = ScreenReplayDetector(threshold=0.99).analyze(img)
    assert strict.is_screen is True
    assert lax.is_screen is False
    # Same image, same score regardless of threshold.
    assert abs(strict.replay_score - lax.replay_score) < 1e-6


def test_score_bounds():
    det = ScreenReplayDetector(threshold=0.5)
    for img in (np.zeros((512, 512, 3), np.uint8), _rgb(_grid())):
        r = det.analyze(img)
        assert 0.0 <= r.replay_score <= 1.0
