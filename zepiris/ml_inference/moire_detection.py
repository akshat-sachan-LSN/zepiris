"""Passive screen-replay (moiré / glare) detection using classical CV.

A photo *of a phone or monitor screen* carries artifacts that a genuine,
directly-captured face does not:

* **Moiré patterns** — interference between the display's pixel grid and the
  camera sensor's grid shows up as elevated, often periodic, high-frequency
  energy in the 2D Fourier spectrum.
* **Specular glare** — emissive displays produce bright, near-clipped, smooth
  highlight regions (screen backlight reflections).

This module fuses those two signals into a single ``replay_score`` in
``[0, 1]``.  It uses only ``cv2`` + ``numpy`` (already project dependencies);
no model download is required.

It is a *heuristic* second line of defence layered on top of the learned
spoof classifier — it raises the bar against typical phone/monitor replays,
but no single still-image check can guarantee liveness against a high-quality
print or 4K display.  Thresholds are tunable so the operating point can be
tightened per deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Work on a fixed-size grayscale image so the frequency-domain metrics are
# comparable regardless of the input resolution.
_ANALYSIS_SIZE = (512, 512)

# Fraction of the (centered) spectrum radius treated as "low frequency".
# Energy beyond this radius is the high-frequency band where screen/moiré
# artifacts concentrate.
_LOW_FREQ_RADIUS_FRAC = 0.15


@dataclass(frozen=True)
class ScreenReplayResult:
    """Outcome of screen-replay analysis.

    Attributes:
        is_screen: True if the image looks like a recaptured screen.
        replay_score: Fused screen-likeness score in [0, 1] (higher = more
            likely a replay).
        high_freq_ratio: Share of spectral energy in the high-frequency band.
        peak_score: Strength of periodic (moiré) peaks in [0, 1].
        glare_ratio: Fraction of the image that is bright, smooth, near-clipped
            highlight (screen backlight glare) in [0, 1].
    """

    is_screen: bool
    replay_score: float
    high_freq_ratio: float
    peak_score: float
    glare_ratio: float


class ScreenReplayDetector:
    """Detect recaptured-screen images from moiré and glare cues.

    Args:
        threshold: ``replay_score`` above which the image is flagged as a
            screen (default 0.5).
        high_freq_weight: Weight of the high-frequency-energy cue in the fused
            score.
        peak_weight: Weight of the periodic-peak (moiré) cue.
        glare_weight: Weight of the specular-glare cue.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        high_freq_weight: float = 0.20,
        peak_weight: float = 0.65,
        glare_weight: float = 0.15,
    ) -> None:
        self.threshold = threshold
        total = high_freq_weight + peak_weight + glare_weight
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self._w_hf = high_freq_weight / total
        self._w_peak = peak_weight / total
        self._w_glare = glare_weight / total

    def analyze(self, image_rgb: np.ndarray) -> ScreenReplayResult:
        """Compute screen-replay cues for an RGB image.

        Args:
            image_rgb: Input image, shape (H, W, 3), dtype uint8, RGB order.

        Returns:
            ScreenReplayResult: fused score and component cues.
        """
        gray = self._to_analysis_gray(image_rgb)

        high_freq_ratio, peak_score = self._spectral_cues(gray)
        glare_ratio = self._glare_cue(gray)

        replay_score = float(
            self._w_hf * _saturate(high_freq_ratio, 0.35)
            + self._w_peak * peak_score
            + self._w_glare * _saturate(glare_ratio, 0.04)
        )
        replay_score = max(0.0, min(1.0, replay_score))

        return ScreenReplayResult(
            is_screen=replay_score > self.threshold,
            replay_score=replay_score,
            high_freq_ratio=float(high_freq_ratio),
            peak_score=float(peak_score),
            glare_ratio=float(glare_ratio),
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _to_analysis_gray(image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, _ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _spectral_cues(gray: np.ndarray) -> tuple[float, float]:
        """Return (high_freq_ratio, peak_score) from the FFT magnitude spectrum.

        ``high_freq_ratio`` is the share of spectral energy beyond a central
        low-frequency disc.

        ``peak_score`` measures *periodicity* — the hallmark of a recaptured
        screen. We isolate sharp spectral peaks by subtracting a blurred copy
        of the log-magnitude spectrum from itself: broadband natural texture
        blurs to nearly itself (small residual), while a regular pixel/moiré
        grid leaves tall isolated spikes (large residual). The score is the
        fraction of high-band energy carried by those peaks.
        """
        f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        magnitude = np.abs(f)
        power = magnitude**2

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_radius = np.sqrt(cy**2 + cx**2)
        low_freq_mask = radius <= (_LOW_FREQ_RADIUS_FRAC * max_radius)
        high_mask = ~low_freq_mask

        total_power = float(power.sum()) + 1e-9
        high_band = power.copy()
        high_band[low_freq_mask] = 0.0
        high_freq_ratio = float(high_band.sum()) / total_power

        # Peakiness via residual against a smoothed spectrum (top-hat style).
        log_mag = np.log1p(magnitude).astype(np.float32)
        smoothed = cv2.GaussianBlur(log_mag, (0, 0), sigmaX=3.0)
        residual = np.clip(log_mag - smoothed, 0.0, None)
        residual[low_freq_mask] = 0.0

        high_energy = float(log_mag[high_mask].sum()) + 1e-9
        peak_energy = float(residual.sum())
        peak_score = _saturate(peak_energy / high_energy, 0.12)

        # Periodicity is meaningless on a near-flat image: with almost no
        # high-frequency content the ratio above explodes on noise. Gate it by
        # the actual high-frequency presence so smooth/blurred real faces score
        # ~0 instead of false-flagging as a screen.
        hf_gate = _saturate(high_freq_ratio, 0.05)
        peak_score *= hf_gate

        return high_freq_ratio, peak_score

    @staticmethod
    def _glare_cue(gray: np.ndarray) -> float:
        """Fraction of pixels that are bright, near-clipped, and locally smooth.

        Screen backlight glare is bright (high value) and flat (low local
        gradient); a genuine scene's bright pixels usually carry texture.
        """
        bright = gray >= 245
        if not bright.any():
            return 0.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        smooth = grad_mag < 15.0
        glare = bright & smooth
        return float(glare.sum()) / float(gray.size)


def _saturate(value: float, scale: float) -> float:
    """Linearly map ``value / scale`` into [0, 1] (clamped)."""
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / scale))
