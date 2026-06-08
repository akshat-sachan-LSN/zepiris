#!/usr/bin/env python3
"""Diagnose liveness scores for one or more images — for threshold calibration.

Runs the SAME pipeline the API uses (color-correct spoof model + passive
screen-replay detector) directly against image files, so you can see exactly
what your real-face selfie and your phone-screen photo score, then tune the
two knobs to separate them:

    ML_SERVICE_SPOOF_THRESHOLD          (prob_live must exceed this)
    ML_SERVICE_SPOOF_SCREEN_THRESHOLD   (replay_score above this = screen)

Usage:
    python scripts/diagnose_liveness.py live_selfie.jpg phone_screen.jpg
    python scripts/diagnose_liveness.py --model models/spoof_model.pth img1.jpg img2.jpg

No network access and no running services required — it loads your local
spoof_model.pth directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from zepiris.ml_inference.moire_detection import ScreenReplayDetector
from zepiris.ml_inference.spoof_detection import SpoofDetectionService

# Defaults mirror MLServiceSettings so the diagnosis matches production.
DEFAULT_SPOOF_THRESHOLD = 0.7
DEFAULT_SCREEN_THRESHOLD = 0.5


def _load_rgb(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose liveness scores per image.")
    parser.add_argument("images", nargs="+", help="image file paths to analyze")
    parser.add_argument("--model", default="models/spoof_model.pth", help="local spoof model .pth")
    parser.add_argument("--spoof-threshold", type=float, default=DEFAULT_SPOOF_THRESHOLD)
    parser.add_argument("--screen-threshold", type=float, default=DEFAULT_SCREEN_THRESHOLD)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        return 2

    spoof = SpoofDetectionService(
        huggingface_repo_id="",
        local_model_path=str(model_path),
        model_source="local",
        spoof_threshold=args.spoof_threshold,
        screen_replay_detector=None,  # run detector separately so we can show sub-scores
    )
    spoof.load_model()
    screen = ScreenReplayDetector(threshold=args.screen_threshold)

    print(
        f"\nThresholds: prob_live > {args.spoof_threshold} AND replay_score <= "
        f"{args.screen_threshold} => LIVE\n"
    )
    header = f"{'image':<28} {'prob_live':>9} {'replay':>7} {'hf':>6} {'peak':>6} {'glare':>6}  verdict"
    print(header)
    print("-" * len(header))

    rows = []
    for img_path in args.images:
        path = Path(img_path)
        rgb = _load_rgb(path)
        if rgb is None:
            print(f"{path.name:<28} {'(could not read image)':>40}")
            continue

        model_result = spoof.forward(rgb)  # detector is None -> pure model verdict
        prob_live = model_result.probability
        s = screen.analyze(rgb)

        is_live = (prob_live > args.spoof_threshold) and (not s.is_screen)
        verdict = "LIVE" if is_live else "SPOOF"
        reason = ""
        if not is_live:
            reasons = []
            if prob_live <= args.spoof_threshold:
                reasons.append("model")
            if s.is_screen:
                reasons.append("screen")
            reason = " (" + "+".join(reasons) + ")"

        print(
            f"{path.name:<28} {prob_live:>9.3f} {s.replay_score:>7.3f} "
            f"{s.high_freq_ratio:>6.3f} {s.peak_score:>6.3f} {s.glare_ratio:>6.3f}  "
            f"{verdict}{reason}"
        )
        rows.append((path.name, prob_live, s.replay_score, is_live))

    print(
        "\nTuning tips:\n"
        "  - If a phone-screen photo shows LIVE: lower --screen-threshold toward its\n"
        "    'replay' score, and/or raise --spoof-threshold toward its 'prob_live'.\n"
        "  - If a real face shows SPOOF: do the opposite (loosen the offending knob).\n"
        "  - Pick values that sit BETWEEN your real and spoof samples, then set them in\n"
        "    .env as ML_SERVICE_SPOOF_THRESHOLD / ML_SERVICE_SPOOF_SCREEN_THRESHOLD.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
