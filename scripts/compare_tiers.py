#!/usr/bin/env python
"""Compare face tiers on speed AND on the separation that decides matches.

Throughput alone cannot justify a tier: a faster model that narrows the gap
between genuine and impostor scores buys latency with accuracy. This measures
both, on your own images, so the trade is a decision rather than a guess.

Genuine pairs come from realistic capture variation applied to each face —
rescaling, JPEG recompression, brightness shifts, small rotations — the kind of
difference between two photos of the same person. Impostor pairs are every
cross-person combination in the directory.

What matters is the **margin**: the gap between the worst genuine score and the
best impostor score. A tier whose margin stays wide is safe to adopt; one whose
margin collapses will produce false accepts on real traffic no matter how fast
it is.

Usage:
    python scripts/compare_tiers.py [image_dir] [--tiers accurate,balanced,fast]

Each image must contain exactly one person, one image per identity.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zepiris.ml_inference.face_embedding import FaceEmbeddingService  # noqa: E402


def _variants(image_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Plausible re-captures of the same face — the genuine-pair generator."""
    h, w = image_rgb.shape[:2]
    out: list[tuple[str, np.ndarray]] = []

    # A second capture at a different distance/resolution.
    small = cv2.resize(image_rgb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    out.append(("downscale", cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)))

    # A phone re-encode of an already-compressed photo.
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 55])
    if ok:
        out.append(("jpeg55", cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))

    # Under- and over-exposed capture.
    out.append(("dim", cv2.convertScaleAbs(image_rgb, alpha=0.65, beta=-12)))
    out.append(("bright", cv2.convertScaleAbs(image_rgb, alpha=1.3, beta=25)))

    # Slight handheld tilt.
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 7, 1.0)
    out.append(("rot7", cv2.warpAffine(image_rgb, rot, (w, h), borderMode=cv2.BORDER_REFLECT)))

    # Mild motion blur.
    out.append(("blur", cv2.GaussianBlur(image_rgb, (5, 5), 0)))
    return out


def _load(image_dir: Path) -> list[tuple[str, np.ndarray]]:
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    images = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  ! skipping unreadable {p.name}")
            continue
        images.append((p.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return images


def evaluate(tier: str, images: list[tuple[str, np.ndarray]]) -> dict | None:
    svc = FaceEmbeddingService(tier=tier, intra_op_threads=1, max_input_side=1600)
    svc.load_model()

    def vec(img: np.ndarray) -> np.ndarray | None:
        v, found, _ = svc.embed_vector(img)
        return v if found else None

    base: dict[str, np.ndarray] = {}
    undetected = []
    for name, img in images:
        v = vec(img)
        if v is None:
            undetected.append(name)
        else:
            base[name] = v

    if len(base) < 2:
        print(f"  {tier}: only {len(base)} face(s) detected — need at least 2")
        return None

    genuine: list[float] = []
    genuine_misses = 0
    for name, img in images:
        if name not in base:
            continue
        for _label, variant in _variants(img):
            v = vec(variant)
            if v is None:
                genuine_misses += 1
                continue
            genuine.append(float(np.dot(base[name], v)))

    impostor = [
        float(np.dot(base[a], base[b])) for a, b in itertools.combinations(sorted(base), 2)
    ]

    # Throughput proxy: full pair matches, one core, back to back.
    pair = [img for _n, img in images[:2]] or [images[0][1], images[0][1]]
    svc.match_pair(pair[0], pair[-1])
    reps = 8
    t = time.perf_counter()
    for _ in range(reps):
        svc.match_pair(pair[0], pair[-1])
    per_pair_ms = (time.perf_counter() - t) / reps * 1000

    return {
        "tier": tier,
        "genuine": genuine,
        "impostor": impostor,
        "undetected": undetected,
        "genuine_misses": genuine_misses,
        "per_pair_ms": per_pair_ms,
        "faces": len(base),
    }


def report(results: list[dict], threshold: float) -> None:
    print()
    print(f"{'tier':<10} {'ms/pair':>8} {'genuine min':>12} {'genuine mean':>13} "
          f"{'impostor max':>13} {'margin':>8} {'FA':>4} {'FR':>4}")
    print("-" * 82)
    for r in results:
        g, i = r["genuine"], r["impostor"]
        margin = min(g) - max(i)
        false_accepts = sum(s >= threshold for s in i)
        false_rejects = sum(s < threshold for s in g)
        print(
            f"{r['tier']:<10} {r['per_pair_ms']:>8.1f} {min(g):>12.3f} "
            f"{statistics.mean(g):>13.3f} {max(i):>13.3f} {margin:>8.3f} "
            f"{false_accepts:>4} {false_rejects:>4}"
        )

    print()
    print(f"threshold = {threshold}   genuine pairs = {len(results[0]['genuine'])}   "
          f"impostor pairs = {len(results[0]['impostor'])}")
    print()
    print("margin = worst genuine score - best impostor score. Wider is safer;")
    print("a margin at or below zero means the two populations overlap and no")
    print("single threshold separates them.")
    for r in results:
        if r["undetected"]:
            print(f"  {r['tier']}: no face found in {', '.join(r['undetected'])}")
        if r["genuine_misses"]:
            print(f"  {r['tier']}: detection failed on {r['genuine_misses']} variant(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_dir", nargs="?", default="gallery",
                    help="Directory of one-face-per-identity images (default: gallery)")
    ap.add_argument("--tiers", default="accurate,balanced,fast",
                    help="Comma-separated tiers to compare")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold to score FA/FR against (default: 0.5)")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"error: {image_dir} is not a directory")
        return 1

    images = _load(image_dir)
    if len(images) < 2:
        print(f"error: need at least 2 images in {image_dir}, found {len(images)}")
        return 1
    print(f"{len(images)} identities from {image_dir}/")

    results = []
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        print(f"evaluating {tier} …")
        r = evaluate(tier, images)
        if r:
            results.append(r)

    if not results:
        print("error: no tier produced usable results")
        return 1
    report(results, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
