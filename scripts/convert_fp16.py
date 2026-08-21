#!/usr/bin/env python
"""Convert the face detector and recognizer ONNX models from FP32 to FP16.

On a T4 (and most other NVIDIA cards with tensor cores) the FP16 path delivers
~8x more TFLOPS than FP32.  The FP32 buffalo_l models run on plain CUDA cores
because ONNX Runtime only dispatches to tensor cores for FP16 graphs — so the
GPU is drawing its full 70 W power budget while using less than 15% of its
theoretical compute.  Converting both models unlocks that headroom.

The converted files are written next to their FP32 originals with a ``_fp16``
suffix, e.g.:

    ~/.insightface/models/buffalo_l/det_10g_fp16.onnx
    ~/.insightface/models/buffalo_l/w600k_r50_fp16.onnx

Point ``ML_SERVICE_FACE_DET_MODEL_PATH`` / ``ML_SERVICE_FACE_REC_MODEL_PATH``
at the output paths (already wired into the GPU compose overlay).

Usage
-----
    # Convert the deployed "balanced" tier (det_500m + w600k_r50):
    python scripts/convert_fp16.py

    # Convert a different tier:
    python scripts/convert_fp16.py --tier accurate

    # Specify explicit input paths (e.g. after a manual download):
    python scripts/convert_fp16.py \\
        --det-path /path/to/det_10g.onnx \\
        --rec-path /path/to/w600k_r50.onnx

    # Dry-run: print paths that would be written without writing anything:
    python scripts/convert_fp16.py --dry-run

Dependencies
------------
    pip install onnxconverter-common
``onnx`` and ``onnxruntime`` already come with the ``ml`` extra;
``onnxconverter-common`` is not a project dependency and has to be installed
for this one-off conversion.

Notes
-----
* ``onnxconverter_common.float16.convert_float_to_float16`` casts weights and
  activations to FP16. ``keep_io_types=True`` keeps the graph's **inputs and
  outputs** in FP32, which is what lets InsightFace feed the same arrays as
  before with no code change; it does not, on its own, hold any operator in
  FP32. The converter's own default block list covers the ops that are unstable
  in half precision, so scores move only by rounding — which is exactly what the
  validation step below is for.

* Validation runs a single dummy forward pass through ORT with the converted
  model before writing to disk.  If the session raises, the FP32 source is
  untouched and the script exits non-zero.

* SCRFD detector inputs are already FP32; the recognizer (ArcFace) runs on a
  112×112 aligned crop.  Both convert cleanly under onnxconverter-common.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Deferred heavy imports — report a clean error if deps are missing
# ---------------------------------------------------------------------------
def _require(pkg: str, pip_name: str | None = None) -> None:
    import importlib

    try:
        importlib.import_module(pkg)
    except ModuleNotFoundError:
        pip = pip_name or pkg
        print(f"[error] {pkg} not found.  Install it with:  pip install {pip}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _output_path(source: str) -> Path:
    """Derive the FP16 output path: same dir, ``<stem>_fp16.onnx``."""
    p = Path(source)
    return p.parent / f"{p.stem}_fp16{p.suffix}"


def _convert_model(
    source_path: str, label: str, *, dry_run: bool = False, validate: bool = True
) -> str:
    """Convert one FP32 ONNX model to FP16.

    Returns the path of the written FP16 file (or the would-be path on
    --dry-run).  Raises on any conversion or validation failure.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    out_path = _output_path(source_path)

    print(f"\n[{label}]  source : {source_path}")
    print(f"[{label}]  output : {out_path}")

    if dry_run:
        print(f"[{label}]  --dry-run: skipping conversion")
        return str(out_path)

    # ---- load ----
    t0 = time.perf_counter()
    model_fp32 = onnx.load(source_path)
    print(f"[{label}]  loaded  ({time.perf_counter() - t0:.2f}s)")

    # ---- convert ----
    # keep_io_types=True preserves FP32 graph inputs/outputs so the calling
    # code is unchanged; ops that need FP32 precision are kept automatically.
    t1 = time.perf_counter()
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    print(f"[{label}]  converted ({time.perf_counter() - t1:.2f}s)")

    # ---- validate: write to a tempfile, run a dummy inference ----
    t2 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        onnx.save(model_fp16, tmp_path)
        if not validate:
            print(f"[{label}]  validation skipped (--skip-validation)")
            os.replace(tmp_path, str(out_path))
            return str(out_path)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress INFO
        sess = ort.InferenceSession(tmp_path, sess_opts, providers=["CPUExecutionProvider"])

        # Build minimal dummy inputs matching the model's declared input shapes.
        # Symbolic dims (batch, height, width) are replaced with 1 / a small
        # concrete value so the pass exercises the graph without a real image.
        inputs = {}
        for inp in sess.get_inputs():
            shape = [
                d if isinstance(d, int) and d > 0 else 1
                for d in inp.shape
            ]
            # SCRFD detector: [N, 3, H, W] — use 128 px so anchor decoding
            # produces at least one candidate.
            if len(shape) == 4 and shape[1] == 3:
                shape[2] = max(shape[2], 128)
                shape[3] = max(shape[3], 128)
            dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
            inputs[inp.name] = np.zeros(shape, dtype=dtype)

        sess.run(None, inputs)
        print(f"[{label}]  validated ({time.perf_counter() - t2:.2f}s)")
    except Exception as exc:
        os.unlink(tmp_path)
        raise RuntimeError(
            f"[{label}] FP16 validation failed — FP32 source is untouched.\n"
            f"  Error: {exc}"
        ) from exc

    # ---- write final output ----
    os.replace(tmp_path, str(out_path))
    size_mb = out_path.stat().st_size / 1_048_576
    print(
        f"[{label}]  written  {out_path.name}  ({size_mb:.1f} MB, "
        f"total {time.perf_counter() - t0:.2f}s)"
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tier",
        default="balanced",
        choices=["accurate", "balanced", "fast"],
        help=(
            "Tier whose stock FP32 models will be converted (default: balanced, "
            "the tier the GPU overlay deploys)"
        ),
    )
    p.add_argument(
        "--det-path",
        metavar="PATH",
        default="",
        help="Explicit detector .onnx path; overrides --tier for the detector",
    )
    p.add_argument(
        "--rec-path",
        metavar="PATH",
        default="",
        help="Explicit recognizer .onnx path; overrides --tier for the recognizer",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would be written without converting anything",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Skip the dummy-inference validation step.  Use only if CPU "
            "validation passes but ORT raises on the target GPU (rare)."
        ),
    )
    return p.parse_args()


def main() -> None:
    # Parse first: --help must work on a machine that has none of the conversion
    # dependencies installed, which is most machines that will run --help.
    args = _parse_args()

    _require("onnx")
    _require("onnxconverter_common", "onnxconverter-common")
    _require("onnxruntime")

    # Resolve source paths — reuse the engine's own logic so the download cache
    # is shared and the paths are guaranteed to be consistent with what the
    # service would load.
    if args.det_path and args.rec_path:
        det_src, rec_src = args.det_path, args.rec_path
    else:
        # Import here so the script can print --help without needing the full
        # zepiris package on PATH.
        try:
            from zepiris.ml_inference.face_engine import EngineConfig, resolve_paths
        except ImportError:
            print(
                "[error] zepiris package not found on sys.path.\n"
                "  Run this script from the repo root with the venv active:\n"
                "    source .venv/bin/activate && python scripts/convert_fp16.py",
                file=sys.stderr,
            )
            sys.exit(1)

        cfg = EngineConfig(tier=args.tier)
        print(f"Resolving FP32 model paths for tier={args.tier!r} …")
        try:
            det_src, rec_src = resolve_paths(cfg)
        except Exception as exc:
            print(f"[error] Could not resolve model paths: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"  detector   : {det_src}")
    print(f"  recognizer : {rec_src}")

    errors: list[str] = []

    try:
        det_out = _convert_model(
            det_src, "detector", dry_run=args.dry_run, validate=not args.skip_validation
        )
    except Exception as exc:
        print(f"\n[error] detector conversion failed:\n  {exc}", file=sys.stderr)
        errors.append("detector")
        det_out = ""

    try:
        rec_out = _convert_model(
            rec_src, "recognizer", dry_run=args.dry_run, validate=not args.skip_validation
        )
    except Exception as exc:
        print(f"\n[error] recognizer conversion failed:\n  {exc}", file=sys.stderr)
        errors.append("recognizer")
        rec_out = ""

    if errors:
        print(f"\n[FAILED] {', '.join(errors)} conversion failed — see errors above.", file=sys.stderr)
        sys.exit(1)

    print("\n[done] Add these to docker-compose.gpu.yml (or your .env):")
    print(f"  ML_SERVICE_FACE_DET_MODEL_PATH: {det_out!r}")
    print(f"  ML_SERVICE_FACE_REC_MODEL_PATH: {rec_out!r}")
    print(
        "\nValidate match scores before deploying — prod_replay reports |dscore|\n"
        "against the scores in the recording, and no verdict may flip near the\n"
        "threshold:\n"
        "  python scripts/prod_replay.py --base-url http://<host>:8000 --rows 400\n"
        "  python scripts/compare_tiers.py   # genuine/impostor margin must not narrow"
    )


if __name__ == "__main__":
    main()
