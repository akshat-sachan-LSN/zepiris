"""ONNX Runtime session construction for the face detector + recognizer.

InsightFace's ``FaceAnalysis`` builds its ORT sessions with default options,
which means every session sizes its intra-op thread pool to the whole machine.
That is the right choice for one request at a time and the wrong one under
concurrency: with N in-flight requests each fanning out across every core, the
pools oversubscribe the CPU and throughput collapses while latency multiplies.

This module builds the sessions itself so the thread pools can be pinned (one
core per session by default) and parallelism comes from running many requests
at once instead — the arrangement that actually scales. It also decouples the
detector from the recognizer so the two can be sized independently:

    tier        detector      recognizer     relative throughput
    --------    ----------    -----------    -------------------
    accurate    det_10g       w600k_r50      1x   (InsightFace buffalo_l)
    balanced    det_500m      w600k_r50      ~2x  (same recognition weights)
    fast        det_500m      w600k_mbf      ~9x  (MobileFaceNet recognition)

``balanced`` keeps the ResNet50 recognition model — the part that determines
match scores — and only swaps the detector, so scores move very little. ``fast``
changes the embedding space itself and needs its threshold recalibrated before
use. See ``scripts/compare_tiers.py``.
"""

from __future__ import annotations

import logging
import os
import os.path as osp
from dataclasses import dataclass

import onnxruntime
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils import storage

logger = logging.getLogger(__name__)

#: tier -> (model pack holding the detector, detector file,
#:          model pack holding the recognizer, recognizer file)
TIERS: dict[str, tuple[str, str, str, str]] = {
    "accurate": ("buffalo_l", "det_10g.onnx", "buffalo_l", "w600k_r50.onnx"),
    "balanced": ("buffalo_s", "det_500m.onnx", "buffalo_l", "w600k_r50.onnx"),
    "fast": ("buffalo_s", "det_500m.onnx", "buffalo_s", "w600k_mbf.onnx"),
}

DEFAULT_TIER = "balanced"


@dataclass
class EngineConfig:
    """How to build the two ORT sessions.

    Attributes:
        tier: Key into :data:`TIERS`, or "custom" when explicit paths are given.
        det_size: Detector input resolution (w, h). Cost scales with area, so
            320x320 costs ~40% of 512x512. A selfie face is large in frame and
            detects fine at 320; small printed document faces need 512+.
        det_thresh: Primary detector confidence.
        intra_op_threads: Threads *inside* one inference. 1 means a request is
            served by a single core and concurrency comes from serving many
            requests at once — highest total throughput. Raise it only when
            request concurrency is low and single-request latency is what
            matters.
        inter_op_threads: Threads across independent graph branches. These
            models are essentially sequential, so >1 buys nothing.
        device: "cpu" or "cuda".
        det_model_path / rec_model_path: Explicit .onnx overrides; when set they
            win over ``tier``.
    """

    tier: str = DEFAULT_TIER
    det_size: tuple[int, int] = (512, 512)
    det_thresh: float = 0.5
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    device: str = "cpu"
    det_model_path: str | None = None
    rec_model_path: str | None = None


class FaceEngine:
    """A detector + recognizer pair, exposed in the shape ``FaceAnalysis`` uses.

    Deliberately duck-types ``FaceAnalysis``: ``.det_model`` and
    ``.models["recognition"]`` are what the embedding service reaches for, so it
    works against either without branching.
    """

    def __init__(self, det_model: SCRFD, rec_model: ArcFaceONNX, config: EngineConfig) -> None:
        self.det_model = det_model
        self.models = {"detection": det_model, "recognition": rec_model}
        self.config = config


def _session_options(cfg: EngineConfig) -> onnxruntime.SessionOptions:
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = cfg.intra_op_threads
    so.inter_op_num_threads = cfg.inter_op_threads
    so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Every request allocates its own tensors; a shared arena across threads is
    # contention we do not need.
    so.enable_cpu_mem_arena = True
    so.log_severity_level = 3
    return so


def _providers(device: str) -> list[str]:
    if device != "cpu":
        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.warning("device=%s requested but CUDA provider unavailable; using CPU", device)
    return ["CPUExecutionProvider"]


def _model_path(pack: str, filename: str) -> str:
    """Locate a model file inside an InsightFace pack, downloading the pack if needed."""
    root = os.environ.get(
        "INSIGHTFACE_HOME", osp.join(osp.expanduser("~"), ".insightface")
    )
    path = osp.join(root, "models", pack, filename)
    if osp.exists(path):
        return path
    # Not cached yet — fetch the pack through InsightFace's own mechanism so the
    # layout matches what it would have produced itself.
    storage.ensure_available("models", pack, root=root)
    if not osp.exists(path):
        raise FileNotFoundError(f"{filename} not found in model pack {pack!r} at {path}")
    return path


def resolve_paths(cfg: EngineConfig) -> tuple[str, str]:
    """Return ``(detector_path, recognizer_path)`` for this config."""
    if cfg.det_model_path and cfg.rec_model_path:
        return cfg.det_model_path, cfg.rec_model_path
    if cfg.tier not in TIERS:
        raise ValueError(f"unknown face tier {cfg.tier!r}; expected one of {sorted(TIERS)}")
    det_pack, det_file, rec_pack, rec_file = TIERS[cfg.tier]
    det_path = cfg.det_model_path or _model_path(det_pack, det_file)
    rec_path = cfg.rec_model_path or _model_path(rec_pack, rec_file)
    return det_path, rec_path


def build_engine(cfg: EngineConfig) -> FaceEngine:
    """Construct the detector + recognizer with pinned thread pools."""
    det_path, rec_path = resolve_paths(cfg)
    so = _session_options(cfg)
    providers = _providers(cfg.device)

    det = SCRFD(
        model_file=det_path,
        session=onnxruntime.InferenceSession(det_path, so, providers=providers),
    )
    det.prepare(0 if cfg.device != "cpu" else -1, det_thresh=cfg.det_thresh, input_size=cfg.det_size)

    rec = ArcFaceONNX(
        model_file=rec_path,
        session=onnxruntime.InferenceSession(rec_path, so, providers=providers),
    )

    logger.info(
        "Face engine ready: tier=%s det=%s@%dx%d rec=%s intra_op=%d device=%s",
        cfg.tier,
        osp.basename(det_path),
        cfg.det_size[0],
        cfg.det_size[1],
        osp.basename(rec_path),
        cfg.intra_op_threads,
        cfg.device,
    )
    return FaceEngine(det, rec, cfg)
