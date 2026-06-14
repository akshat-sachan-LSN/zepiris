"""Online threshold calibration from production feedback.

The recognition model itself is never retrained in production (online
fine-tuning of a face model is unauditable and drifts); what *can* safely keep
learning is the decision threshold. Indian document photos — Aadhaar's small
laminated print, PAN's older low-contrast photos — produce score distributions
that differ from clean reference photos and from each other, so each document
type earns its own learned operating point.

The loop:

1. Every verification logs its score + decision (scores only — no images, no
   PII) to ``samples.jsonl``.
2. Operators report confirmed outcomes (``genuine`` = the selfie and document
   really were the same person) via the feedback endpoint, appended to
   ``feedback.jsonl``.
3. On each feedback, the labelled scores for that document type are re-fit:
   pick the threshold meeting the false-accept-rate target with the highest
   true-accept-rate, clamp it to a sane band, persist to ``thresholds.json``.
4. Verify requests without an explicit ``threshold`` use the learned value
   when enough labelled data exists, else the configured default.

Everything is plain JSONL/JSON on disk: auditable, diffable, trivially
resettable by deleting the directory.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

#: doc_type bucket used when the caller does not say which document it is.
GENERIC_DOC_TYPE = "document"
#: doc_type bucket for plain face-to-face matches.
FACE_KIND = "facematch"


class AdaptiveThresholdLearner:
    """File-backed, per-document-type threshold calibration."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        min_genuine: int = 20,
        min_impostor: int = 20,
        far_target: float = 0.01,
        floor: float = 0.25,
        ceiling: float = 0.70,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            data_dir: Directory for samples.jsonl / feedback.jsonl / thresholds.json.
            min_genuine: Labelled genuine pairs required before a threshold is learned.
            min_impostor: Labelled impostor pairs required before a threshold is learned.
            far_target: Maximum acceptable false-accept rate when fitting.
            floor: Lowest threshold the learner may ever emit (safety clamp).
            ceiling: Highest threshold the learner may ever emit (safety clamp).
            enabled: When False every method is a cheap no-op (tests, opt-out).
        """
        self._dir = Path(data_dir)
        self._min_genuine = min_genuine
        self._min_impostor = min_impostor
        self._far_target = far_target
        self._floor = floor
        self._ceiling = ceiling
        self._enabled = enabled
        self._lock = threading.Lock()
        self._samples_path = self._dir / "samples.jsonl"
        self._feedback_path = self._dir / "feedback.jsonl"
        self._thresholds_path = self._dir / "thresholds.json"
        self._thresholds: dict[str, dict] = self._load_thresholds()

    # -- recording ----------------------------------------------------------

    def record_sample(
        self,
        *,
        request_id: str,
        doc_type: str,
        score: float,
        threshold: float,
        is_match: bool,
        liveness_score: float | None = None,
        blur_score: float | None = None,
        nsfw_safe_score: float | None = None,
    ) -> None:
        """Log one verification's scores (called on every scored verify).

        The match ``score`` drives threshold calibration; the optional quality
        scores (liveness/blur/nsfw) are logged alongside so the labelled history
        carries the full signal for future analysis and calibration.
        """
        if not self._enabled:
            return
        row = {
            "request_id": request_id,
            "doc_type": doc_type,
            "score": score,
            "threshold": threshold,
            "is_match": is_match,
            "liveness_score": liveness_score,
            "blur_score": blur_score,
            "nsfw_safe_score": nsfw_safe_score,
            "ts": time.time(),
        }
        with self._lock:
            self._append(self._samples_path, row)

    def record_feedback(self, *, request_id: str, genuine: bool) -> dict:
        """Record a confirmed outcome and re-fit the affected doc type.

        Returns a summary: whether the request_id matched a logged sample and
        the current learned thresholds.
        """
        if not self._enabled:
            return {"recorded": False, "matched_sample": False, "thresholds": {}}
        with self._lock:
            self._append(
                self._feedback_path,
                {"request_id": request_id, "genuine": genuine, "ts": time.time()},
            )
            doc_type = self._doc_type_of(request_id)
            if doc_type is not None:
                self._recalibrate(doc_type)
            return {
                "recorded": True,
                "matched_sample": doc_type is not None,
                "doc_type": doc_type,
                "thresholds": {k: v["threshold"] for k, v in self._thresholds.items()},
            }

    # -- lookup --------------------------------------------------------------

    def learned_threshold(self, doc_type: str) -> float | None:
        """The learned threshold for this doc type, or None if not enough data yet."""
        if not self._enabled:
            return None
        entry = self._thresholds.get(doc_type)
        return float(entry["threshold"]) if entry else None

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _append(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn write must not poison the whole log
        return rows

    def _load_thresholds(self) -> dict[str, dict]:
        if not self._thresholds_path.exists():
            return {}
        try:
            return json.loads(self._thresholds_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _doc_type_of(self, request_id: str) -> str | None:
        for row in self._read_jsonl(self._samples_path):
            if row.get("request_id") == request_id:
                return row.get("doc_type")
        return None

    def _labelled_scores(self, doc_type: str) -> tuple[list[float], list[float]]:
        """Join samples with feedback -> (genuine_scores, impostor_scores)."""
        labels: dict[str, bool] = {}
        for row in self._read_jsonl(self._feedback_path):
            rid = row.get("request_id")
            if rid is not None and isinstance(row.get("genuine"), bool):
                labels[rid] = row["genuine"]  # latest feedback wins

        genuine: list[float] = []
        impostor: list[float] = []
        for row in self._read_jsonl(self._samples_path):
            if row.get("doc_type") != doc_type:
                continue
            rid = row.get("request_id")
            score = row.get("score")
            if rid not in labels or not isinstance(score, (int, float)):
                continue
            (genuine if labels[rid] else impostor).append(float(score))
        return genuine, impostor

    def _recalibrate(self, doc_type: str) -> None:
        genuine, impostor = self._labelled_scores(doc_type)
        if len(genuine) < self._min_genuine or len(impostor) < self._min_impostor:
            return
        threshold = self._fit_threshold(genuine, impostor)
        self._thresholds[doc_type] = {
            "threshold": threshold,
            "genuine_count": len(genuine),
            "impostor_count": len(impostor),
            "updated_ts": time.time(),
        }
        self._thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._thresholds_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._thresholds, indent=2), encoding="utf-8")
        tmp.replace(self._thresholds_path)

    def _fit_threshold(self, genuine: list[float], impostor: list[float]) -> float:
        """Lowest threshold meeting the FAR target with the highest TAR.

        Falls back to the best balanced accuracy when no candidate meets the
        FAR target, and always clamps into [floor, ceiling] so a burst of bad
        labels can never push the system into absurd accept/reject behavior.
        """
        scores = sorted(set(genuine) | set(impostor))
        candidates = [(a + b) / 2.0 for a, b in zip(scores, scores[1:])]
        candidates += [scores[0] - 1e-6, scores[-1] + 1e-6]

        def rates(t: float) -> tuple[float, float]:
            tar = sum(s >= t for s in genuine) / len(genuine)
            far = sum(s >= t for s in impostor) / len(impostor)
            return tar, far

        feasible = [(t, *rates(t)) for t in candidates]
        meeting = [(tar, -t) for t, tar, far in feasible if far <= self._far_target]
        if meeting:
            tar, neg_t = max(meeting)
            best = -neg_t
        else:
            # No threshold meets the FAR target yet: maximize balanced
            # accuracy (equivalent to maximizing TAR - FAR).
            best = max(candidates, key=lambda t: rates(t)[0] - rates(t)[1])
        return min(max(best, self._floor), self._ceiling)
