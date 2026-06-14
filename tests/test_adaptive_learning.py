"""Tests for the file-backed adaptive threshold learner."""

from __future__ import annotations

import json

import pytest

from zepiris.services.learning import AdaptiveThresholdLearner


def _learner(tmp_path, **kwargs) -> AdaptiveThresholdLearner:
    defaults = dict(min_genuine=3, min_impostor=3, far_target=0.01)
    defaults.update(kwargs)
    return AdaptiveThresholdLearner(tmp_path / "learning", **defaults)


def _feed(learner, doc_type, genuine_scores, impostor_scores, prefix=""):
    """Log one sample + matching feedback per score."""
    i = 0
    for score, genuine in [(s, True) for s in genuine_scores] + [
        (s, False) for s in impostor_scores
    ]:
        rid = f"{prefix}{doc_type}-{i}"
        learner.record_sample(
            request_id=rid, doc_type=doc_type, score=score, threshold=0.4, is_match=score >= 0.4
        )
        learner.record_feedback(request_id=rid, genuine=genuine)
        i += 1


def test_no_threshold_until_enough_labelled_data(tmp_path) -> None:
    learner = _learner(tmp_path, min_genuine=5, min_impostor=5)
    _feed(learner, "aadhaar", [0.6, 0.7], [0.2, 0.3])
    assert learner.learned_threshold("aadhaar") is None


def test_learns_separating_threshold(tmp_path) -> None:
    learner = _learner(tmp_path)
    genuine = [0.55, 0.6, 0.65, 0.7]
    impostor = [0.2, 0.25, 0.3, 0.35]
    _feed(learner, "aadhaar", genuine, impostor)

    t = learner.learned_threshold("aadhaar")
    assert t is not None
    assert max(impostor) < t <= min(genuine)  # separates the distributions
    # all genuine accepted, all impostors rejected at the learned threshold
    assert all(s >= t for s in genuine)
    assert all(s < t for s in impostor)


def test_threshold_is_per_doc_type(tmp_path) -> None:
    learner = _learner(tmp_path)
    _feed(learner, "aadhaar", [0.5, 0.55, 0.6], [0.2, 0.25, 0.3])
    assert learner.learned_threshold("aadhaar") is not None
    assert learner.learned_threshold("pan") is None


def test_threshold_clamped_to_safety_band(tmp_path) -> None:
    learner = _learner(tmp_path, floor=0.25, ceiling=0.70)
    # absurd labels: even "genuine" pairs score near 1.0 and impostors at 0.9
    _feed(learner, "pan", [0.97, 0.98, 0.99], [0.9, 0.91, 0.92])
    t = learner.learned_threshold("pan")
    assert t == pytest.approx(0.70)  # ceiling clamp, never drifts to 0.95


def test_thresholds_persist_across_instances(tmp_path) -> None:
    learner = _learner(tmp_path)
    _feed(learner, "aadhaar", [0.5, 0.55, 0.6], [0.2, 0.25, 0.3])
    t = learner.learned_threshold("aadhaar")

    reloaded = _learner(tmp_path)
    assert reloaded.learned_threshold("aadhaar") == pytest.approx(t)

    on_disk = json.loads((tmp_path / "learning" / "thresholds.json").read_text())
    assert "aadhaar" in on_disk


def test_disabled_learner_is_a_noop(tmp_path) -> None:
    learner = _learner(tmp_path, enabled=False)
    learner.record_sample(request_id="x", doc_type="aadhaar", score=0.5, threshold=0.4, is_match=True)
    out = learner.record_feedback(request_id="x", genuine=True)
    assert out["recorded"] is False
    assert learner.learned_threshold("aadhaar") is None
    assert not (tmp_path / "learning").exists()  # nothing written


def test_torn_jsonl_line_does_not_poison_log(tmp_path) -> None:
    learner = _learner(tmp_path)
    _feed(learner, "aadhaar", [0.5, 0.55], [0.2, 0.25])
    samples = tmp_path / "learning" / "samples.jsonl"
    with samples.open("a") as f:
        f.write('{"request_id": "torn\n')  # interrupted write
    _feed(learner, "aadhaar", [0.6], [0.3], prefix="b-")
    assert learner.learned_threshold("aadhaar") is not None
