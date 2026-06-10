"""Face detection robustness: padding-retry fallback for frame-filling faces."""

import numpy as np
import pytest

from zepiris.ml_inference.face_embedding import FaceEmbeddingService


class _FakeDetModel:
    """Detector that fails on the original image but succeeds once padded.

    Mirrors the real-world edge case: a tightly cropped face that fills the frame
    is missed until a margin is added around it.
    """

    def __init__(self, original_hw: tuple[int, int]) -> None:
        self._original_hw = original_hw
        self.calls: list[tuple[int, int]] = []

    def detect(self, image, max_num=0, metric="default"):
        h, w = image.shape[:2]
        self.calls.append((h, w))
        if (h, w) == self._original_hw:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
        bbox = np.array([[w * 0.25, h * 0.25, w * 0.75, h * 0.75, 0.99]], dtype=np.float32)
        kps = np.full((1, 5, 2), w * 0.5, dtype=np.float32)
        return bbox, kps


class _FakeApp:
    def __init__(self, det_model: _FakeDetModel) -> None:
        self.det_model = det_model
        self.models: dict = {}


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _service_with_fake(monkeypatch, image, **kwargs) -> tuple[FaceEmbeddingService, _FakeDetModel]:
    service = FaceEmbeddingService(**kwargs)
    det = _FakeDetModel(original_hw=image.shape[:2])
    monkeypatch.setattr(service, "load_model", lambda: _FakeApp(det))
    return service, det


def test_padding_retry_recovers_frame_filling_face(monkeypatch, image) -> None:
    # Isolate the padding path: disable the low-threshold and upscale retries so
    # the detector is tried exactly twice (original, then padded).
    service, det = _service_with_fake(
        monkeypatch,
        image,
        enable_padding_retry=True,
        padding_fraction=0.25,
        low_det_thresh=0.5,
        enable_upscale_retry=False,
    )

    result = service.preprocess(image)

    assert result["face"] is not None
    # Recognition runs on the padded image so the embedding keeps full context.
    assert result["image"].shape[:2] != image.shape[:2]
    assert result["image"].shape[0] > image.shape[0]
    # Detector was tried twice: original (failed) then padded (succeeded).
    assert len(det.calls) == 2


def test_padding_retry_disabled_returns_no_face(monkeypatch, image) -> None:
    service, det = _service_with_fake(
        monkeypatch,
        image,
        enable_padding_retry=False,
        low_det_thresh=0.5,
        enable_upscale_retry=False,
    )

    result = service.preprocess(image)

    assert result["face"] is None
    assert result["image"].shape[:2] == image.shape[:2]
    assert len(det.calls) == 1


def test_first_pass_hit_skips_padding(monkeypatch, image) -> None:
    service = FaceEmbeddingService(enable_padding_retry=True)

    class _AlwaysDetect:
        calls = 0

        def detect(self, img, max_num=0, metric="default"):
            type(self).calls += 1
            h, w = img.shape[:2]
            bbox = np.array([[w * 0.3, h * 0.3, w * 0.7, h * 0.7, 0.99]], dtype=np.float32)
            kps = np.full((1, 5, 2), w * 0.5, dtype=np.float32)
            return bbox, kps

    det = _AlwaysDetect()
    monkeypatch.setattr(service, "load_model", lambda: _FakeApp(det))

    result = service.preprocess(image)

    assert result["face"] is not None
    assert result["image"].shape[:2] == image.shape[:2]
    assert det.calls == 1


class _MultiFaceDet:
    """Detector returning a small central face and a large off-center face."""

    def detect(self, image, max_num=0, metric="default"):
        h, w = image.shape[:2]
        small_central = [w * 0.45, h * 0.45, w * 0.55, h * 0.55, 0.97]   # tiny, centered
        large_corner = [w * 0.05, h * 0.05, w * 0.55, h * 0.75, 0.95]    # big, off-center
        bbox = np.array([small_central, large_corner], dtype=np.float32)
        kps = np.zeros((2, 5, 2), dtype=np.float32)
        kps[1] = w * 0.3  # distinct kps for the large face
        return bbox, kps


def test_selects_largest_face_not_central(monkeypatch) -> None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    service = FaceEmbeddingService()
    monkeypatch.setattr(service, "load_model", lambda: _FakeApp(_MultiFaceDet()))
    face = service._select_face(img)
    assert face is not None
    # The large off-center face spans most of the frame; its width >> the tiny one.
    assert (face.bbox[2] - face.bbox[0]) > 0.3 * 200


class _ThreshAwareDet:
    """Detector that only finds a face when det_thresh is lowered (mimics a faint
    document photo) — verifies the low-threshold fallback in preprocess()."""

    def __init__(self) -> None:
        self.det_thresh = 0.5
        self.calls: list[float] = []

    def detect(self, image, max_num=0, metric="default"):
        self.calls.append(self.det_thresh)
        if self.det_thresh <= 0.35:  # only the low-confidence retry succeeds
            h, w = image.shape[:2]
            bbox = np.array([[w * 0.2, h * 0.2, w * 0.6, h * 0.6, 0.4]], dtype=np.float32)
            kps = np.full((1, 5, 2), w * 0.3, dtype=np.float32)
            return bbox, kps
        return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)


def test_low_threshold_fallback_recovers_faint_face(monkeypatch) -> None:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    service = FaceEmbeddingService(enable_padding_retry=False, enable_upscale_retry=False)
    det = _ThreshAwareDet()
    monkeypatch.setattr(service, "load_model", lambda: _FakeApp(det))
    out = service.preprocess(img)
    assert out["face"] is not None              # recovered via the low-thresh retry
    assert any(t <= 0.35 for t in det.calls)    # the fallback threshold was used
