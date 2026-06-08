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
    service, det = _service_with_fake(
        monkeypatch, image, enable_padding_retry=True, padding_fraction=0.25
    )

    result = service.preprocess(image)

    assert result["face"] is not None
    # Recognition runs on the padded image so the embedding keeps full context.
    assert result["image"].shape[:2] != image.shape[:2]
    assert result["image"].shape[0] > image.shape[0]
    # Detector was tried twice: original (failed) then padded (succeeded).
    assert len(det.calls) == 2


def test_padding_retry_disabled_returns_no_face(monkeypatch, image) -> None:
    service, det = _service_with_fake(monkeypatch, image, enable_padding_retry=False)

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
