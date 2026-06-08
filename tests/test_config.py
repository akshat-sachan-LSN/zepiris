"""Settings / environment behavior."""

from pathlib import Path

import pytest

from zepiris.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_requires_ml_inference_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No .env in cwd so validation depends only on environment variables."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZEPIRIS_ML_INFERENCE_SERVICE_URL", raising=False)
    monkeypatch.delenv("ML_INFERENCE_SERVICE_URL", raising=False)
    with pytest.raises(ValueError, match="ZEPIRIS_ML_INFERENCE_SERVICE_URL"):
        Settings()


def test_settings_accepts_ml_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPIRIS_ML_INFERENCE_SERVICE_URL", "http://ml-test:8001")
    s = Settings()
    assert s.ml_inference_service_url == "http://ml-test:8001"


def test_verify_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPIRIS_ML_INFERENCE_SERVICE_URL", "http://ml:8001")
    s = Settings()
    assert s.verify_threshold == 0.5
    assert s.reference_fetch_timeout_seconds == 10.0
    assert s.reference_max_bytes == 5 * 1024 * 1024
    assert not hasattr(s, "milvus_host")
    assert not hasattr(s, "minio_endpoint")


def test_get_settings_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEPIRIS_ML_INFERENCE_SERVICE_URL", "http://cached:8001")
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
