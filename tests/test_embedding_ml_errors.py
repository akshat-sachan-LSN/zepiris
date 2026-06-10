"""ML-client HTTP failures surface as clean ZepIris errors, not bare 500s."""

import httpx
import pytest

from zepiris.exceptions import MLInferenceUpstreamError
from zepiris.services.embedding import _wrap_ml_errors


def _raise(status: int):
    req = httpx.Request("POST", "http://ml/v1/face/embed")
    resp = httpx.Response(status, json={"detail": "model not loaded"}, request=req)
    raise httpx.HTTPStatusError("err", request=req, response=resp)


def test_embed_503_becomes_service_error():
    with pytest.raises(MLInferenceUpstreamError) as ei:
        _wrap_ml_errors(lambda: _raise(503))
    assert ei.value.status_code == 503
    assert ei.value.detail["message"] == "ml_inference_request_failed"


def test_embed_500_becomes_502_bad_gateway():
    with pytest.raises(MLInferenceUpstreamError) as ei:
        _wrap_ml_errors(lambda: _raise(500))
    assert ei.value.status_code == 502
