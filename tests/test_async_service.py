import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from image_pipeline.service import (
    GenerateTaskRequest,
    _cost_metrics,
    app,
    require_service_token,
)


def test_size_aliases_normalize() -> None:
    assert GenerateTaskRequest(prompt="x", size="2048x2048").size == "2k"
    assert GenerateTaskRequest(prompt="x", target="3840x2160").size == "4k"


def test_infrastructure_cost_uses_real_bytes_and_seconds() -> None:
    cost = _cost_metrics(10_000_000, 12.0, None)
    assert cost["gpu_cloud_cny_range"] == [0.01, 0.016667]
    assert cost["storage_cny_first_month"] == 0.0012
    assert cost["download_traffic_cny_once"] == 0.005
    assert cost["api_cost_cny"] is None


def test_frontend_is_served_from_root() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert 'lang="zh-CN"' in response.text
    assert 'value="2k" checked' in response.text
    assert 'fetch("/v1/generate"' in response.text
    assert "POLL_INTERVAL_MS = 4000" in response.text
    assert "sessionStorage" in response.text
    assert 'headers.Authorization = `Bearer ${token}`' in response.text
    assert "生成失败，请重试" in response.text
    assert "https://" not in response.text


def test_optional_service_token_uses_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_SERVICE_TOKEN", "test-service-token")
    require_service_token("Bearer test-service-token")
    with pytest.raises(HTTPException) as error:
        require_service_token("Bearer wrong-token")
    assert error.value.status_code == 401
