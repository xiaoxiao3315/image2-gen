from __future__ import annotations

import asyncio
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import image_pipeline.service as service


def test_public_placeholder_tokens_are_never_treated_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_SERVICE_TOKEN", "<set-with-a-random-leader-token>")
    monkeypatch.setenv(
        "IMAGE_UPSCALE_WORKER_TOKEN",
        "<set-with-a-distinct-random-worker-token-at-least-32-chars>",
    )
    monkeypatch.setenv(
        "IMAGE_LITELLM_BACKEND_TOKEN",
        "<set-with-a-third-distinct-random-token-at-least-32-chars>",
    )

    with pytest.raises(HTTPException) as caught:
        service.require_service_token(
            "Bearer <set-with-a-random-leader-token>"
        )
    assert caught.value.status_code == 503
    assert service._service_token_ready() is False
    assert service._worker_token_ready() is False
    assert service._litellm_backend_token_ready() is False

    monkeypatch.setenv("IMAGE_SERVICE_TOKEN", "")
    monkeypatch.setenv("IMAGE_REQUIRE_SERVICE_AUTH", "true")
    with pytest.raises(HTTPException) as required:
        service.require_service_token(None)
    assert required.value.status_code == 503


def test_idempotency_is_scoped_and_rejects_payload_reuse(tmp_path: Any) -> None:
    store = service.TaskStore(tmp_path / "idempotency.db")

    first, reused = store.create_batch_idempotent(
        "prompt-a", "2k", 2, 2, "tenant-a", "request-key-001"
    )
    assert reused is False

    second, reused = store.create_batch_idempotent(
        "prompt-a", "2k", 2, 2, "tenant-a", "request-key-001"
    )
    assert reused is True
    assert [task["task_id"] for task in second] == [task["task_id"] for task in first]

    with pytest.raises(service.IdempotencyConflict):
        store.create_batch_idempotent(
            "different-prompt", "2k", 2, 2, "tenant-a", "request-key-001"
        )

    other_tenant, reused = store.create_batch_idempotent(
        "prompt-a", "2k", 2, 2, "tenant-b", "request-key-001"
    )
    assert reused is False
    assert other_tenant[0]["batch_id"] != first[0]["batch_id"]

    with store.connect() as connection:
        connection.execute("DELETE FROM tasks WHERE task_id=?", (first[1]["task_id"],))
    with pytest.raises(service.IdempotencyConflict, match="incomplete"):
        store.create_batch_idempotent(
            "prompt-a", "2k", 2, 2, "tenant-a", "request-key-001"
        )

    with store.connect() as connection:
        connection.execute("DELETE FROM tasks WHERE batch_id=?", (first[0]["batch_id"],))
    recreated, reused = store.create_batch_idempotent(
        "prompt-a", "2k", 2, 2, "tenant-a", "request-key-001"
    )
    assert reused is False
    assert recreated[0]["batch_id"] != first[0]["batch_id"]


def test_openai_image_backend_is_private_idempotent_and_returns_public_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_token = "b" * 32
    calls: list[dict[str, Any]] = []

    class FakeManager:
        def submit_batch_idempotent(
            self,
            prompt: str,
            size: str,
            count: int,
            requested_concurrency: int,
            tenant_scope: str,
            idempotency_key: str,
        ) -> tuple[list[dict[str, Any]], bool]:
            calls.append(
                {
                    "prompt": prompt,
                    "size": size,
                    "count": count,
                    "concurrency": requested_concurrency,
                    "tenant_scope": tenant_scope,
                    "idempotency_key": idempotency_key,
                }
            )
            return (
                [
                    {"task_id": f"task-{index}", "batch_id": "batch-1"}
                    for index in range(count)
                ],
                False,
            )

    async def fake_wait(
        _batch_id: str, _timeout_seconds: int
    ) -> list[dict[str, Any]]:
        return [
            {"status": "done", "image_filename": "one.png"},
            {"status": "done", "image_filename": "two.png"},
        ]

    monkeypatch.setenv("IMAGE_LITELLM_BACKEND_TOKEN", backend_token)
    monkeypatch.setenv("IMAGE_LITELLM_PRIVATE_MODEL", "image-pipeline-private")
    monkeypatch.setenv("IMAGE_PUBLIC_MODEL_ALIAS", "image-gen")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://public.example")
    monkeypatch.delenv("IMAGE_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("IMAGE_UPSCALE_WORKER_TOKEN", raising=False)
    monkeypatch.setattr(service, "manager", FakeManager())
    monkeypatch.setattr(service, "_wait_for_terminal_batch", fake_wait)
    client = TestClient(service.app)
    body = {
        "model": "image-pipeline-private",
        "prompt": "draw a clean city park",
        "n": 2,
        "size": "2048x2048",
        "response_format": "url",
        "user": "opaque-tenant-scope",
    }

    unauthenticated = client.post("/v1/images/generations", json=body)
    assert unauthenticated.status_code == 401

    rejected = client.post(
        "/v1/images/generations",
        headers={"Authorization": f"Bearer {backend_token}"},
        json={**body, "model": "image-gen"},
    )
    assert rejected.status_code == 403

    response = client.post(
        "/v1/images/generations",
        headers={
            "Authorization": f"Bearer {backend_token}",
            "Idempotency-Key": "customer-request-001",
        },
        json=body,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "image-gen"
    assert payload["data"] == [
        {"url": "https://public.example/images/one.png"},
        {"url": "https://public.example/images/two.png"},
    ]
    assert "image-pipeline-private" not in response.text
    assert calls == [
        {
            "prompt": "draw a clean city park",
            "size": "2k",
            "count": 2,
            "concurrency": 2,
            "tenant_scope": "opaque-tenant-scope",
            "idempotency_key": "customer-request-001",
        }
    ]


def test_async_gateway_wait_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NeverDoneStore:
        def get_batch(self, _batch_id: str) -> list[dict[str, Any]]:
            time.sleep(0.01)
            return [{"status": "awaiting_upscale"}]

    monkeypatch.setattr(service, "store", NeverDoneStore())

    async def exercise() -> None:
        waiter = asyncio.create_task(service._wait_for_terminal_batch("batch", 1))
        ticks = 0
        started = time.monotonic()
        while time.monotonic() - started < 0.15:
            await asyncio.sleep(0.01)
            ticks += 1
        assert ticks >= 8
        with pytest.raises(HTTPException) as error:
            await waiter
        assert error.value.status_code == 504

    asyncio.run(exercise())


def test_operational_stats_report_queue_and_today_success_rate(tmp_path: Any) -> None:
    store = service.TaskStore(tmp_path / "stats.db")
    terminal = store.create_batch("p", "2k", 2, 2)
    queued = store.create("q", "4k")
    now = time.time()
    store.update(terminal[0]["task_id"], status="done", completed_at=now)
    store.update(terminal[1]["task_id"], status="failed", completed_at=now)

    stats = store.operational_stats(now=now)
    assert stats["queue_length"] == 1
    assert stats["done_total"] == 1
    assert stats["failed_total"] == 1
    assert stats["today_done"] == 1
    assert stats["today_failed"] == 1
    assert stats["today_success_rate"] == 0.5
    assert (store.get(queued["task_id"]) or {})["status"] == "queued"


def _install_litellm_stub() -> None:
    if "litellm.integrations.custom_logger" in sys.modules:
        return
    litellm = ModuleType("litellm")
    integrations = ModuleType("litellm.integrations")
    custom_logger = ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:
        pass

    custom_logger.CustomLogger = CustomLogger  # type: ignore[attr-defined]
    sys.modules["litellm"] = litellm
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger


def test_gateway_hook_hides_private_model_and_forwards_opaque_identity() -> None:
    _install_litellm_stub()
    from gateway.hook import ImageGatewayHook

    hook = ImageGatewayHook()
    raw_virtual_key = "virtual-key-that-must-not-be-forwarded"
    data = {
        "model": "image-gen",
        "prompt": "p",
        "proxy_server_request": {
            "headers": {"Idempotency-Key": "customer-request-001"}
        },
    }
    rewritten = asyncio.run(
        hook.async_pre_call_hook(
            SimpleNamespace(team_id="team-a", api_key=raw_virtual_key),
            None,
            data,
            "image_generation",
        )
    )

    assert rewritten["model"] == "image-pipeline-private"
    assert len(rewritten["user"]) == 32
    assert raw_virtual_key not in str(rewritten)
    assert rewritten["extra_headers"]["Idempotency-Key"] == "customer-request-001"

    second_key = asyncio.run(
        hook.async_pre_call_hook(
            SimpleNamespace(team_id="team-a", api_key="another-virtual-key"),
            None,
            {"model": "image-gen"},
            "image_generation",
        )
    )
    assert second_key["user"] != rewritten["user"]

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            hook.async_pre_call_hook(
                SimpleNamespace(team_id="team-a"),
                None,
                {"model": "image-pipeline-private"},
                "image_generation",
            )
        )
    assert error.value.status_code == 403

    with pytest.raises(HTTPException) as wrong_endpoint:
        asyncio.run(
            hook.async_pre_call_hook(
                SimpleNamespace(api_key="virtual-key"),
                None,
                {"model": "image-gen"},
                "chat_completion",
            )
        )
    assert wrong_endpoint.value.status_code == 403


def test_public_health_and_openapi_do_not_expose_private_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "private-upstream-model")
    client = TestClient(service.app)
    health = client.get("/health")
    schema = client.get("/openapi.json")

    assert health.status_code == 200
    assert "model" not in health.json()
    assert "private-upstream-model" not in health.text
    assert "private-upstream-model" not in schema.text
    assert "gpt-image-2" not in schema.text
