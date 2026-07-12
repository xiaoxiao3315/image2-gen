from __future__ import annotations

import base64
import json
import threading
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import image_pipeline.service as service
from image_pipeline.generator import GptImageGenerator, NativeBatchUnsupported
from image_pipeline.models import GenerationResult


def _png_bytes(size: tuple[int, int] = (12, 8), color: str = "navy") -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _result(path: Path, quality: str = "low") -> GenerationResult:
    metadata = service.inspect_image(path)
    return GenerationResult(
        requested_model="test-model",
        requested_quality=quality,
        requested_size="1536x1024",
        request_body={},
        status_code=200,
        request_id=None,
        api_seconds=0.1,
        download_seconds=0.01,
        total_seconds=0.11,
        response_image_trail="test",
        usage=None,
        image=metadata,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("count", 0), ("count", 6), ("concurrency", 0), ("concurrency", 6)],
)
def test_batch_request_rejects_values_outside_hard_limit(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        service.GenerateTaskRequest(prompt="p", **{field: value})


def test_generation_worker_default_and_hard_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("IMAGE_GENERATION_WORKERS", raising=False)
    manager = service.TaskManager(
        service.TaskStore(tmp_path / "default.db"),
        source_dir=tmp_path / "sources-default",
    )
    assert manager.worker_count == 3

    monkeypatch.setenv("IMAGE_GENERATION_WORKERS", "5")
    manager = service.TaskManager(
        service.TaskStore(tmp_path / "five.db"),
        source_dir=tmp_path / "sources-five",
    )
    assert manager.worker_count == 5

    monkeypatch.setenv("IMAGE_GENERATION_WORKERS", "6")
    with pytest.raises(RuntimeError, match="from 1 to 5"):
        service.TaskManager(
            service.TaskStore(tmp_path / "too-many.db"),
            source_dir=tmp_path / "sources-too-many",
        )


def test_single_request_compatibility_and_batch_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeManager:
        def submit_batch(
            self, _prompt: str, _size: str, count: int, requested_concurrency: int
        ) -> list[dict[str, Any]]:
            return [
                {
                    "task_id": f"task-{index}",
                    "batch_id": "batch-test",
                    "status": "queued",
                }
                for index in range(count)
            ]

    monkeypatch.setattr(service, "manager", FakeManager())
    client = TestClient(service.app)
    single = client.post("/v1/generate", json={"prompt": "p", "size": "2k"})
    assert single.status_code == 202
    assert single.json()["task_id"] == "task-0"
    assert single.json()["result_url"].endswith("/v1/result/task-0")
    assert single.json()["task_ids"] == ["task-0"]

    batch = client.post(
        "/v1/generate",
        json={"prompt": "p", "size": "4k", "count": 5, "concurrency": 2},
    )
    assert batch.status_code == 202
    payload = batch.json()
    assert payload["batch_id"] == "batch-test"
    assert payload["count"] == 5
    assert payload["concurrency"] == 2
    assert len(payload["task_ids"]) == len(payload["result_urls"]) == 5
    assert payload["batch_result_url"].endswith("/v1/batch/batch-test")
    assert "task_id" not in payload


def test_batch_store_migration_grouping_and_aggregate_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = service.TaskStore(tmp_path / "tasks.db")
    records = store.create_batch("same prompt", "2k", 3, requested_concurrency=2)
    assert len({record["batch_id"] for record in records}) == 1
    assert [record["batch_index"] for record in records] == [0, 1, 2]
    store.update(records[0]["task_id"], status="done", image_filename="a.png", width=2048,
                 height=2048, file_bytes=1, sha256="a", metrics_json="{}")
    store.update(records[1]["task_id"], status="failed", error="safe error")
    monkeypatch.setattr(service, "store", store)

    response = TestClient(service.app).get(f"/v1/batch/{records[0]['batch_id']}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "processing"
    assert payload["summary"] == {"done": 1, "failed": 1, "queued": 1}
    assert [item["batch_index"] for item in payload["results"]] == [0, 1, 2]

    # A restart restores processing children and enqueues the group only once.
    store.update(records[2]["task_id"], status="processing")
    restarted = service.TaskStore(tmp_path / "tasks.db")
    assert restarted.queued_batch_ids() == [records[0]["batch_id"]]


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload
        self._content = b"encoded response"

    def json(self) -> dict[str, Any]:
        return self._payload


class _SingleResponseSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, _endpoint: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return self.response


def _generator(response: _FakeResponse) -> tuple[GptImageGenerator, _SingleResponseSession]:
    settings = SimpleNamespace(
        api_proxy=None,
        api_key="unit-test-key",
        api_base_url="https://example.test/v1",
        api_connect_timeout_seconds=1,
        api_timeout_seconds=2,
        model="gpt-image-2",
        source_size="1536x1024",
    )
    generator = GptImageGenerator(settings)
    session = _SingleResponseSession(response)
    generator.session = session  # type: ignore[assignment]
    return generator, session


def test_generator_native_n_writes_each_candidate(tmp_path: Path) -> None:
    encoded = base64.b64encode(_png_bytes()).decode()
    response = _FakeResponse(
        200,
        {"data": [{"b64_json": encoded}, {"b64_json": encoded}, {"b64_json": encoded}]},
    )
    generator, session = _generator(response)
    results = generator.generate_many("p", "low", tmp_path, n=3)

    assert session.calls[0]["json"]["n"] == 3
    assert len(results) == 3
    assert all(Path(result.image.path).is_file() for result in results)
    assert response._content == b""


def test_generator_only_classifies_explicit_400_422_as_native_unsupported(
    tmp_path: Path,
) -> None:
    generator, _ = _generator(
        _FakeResponse(422, {"error": {"message": "parameter n is not supported; only n=1"}})
    )
    with pytest.raises(NativeBatchUnsupported):
        generator.generate_many("p", "low", tmp_path, n=5)


def test_unsupported_native_batch_falls_back_with_requested_concurrency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "unit-test-key")
    monkeypatch.setenv("IMAGE_API_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("IMAGE_GENERATION_WORKERS", "3")
    active = 0
    peak = 0
    lock = threading.Lock()

    class FakeGenerator:
        def generate_many(self, *_args: Any, **_kwargs: Any) -> list[GenerationResult]:
            raise NativeBatchUnsupported("n unsupported")

        def generate(self, _prompt: str, quality: str, output_dir: Path) -> GenerationResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.04)
                output_dir.mkdir(parents=True)
                path = output_dir / "source.png"
                path.write_bytes(_png_bytes())
                return _result(path, quality)
            finally:
                with lock:
                    active -= 1

    store = service.TaskStore(tmp_path / "fallback.db")
    limiter = service.WeightedLimiter(5)
    manager = service.TaskManager(
        store,
        source_dir=tmp_path / "sources",
        generator_factory=lambda _: FakeGenerator(),
        generation_limiter=limiter,
    )
    manager.start()
    tasks = manager.submit_batch("p", "2k", count=5, requested_concurrency=2)
    manager.queue.join()
    manager.stop()

    assert peak == 2
    assert limiter.peak == 5  # the initial native n=5 request reserves five logical slots
    rows = [store.get(task["task_id"]) for task in tasks]
    assert all(row and row["status"] == "awaiting_upscale" for row in rows)
    assert {
        json.loads(row["metrics_json"])["generation_mode"] for row in rows if row
    } == {"single_fallback"}


def test_fallback_cannot_exceed_configured_generation_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "unit-test-key")
    monkeypatch.setenv("IMAGE_API_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("IMAGE_GENERATION_WORKERS", "2")
    active = 0
    peak = 0
    lock = threading.Lock()

    class FakeGenerator:
        def generate_many(self, *_args: Any, **_kwargs: Any) -> list[GenerationResult]:
            raise NativeBatchUnsupported("n unsupported")

        def generate(self, _prompt: str, quality: str, output_dir: Path) -> GenerationResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.03)
                output_dir.mkdir(parents=True)
                path = output_dir / "source.png"
                path.write_bytes(_png_bytes())
                return _result(path, quality)
            finally:
                with lock:
                    active -= 1

    manager = service.TaskManager(
        service.TaskStore(tmp_path / "worker-cap.db"),
        source_dir=tmp_path / "sources-cap",
        generator_factory=lambda _: FakeGenerator(),
        generation_limiter=service.WeightedLimiter(5),
    )
    manager.start()
    manager.submit_batch("p", "2k", count=5, requested_concurrency=5)
    manager.queue.join()
    manager.stop()
    assert peak == 2


def test_native_batches_share_global_five_slot_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "unit-test-key")
    monkeypatch.setenv("IMAGE_API_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("IMAGE_GENERATION_WORKERS", "2")
    active_weight = 0
    observed_peak = 0
    lock = threading.Lock()

    class FakeGenerator:
        def generate_many(
            self, _prompt: str, quality: str, output_dir: Path, n: int
        ) -> list[GenerationResult]:
            nonlocal active_weight, observed_peak
            with lock:
                active_weight += n
                observed_peak = max(observed_peak, active_weight)
            try:
                time.sleep(0.04)
                output_dir.mkdir(parents=True)
                results = []
                for index in range(n):
                    path = output_dir / f"source-{index}.png"
                    path.write_bytes(_png_bytes(color="green"))
                    results.append(_result(path, quality))
                return results
            finally:
                with lock:
                    active_weight -= n

    store = service.TaskStore(tmp_path / "native.db")
    limiter = service.WeightedLimiter(5)
    manager = service.TaskManager(
        store,
        source_dir=tmp_path / "sources-native",
        generator_factory=lambda _: FakeGenerator(),
        generation_limiter=limiter,
    )
    manager.start()
    first = manager.submit_batch("p1", "2k", count=3, requested_concurrency=3)
    second = manager.submit_batch("p2", "2k", count=3, requested_concurrency=3)
    manager.queue.join()
    manager.stop()

    assert observed_peak <= 5
    assert limiter.peak <= 5
    for task in first + second:
        row = store.get(task["task_id"])
        assert row and row["status"] == "awaiting_upscale"
        assert json.loads(row["metrics_json"])["generation_mode"] == "native_n"
