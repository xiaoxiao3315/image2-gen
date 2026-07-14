from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from PIL import Image

import image_pipeline.generator as generator_module
from image_pipeline.generator import GptImageGenerator, ImageGenerationError
from image_pipeline.service import TaskStore
from image_pipeline.telemetry import TelemetryStore, normalize_error
from scripts.cleanup_service_data import cleanup


def _png_payload() -> dict[str, Any]:
    output = BytesIO()
    Image.new("RGB", (8, 6), "navy").save(output, format="PNG")
    return {"data": [{"b64_json": base64.b64encode(output.getvalue()).decode()}]}


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        request_id: str | None = None,
        json_error: bool = False,
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self._json_error = json_error
        self.headers = {} if request_id is None else {"x-request-id": request_id}
        self._content = b"payload"

    def json(self) -> dict[str, Any]:
        if self._json_error:
            raise ValueError("malformed")
        return self._payload


class SequenceSession:
    def __init__(self, outcomes: list[FakeResponse | Exception]):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def post(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
        raise requests.ConnectionError("download failed with secret text")


def _generator(outcomes: list[FakeResponse | Exception]) -> GptImageGenerator:
    settings = SimpleNamespace(
        api_proxy=None,
        api_key="unit-test-secret-key",
        api_base_url="https://provider.invalid/v1",
        api_connect_timeout_seconds=1,
        api_timeout_seconds=2,
        model="test-model",
        source_size="1536x1024",
    )
    generator = GptImageGenerator(settings)
    generator.session = SequenceSession(outcomes)  # type: ignore[assignment]
    return generator


def _recording_observer(events: list[tuple[str, dict[str, Any]]]):
    def observe(event: str, payload: dict[str, Any]) -> None:
        events.append((event, dict(payload)))

    return observe


def test_schema_is_additive_and_preserves_existing_task(tmp_path: Path) -> None:
    database = tmp_path / "tasks.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE tasks(task_id TEXT PRIMARY KEY,prompt TEXT)")
        connection.execute("INSERT INTO tasks VALUES('kept','private prompt')")

    telemetry = TelemetryStore(database)

    assert telemetry.enabled
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT task_id FROM tasks").fetchone()[0] == "kept"
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"task_events", "generation_attempts"} <= tables


def test_best_effort_write_lock_returns_quickly(tmp_path: Path) -> None:
    database = tmp_path / "tasks.db"
    telemetry = TelemetryStore(database, timeout_seconds=0.02)
    blocker = sqlite3.connect(database, timeout=1)
    blocker.execute("BEGIN IMMEDIATE")
    started = time.perf_counter()
    try:
        assert not telemetry.try_event("task", "accepted")
    finally:
        blocker.rollback()
        blocker.close()
    assert time.perf_counter() - started < 0.5
    assert telemetry.failure_count >= 1


def test_503_then_success_reports_one_finished_event_per_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(generator_module.time, "sleep", lambda _seconds: None)
    events: list[tuple[str, dict[str, Any]]] = []
    generator = _generator(
        [
            FakeResponse(503, request_id="retry-1"),
            FakeResponse(200, _png_payload(), request_id="success-2"),
        ]
    )

    result = generator.generate(
        "private prompt",
        "low",
        tmp_path,
        attempt_observer=_recording_observer(events),
    )

    finished = [payload for event, payload in events if event == "finished"]
    assert len(finished) == 2
    assert finished[0]["http_status"] == 503
    assert finished[0]["will_retry"] is True
    assert finished[0]["backoff_seconds"] == 1.0
    assert finished[0]["provider_request_id"] == "retry-1"
    assert finished[1]["outcome"] == "success"
    assert finished[1]["provider_request_id"] == "success-2"
    assert result.request_id == "success-2"
    assert "private prompt" not in repr(events)


def test_read_timeout_then_success_and_three_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(generator_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "2")
    events: list[tuple[str, dict[str, Any]]] = []
    _generator(
        [requests.ReadTimeout("sensitive"), FakeResponse(200, _png_payload())]
    ).generate(
        "p", "low", tmp_path / "success", attempt_observer=_recording_observer(events)
    )
    first = next(payload for event, payload in events if event == "finished")
    assert normalize_error(first["error"]) == "read_timeout"

    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "3")
    failed_events: list[tuple[str, dict[str, Any]]] = []
    with pytest.raises(ImageGenerationError):
        _generator(
            [
                requests.ConnectTimeout("one"),
                requests.ReadTimeout("two"),
                requests.ConnectionError("three"),
            ]
        ).generate(
            "p",
            "low",
            tmp_path / "failed",
            attempt_observer=_recording_observer(failed_events),
        )
    assert len([1 for event, _ in failed_events if event == "finished"]) == 3


def test_normalize_error_uses_only_explicit_cause() -> None:
    explicit_cause = requests.ReadTimeout("explicit timeout")
    try:
        raise ImageGenerationError("request failed") from explicit_cause
    except ImageGenerationError as wrapped:
        assert wrapped.__cause__ is explicit_cause
        assert normalize_error(wrapped) == "read_timeout"


def test_normalize_error_ignores_implicit_context() -> None:
    try:
        raise requests.ReadTimeout("unrelated earlier timeout")
    except requests.ReadTimeout as unrelated:
        try:
            raise ImageGenerationError("current failure")
        except ImageGenerationError as current:
            assert current.__cause__ is None
            assert current.__context__ is unrelated
            assert normalize_error(current) == "unknown"


def test_normalize_error_limits_explicit_cause_to_two_hops() -> None:
    outer = ImageGenerationError("outer")
    middle = ImageGenerationError("middle")
    timeout = requests.ReadTimeout("explicit timeout")
    outer.__cause__ = middle
    middle.__cause__ = timeout
    assert normalize_error(outer) == "read_timeout"

    inner = ImageGenerationError("inner")
    middle.__cause__ = inner
    inner.__cause__ = timeout
    assert normalize_error(outer) == "unknown"

    middle.__cause__ = outer
    assert normalize_error(outer) == "unknown"


def test_task_manager_terminal_failure_keeps_wrapped_timeout_category(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "unit-test-key")
    monkeypatch.setenv("IMAGE_API_BASE_URL", "https://provider.invalid/v1")
    store = TaskStore(tmp_path / "manager.db")
    manager = __import__(
        "image_pipeline.service", fromlist=["TaskManager"]
    ).TaskManager(store, source_dir=tmp_path / "sources")
    task = store.create("p", "2k")
    store.update(task["task_id"], status="processing")
    try:
        raise ImageGenerationError("request failed") from requests.ReadTimeout(
            "sensitive timeout"
        )
    except ImageGenerationError as wrapped:
        manager._fail_task(task["task_id"], wrapped)

    timeline = store.telemetry.task_timeline(task["task_id"])
    terminal = [
        event for event in timeline["events"] if event["event_type"] == "terminal_failed"
    ]
    assert terminal[-1]["details"]["category"] == "read_timeout"


def test_malformed_response_and_download_failure_are_classifiable(tmp_path: Path) -> None:
    malformed_events: list[tuple[str, dict[str, Any]]] = []
    with pytest.raises(ImageGenerationError, match="non-JSON"):
        _generator([FakeResponse(200, request_id="malformed-id", json_error=True)]).generate(
            "p",
            "low",
            tmp_path / "malformed",
            attempt_observer=_recording_observer(malformed_events),
        )
    malformed = [payload for event, payload in malformed_events if event == "finished"]
    assert len(malformed) == 1
    assert malformed[0]["phase"] == "json"
    assert malformed[0]["provider_request_id"] == "malformed-id"

    download_events: list[tuple[str, dict[str, Any]]] = []
    with pytest.raises(ImageGenerationError, match="could not be downloaded"):
        _generator(
            [
                FakeResponse(
                    200,
                    {"data": [{"url": "https://download.invalid/image"}]},
                    request_id="post-id",
                )
            ]
        ).generate(
            "p",
            "low",
            tmp_path / "download",
            attempt_observer=_recording_observer(download_events),
        )
    download = [payload for event, payload in download_events if event == "finished"]
    assert len(download) == 1
    assert download[0]["phase"] == "image_download"


def test_native_batch_is_one_physical_attempt_and_observer_failure_isolated(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    single = _png_payload()["data"][0]
    payload = {"data": [dict(single) for _ in range(3)]}
    generator = _generator([FakeResponse(200, payload)])
    results = generator.generate_many(
        "p", "low", tmp_path / "native", n=3, attempt_observer=_recording_observer(events)
    )
    assert len(results) == 3
    assert len([1 for event, _ in events if event == "finished"]) == 1

    def broken_observer(_event: str, _payload: dict[str, Any]) -> None:
        raise RuntimeError("telemetry unavailable")

    result = _generator([FakeResponse(200, _png_payload())]).generate(
        "p", "low", tmp_path / "broken", attempt_observer=broken_observer
    )
    assert result.image.width == 8


def test_taskstore_transition_survives_telemetry_failure(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.db")

    def fail(*_args: Any, **_kwargs: Any) -> sqlite3.Connection:
        raise sqlite3.OperationalError("telemetry sink failed")

    store.telemetry._connect = fail  # type: ignore[method-assign]
    task = store.create("p", "2k")

    assert (store.get(task["task_id"]) or {})["status"] == "queued"
    assert store.telemetry.failure_count >= 2


def test_task_lifecycle_queries_slow_upscale_and_security(tmp_path: Path) -> None:
    database = tmp_path / "tasks.db"
    store = TaskStore(database)
    task = store.create("TOP SECRET PROMPT", "2k")
    task_id = task["task_id"]
    base = task["created_at"]
    store.telemetry.try_start_attempt(
        task_ids=[task_id],
        batch_id=task["batch_id"],
        attempt_no=1,
        request_kind="single",
        requested_n=1,
        started_at=base + 10,
        route_label="private-route",
    )
    attempt_id = store.telemetry.try_start_attempt(
        task_ids=[task_id],
        batch_id=task["batch_id"],
        attempt_no=2,
        request_kind="single",
        requested_n=1,
        started_at=base + 20,
        route_label="private-route",
    )
    assert store.telemetry.try_finish_attempt(
        attempt_id,
        finished_at=base + 22,
        duration_seconds=2,
        http_status=200,
        outcome="success",
        error_category=None,
        will_retry=False,
        backoff_seconds=None,
        error_summary=None,
        provider_request_id="request-safe-id",
    )
    store.telemetry.try_event(task_id, "upscale_queued", occurred_at=base + 30)
    store.telemetry.try_event(
        task_id, "upscale_started", occurred_at=base + 40, duration_seconds=10
    )
    store.telemetry.try_event(
        task_id, "upscale_finished", occurred_at=base + 150, duration_seconds=110
    )
    store.telemetry.try_event(
        task_id, "delivery_completed", occurred_at=base + 151, duration_seconds=151
    )
    store.update(task_id, status="done", started_at=base + 10, completed_at=base + 151)

    timeline = store.telemetry.task_timeline(task_id)
    encoded = json.dumps(timeline, ensure_ascii=False)
    assert "TOP SECRET PROMPT" not in encoded
    assert "private-route" not in encoded
    assert "request-safe-id" not in encoded
    assert len(timeline["generation_attempts"][1]["provider_request_id"]) == 64
    assert timeline["task"]["status"] == "done"
    stats = store.telemetry.window_stats(task["created_at"] - 1, task["created_at"] + 300)
    assert stats["latency_seconds"]["upscale_queue"]["p95"] == 10
    assert stats["latency_seconds"]["upscale_execution"]["p95"] == 110


def test_restart_recovery_appends_event_and_closes_open_attempt(tmp_path: Path) -> None:
    database = tmp_path / "tasks.db"
    store = TaskStore(database)
    task = store.create("p", "2k")
    untouched = store.create("q", "2k")
    store.update(task["task_id"], status="processing", started_at=time.time() - 5)
    attempt_id = store.telemetry.try_start_attempt(
        task_ids=[task["task_id"]],
        batch_id=task["batch_id"],
        attempt_no=1,
        request_kind="single",
        requested_n=1,
        started_at=time.time() - 4,
    )
    assert attempt_id

    restarted = TaskStore(database)
    row = restarted.get(task["task_id"])
    assert row and row["status"] == "queued"
    timeline = restarted.telemetry.task_timeline(task["task_id"])
    assert timeline["generation_attempts"][0]["outcome"] == "interrupted"
    assert timeline["generation_attempts"][0]["error_category"] == "unknown"
    recovery = [
        event
        for event in timeline["events"]
        if event["event_type"] == "generation_queued"
        and event["details"].get("reason") == "restart_recovery"
    ]
    assert len(recovery) == 1

    untouched_timeline = restarted.telemetry.task_timeline(untouched["task_id"])
    assert not [
        event
        for event in untouched_timeline["events"]
        if event["event_type"] == "generation_queued"
        and event["details"].get("reason") == "restart_recovery"
    ]

    reopened = TaskStore(database)
    reopened_timeline = reopened.telemetry.task_timeline(task["task_id"])
    reopened_recovery = [
        event
        for event in reopened_timeline["events"]
        if event["event_type"] == "generation_queued"
        and event["details"].get("reason") == "restart_recovery"
    ]
    assert len(reopened_recovery) == 1


def test_reconcile_open_attempt_after_task_advanced(tmp_path: Path) -> None:
    database = tmp_path / "advanced.db"
    store = TaskStore(database)
    task = store.create("p", "2k")
    attempt_id = store.telemetry.try_start_attempt(
        task_ids=[task["task_id"]],
        batch_id=task["batch_id"],
        attempt_no=1,
        request_kind="single",
        requested_n=1,
        started_at=time.time() - 4,
    )
    assert attempt_id
    store.update(task["task_id"], status="awaiting_upscale")

    restarted = TaskStore(database)
    timeline = restarted.telemetry.task_timeline(task["task_id"])

    assert timeline["generation_attempts"][0]["outcome"] == "telemetry_interrupted"
    assert timeline["generation_attempts"][0]["finished_at"] is not None


def test_window_stats_attributes_attempt_to_finish_window(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "attempt-window.db")
    task = store.create("p", "2k")
    attempt_id = store.telemetry.try_start_attempt(
        task_ids=[task["task_id"]],
        batch_id=task["batch_id"],
        attempt_no=1,
        request_kind="single",
        requested_n=1,
        started_at=9,
    )
    assert attempt_id
    assert store.telemetry.try_finish_attempt(
        attempt_id,
        finished_at=11,
        duration_seconds=2,
        http_status=503,
        outcome="failed",
        error_category="http_503",
        will_retry=False,
        backoff_seconds=None,
        error_summary="http_503",
        provider_request_id=None,
    )

    first = store.telemetry.window_stats(0, 10)
    second = store.telemetry.window_stats(10, 20)

    assert first["latency_seconds"]["generation_attempt"]["count"] == 0
    assert second["latency_seconds"]["generation_attempt"]["p95"] == 2
    assert second["failure_categories"] == {"http_503": 1}


def test_restart_recovery_survives_telemetry_insert_failure(tmp_path: Path) -> None:
    database = tmp_path / "recovery-telemetry-failure.db"
    store = TaskStore(database)
    task = store.create("p", "2k")
    store.update(task["task_id"], status="processing", started_at=time.time() - 2)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TRIGGER fail_recovery_event BEFORE INSERT ON task_events "
            "BEGIN SELECT RAISE(FAIL, 'telemetry unavailable'); END"
        )

    restarted = TaskStore(database)

    row = restarted.get(task["task_id"])
    assert row and row["status"] == "queued"
    assert restarted.telemetry.failure_count >= 1


def test_cleanup_removes_task_telemetry_rows(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "sources").mkdir()
    store = TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "2k")
    store.update(task["task_id"], status="failed", completed_at=10)
    store.telemetry.try_event(task["task_id"], "terminal_failed", occurred_at=10)
    store.telemetry.try_start_attempt(
        task_ids=[task["task_id"]],
        batch_id=task["batch_id"],
        attempt_no=1,
        request_kind="single",
        requested_n=1,
        started_at=2,
    )

    assert cleanup(tmp_path, 7, now=1_000_000) == 1
    with sqlite3.connect(tmp_path / "tasks.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM generation_attempts").fetchone()[0] == 0


def test_concurrent_append_has_no_lost_rows(tmp_path: Path) -> None:
    telemetry = TelemetryStore(tmp_path / "tasks.db", timeout_seconds=1)
    threads = [
        threading.Thread(
            target=lambda index=index: telemetry.try_event(
                f"task-{index}", "accepted", occurred_at=float(index)
            )
        )
        for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with sqlite3.connect(tmp_path / "tasks.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 8


def test_window_stats_excludes_completion_after_until(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "2k")
    since = task["created_at"] - 1
    until = task["created_at"] + 10
    store.update(
        task["task_id"],
        status="done",
        started_at=task["created_at"] + 2,
        completed_at=task["created_at"] + 20,
    )

    stats = store.telemetry.window_stats(since, until)

    assert stats["requests"] == 1
    assert stats["delivered"] == 0
    assert stats["throughput_delivered_per_hour"] == 0
    assert stats["latency_seconds"]["end_to_end"]["count"] == 0


def test_window_stats_counts_delivery_for_pre_window_arrival(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "2k")
    since = task["created_at"] + 10
    until = task["created_at"] + 30
    store.update(
        task["task_id"],
        status="done",
        started_at=task["created_at"] + 2,
        completed_at=task["created_at"] + 20,
    )

    stats = store.telemetry.window_stats(since, until)

    assert stats["requests"] == 0
    assert stats["delivered"] == 1
    assert stats["throughput_delivered_per_hour"] > 0
    assert stats["latency_seconds"]["end_to_end"]["p95"] == 20


def test_window_stats_excludes_generation_start_after_until(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "2k")
    since = task["created_at"] - 1
    until = task["created_at"] + 10
    store.update(task["task_id"], started_at=task["created_at"] + 20)

    stats = store.telemetry.window_stats(since, until)

    assert stats["requests"] == 1
    assert stats["latency_seconds"]["generation_queue"]["count"] == 0


def test_window_stats_includes_carry_in_generation_queue_and_backlog(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "2k")
    since = task["created_at"] + 10
    until = task["created_at"] + 30
    store.update(task["task_id"], started_at=task["created_at"] + 20)
    store.telemetry.try_event(
        task["task_id"],
        "generation_started",
        occurred_at=task["created_at"] + 20,
        duration_seconds=20,
    )

    stats = store.telemetry.window_stats(since, until)

    assert stats["requests"] == 0
    assert stats["latency_seconds"]["generation_queue"]["p95"] == 20

    queued = store.create("q", "2k")
    backlog_stats = store.telemetry.window_stats(queued["created_at"] + 10, queued["created_at"] + 20)
    assert backlog_stats["oldest_queued_age_seconds"] >= 20


def test_closed_window_queue_stats_survive_restart_recovery(tmp_path: Path) -> None:
    database = tmp_path / "historical-restart.db"
    store = TaskStore(database)
    task = store.create("p", "2k")
    base = task["created_at"]
    store.update(task["task_id"], status="processing", started_at=base + 5)
    store.telemetry.try_event(
        task["task_id"],
        "generation_started",
        occurred_at=base + 5,
        duration_seconds=5,
    )
    before = store.telemetry.window_stats(base, base + 10)

    restarted = TaskStore(database)
    after = restarted.telemetry.window_stats(base, base + 10)

    assert before["latency_seconds"]["generation_queue"] == after["latency_seconds"][
        "generation_queue"
    ]
    assert before["oldest_queued_age_seconds"] == after["oldest_queued_age_seconds"]


def test_expired_upscale_lease_closes_attempt_and_requeues(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "2k")
    store.update(task["task_id"], status="awaiting_upscale", source_filename="source.png")
    first = store.claim_upscale(lease_seconds=10, worker_id="gpu-a", now=100)
    assert first is not None

    second = store.claim_upscale(lease_seconds=10, worker_id="gpu-b", now=110)

    assert second is not None
    assert second["task_id"] == task["task_id"]
    timeline = store.telemetry.task_timeline(task["task_id"])
    expired = [
        event
        for event in timeline["events"]
        if event["event_type"] == "upscale_finished"
        and event["details"].get("outcome") == "lease_expired"
    ]
    recovered_queue = [
        event
        for event in timeline["events"]
        if event["event_type"] == "upscale_queued"
        and event["details"].get("reason") == "lease_expired"
    ]
    starts = [event for event in timeline["events"] if event["event_type"] == "upscale_started"]
    assert len(expired) == 1
    assert len(recovered_queue) == 1
    assert len(starts) == 2
    assert starts[-1]["duration_seconds"] == 0


def test_retry_start_after_transient_initialization_failure(tmp_path: Path) -> None:
    database = tmp_path / "retry-init.db"
    blocker = sqlite3.connect(database, timeout=1)
    blocker.execute("BEGIN IMMEDIATE")
    telemetry = TelemetryStore(database, timeout_seconds=0.01)
    assert not telemetry.enabled
    blocker.rollback()
    blocker.close()

    assert telemetry.try_event("task", "accepted")
    assert telemetry.enabled


def test_oversized_details_remain_valid_json(tmp_path: Path) -> None:
    database = tmp_path / "details.db"
    telemetry = TelemetryStore(database)
    details = {f"k{index:02d}": "a" * 120 for index in range(20)}

    assert telemetry.try_event("task", "accepted", details=details)

    with sqlite3.connect(database) as connection:
        stored = connection.execute("SELECT details_json FROM task_events").fetchone()[0]
    decoded = json.loads(stored)
    assert isinstance(decoded, dict)
    assert len(stored) <= 2000


def test_read_only_cli_store_handles_pretelemetry_database(tmp_path: Path) -> None:
    database = tmp_path / "old.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE tasks(task_id TEXT PRIMARY KEY,batch_id TEXT,batch_index INTEGER,"
            "batch_size INTEGER,size TEXT,status TEXT,created_at REAL,started_at REAL,"
            "completed_at REAL,upscale_attempts INTEGER)"
        )
        connection.execute(
            "INSERT INTO tasks VALUES('old','old',0,1,'2k','queued',1,NULL,NULL,0)"
        )

    telemetry = TelemetryStore(database, initialize=False)
    timeline = telemetry.task_timeline("old")
    stats = telemetry.window_stats(0, 2)

    assert timeline["events"] == []
    assert timeline["generation_attempts"] == []
    assert stats["requests"] == 1


def test_additive_migration_extends_partial_telemetry_tables(tmp_path: Path) -> None:
    database = tmp_path / "partial.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE task_events(event_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO task_events(event_id) VALUES('kept-event')")
        connection.execute("CREATE TABLE generation_attempts(attempt_id TEXT PRIMARY KEY)")
        connection.execute(
            "INSERT INTO generation_attempts(attempt_id) VALUES('kept-attempt')"
        )

    telemetry = TelemetryStore(database)

    assert telemetry.enabled
    with sqlite3.connect(database) as connection:
        event_columns = {row[1] for row in connection.execute("PRAGMA table_info(task_events)")}
        attempt_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(generation_attempts)")
        }
        assert connection.execute("SELECT event_id FROM task_events").fetchone()[0] == "kept-event"
        assert connection.execute("SELECT attempt_id FROM generation_attempts").fetchone()[0] == "kept-attempt"
    assert {"task_id", "event_type", "occurred_at", "details_json"} <= event_columns
    assert {"task_ids_json", "attempt_no", "requested_n", "started_at"} <= attempt_columns
