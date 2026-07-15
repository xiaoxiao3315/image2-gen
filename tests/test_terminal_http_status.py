from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

import image_pipeline.generator as generator_module
from image_pipeline.generator import (
    GptImageGenerator,
    ImageGenerationError,
    NativeBatchIncomplete,
    NativeBatchUnsupported,
)
from image_pipeline.service import TaskManager, TaskStore


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload or {}
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


class SequenceSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def post(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
        self.calls += 1
        return next(self.responses)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        api_proxy=None,
        api_key="unit-test-key",
        api_base_url="https://provider.invalid/v1",
        api_connect_timeout_seconds=1,
        api_timeout_seconds=2,
        model="test-model",
        source_size="1536x1024",
    )


def _generator(responses: list[FakeResponse]) -> tuple[GptImageGenerator, SequenceSession]:
    generator = GptImageGenerator(_settings())
    session = SequenceSession(responses)
    generator.session = session  # type: ignore[assignment]
    return generator, session


def test_generation_error_metadata_is_backward_compatible() -> None:
    plain = ImageGenerationError("plain failure")
    assert str(plain) == "plain failure"
    assert plain.http_status is None
    assert plain.phase is None

    structured = ImageGenerationError(
        "provider failure", http_status=503, phase="request"
    )
    assert str(structured) == "provider failure"
    assert structured.http_status == 503
    assert structured.phase == "request"


def test_native_batch_exception_constructors_remain_compatible() -> None:
    incomplete = NativeBatchIncomplete("partial", [], 3)
    assert incomplete.partial_results == []
    assert incomplete.requested_count == 3
    assert incomplete.http_status is None
    assert incomplete.phase is None

    structured = NativeBatchIncomplete(
        "partial", [], 3, http_status=200, phase="candidate"
    )
    assert structured.http_status == 200
    assert structured.phase == "candidate"

    unsupported = NativeBatchUnsupported(
        "unsupported", http_status=422, phase="request"
    )
    assert isinstance(unsupported, ImageGenerationError)
    assert unsupported.http_status == 422
    assert unsupported.phase == "request"


@pytest.mark.parametrize("status", [429, 401, 500])
def test_final_nonretryable_http_error_carries_structured_metadata(
    status: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(generator_module.time, "sleep", sleeps.append)
    generator, session = _generator([FakeResponse(status)])

    with pytest.raises(ImageGenerationError) as caught:
        generator.generate("p", "low", tmp_path)

    assert session.calls == 1
    assert sleeps == []
    assert caught.value.http_status == status
    assert caught.value.phase == "request"


def test_exhausted_503_keeps_retry_behavior_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "2")
    sleeps: list[float] = []
    monkeypatch.setattr(generator_module.time, "sleep", sleeps.append)
    generator, session = _generator([FakeResponse(503), FakeResponse(503)])

    with pytest.raises(ImageGenerationError) as caught:
        generator.generate("p", "low", tmp_path)

    assert session.calls == 2
    assert sleeps == [1.0]
    assert caught.value.http_status == 503
    assert caught.value.phase == "request"


def test_explicit_native_unsupported_keeps_structured_metadata(tmp_path: Path) -> None:
    generator, session = _generator(
        [FakeResponse(422, {"error": "parameter n is not supported"})]
    )

    with pytest.raises(NativeBatchUnsupported) as caught:
        generator.generate_many("p", "low", tmp_path, n=2)

    assert session.calls == 1
    assert caught.value.http_status == 422
    assert caught.value.phase == "request"


@pytest.mark.parametrize(
    ("responses", "expected_category", "expected_attempts"),
    [
        ([FakeResponse(429)], "http_429", 1),
        ([FakeResponse(401)], "http_4xx", 1),
        ([FakeResponse(500)], "http_5xx", 1),
        ([FakeResponse(503), FakeResponse(503)], "http_503", 2),
    ],
)
def test_task_terminal_category_matches_final_physical_attempt(
    responses: list[FakeResponse],
    expected_category: str,
    expected_attempts: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(generator_module.time, "sleep", lambda _seconds: None)
    store = TaskStore(tmp_path / f"{expected_category}.db")
    manager = TaskManager(store, source_dir=tmp_path / f"sources-{expected_category}")
    task = store.create("p", "2k")
    store.update(task["task_id"], status="processing")
    generator, session = _generator(responses)
    observer = manager._attempt_observer([task], _settings(), "single")

    with pytest.raises(ImageGenerationError) as caught:
        generator.generate(
            "p",
            "low",
            tmp_path / f"output-{expected_category}",
            attempt_observer=observer,
        )
    manager._fail_task(task["task_id"], caught.value)

    assert session.calls == expected_attempts
    timeline = store.telemetry.task_timeline(task["task_id"])
    attempts = timeline["generation_attempts"]
    terminal = [
        event for event in timeline["events"] if event["event_type"] == "terminal_failed"
    ]
    assert len(attempts) == expected_attempts
    assert all(attempt["error_category"] == expected_category for attempt in attempts)
    assert attempts[-1]["will_retry"] == 0
    if expected_attempts > 1:
        assert all(attempt["will_retry"] == 1 for attempt in attempts[:-1])
    assert terminal[-1]["details"] == {
        "category": expected_category,
        "stage": "generation",
    }


def test_task_failure_without_metadata_keeps_existing_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "unit-test-key")
    monkeypatch.setenv("IMAGE_API_BASE_URL", "https://provider.invalid/v1")
    store = TaskStore(tmp_path / "fallback-classification.db")
    manager = TaskManager(store, source_dir=tmp_path / "sources")

    unknown_task = store.create("p", "2k")
    store.update(unknown_task["task_id"], status="processing")
    manager._fail_task(unknown_task["task_id"], RuntimeError("local failure"))

    timeout_task = store.create("q", "2k")
    store.update(timeout_task["task_id"], status="processing")
    try:
        raise ImageGenerationError("request failed") from requests.ReadTimeout(
            "sensitive timeout"
        )
    except ImageGenerationError as wrapped:
        manager._fail_task(timeout_task["task_id"], wrapped)

    unknown_timeline = store.telemetry.task_timeline(unknown_task["task_id"])
    timeout_timeline = store.telemetry.task_timeline(timeout_task["task_id"])
    assert unknown_timeline["events"][-1]["details"]["category"] == "unknown"
    assert timeout_timeline["events"][-1]["details"]["category"] == "read_timeout"


def test_task_failure_ignores_untrusted_exception_metadata(tmp_path: Path) -> None:
    class MisleadingError(RuntimeError):
        http_status = 503
        phase = "request"

    class HostileError(RuntimeError):
        @property
        def http_status(self) -> int:
            raise AssertionError("arbitrary exception metadata must not be read")

        @property
        def phase(self) -> str:
            raise AssertionError("arbitrary exception metadata must not be read")

    store = TaskStore(tmp_path / "untrusted-metadata.db")
    manager = TaskManager(store, source_dir=tmp_path / "sources")

    for index, error in enumerate((MisleadingError("local"), HostileError("local"))):
        task = store.create(f"p-{index}", "2k")
        store.update(task["task_id"], status="processing")
        manager._fail_task(task["task_id"], error)
        timeline = store.telemetry.task_timeline(task["task_id"])
        assert timeline["events"][-1]["details"]["category"] == "unknown"
