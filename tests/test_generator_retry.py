from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from PIL import Image

import image_pipeline.generator as generator_module
from image_pipeline.generator import (
    GptImageGenerator,
    ImageGenerationError,
    NativeBatchIncomplete,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


class SequenceSession:
    def __init__(self, outcomes: list[FakeResponse | Exception]):
        self.outcomes = iter(outcomes)
        self.calls: list[dict[str, Any]] = []

    def post(self, _endpoint: str, **kwargs: Any) -> FakeResponse:
        captured = dict(kwargs)
        captured["headers"] = dict(kwargs["headers"])
        self.calls.append(captured)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _png_payload() -> dict[str, Any]:
    buffer = BytesIO()
    Image.new("RGB", (7, 5), "navy").save(buffer, format="PNG")
    return {"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]}


def test_http_200_short_native_batch_preserves_partial_results(tmp_path: Path) -> None:
    generator, session = _generator([FakeResponse(200, _png_payload())])

    with pytest.raises(NativeBatchIncomplete) as caught:
        generator.generate_many(
            "test prompt",
            "low",
            tmp_path,
            n=3,
            idempotency_key="partial-request-001",
        )

    assert caught.value.requested_count == 3
    assert len(caught.value.partial_results) == 1
    partial = caught.value.partial_results[0]
    assert Path(partial.image.path).is_file()
    assert (partial.image.width, partial.image.height) == (7, 5)
    assert session.calls[0]["headers"]["Idempotency-Key"] == "partial-request-001"


def _generator(outcomes: list[FakeResponse | Exception]) -> tuple[GptImageGenerator, SequenceSession]:
    settings = SimpleNamespace(
        api_proxy=None,
        api_key="test-only-key",
        api_base_url="https://example.test/v1",
        api_connect_timeout_seconds=20,
        api_timeout_seconds=30,
        model="gpt-image-2",
        source_size="1536x1024",
    )
    generator = GptImageGenerator(settings)
    session = SequenceSession(outcomes)
    generator.session = session  # type: ignore[assignment]
    return generator, session


def test_retries_503_with_exponential_backoff_and_stable_idempotency_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("IMAGE_API_MAX_ATTEMPTS", raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(generator_module.time, "sleep", sleeps.append)
    generator, session = _generator(
        [FakeResponse(503), FakeResponse(503), FakeResponse(200, _png_payload())]
    )

    result = generator.generate(
        "test prompt", "low", tmp_path, idempotency_key="stable-request-001"
    )

    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]
    assert session.calls[0]["timeout"] == (20, 30)
    idempotency_keys = {
        call["headers"]["Idempotency-Key"] for call in session.calls
    }
    assert idempotency_keys == {"stable-request-001"}
    assert (result.image.width, result.image.height) == (7, 5)


@pytest.mark.parametrize(
    "transient_error",
    [
        requests.ConnectTimeout("connect timed out"),
        requests.ReadTimeout("read timed out"),
        requests.ConnectionError("disconnected"),
    ],
)
def test_retries_connection_and_timeout_errors(
    transient_error: requests.RequestException,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "2")
    sleeps: list[float] = []
    monkeypatch.setattr(generator_module.time, "sleep", sleeps.append)
    generator, session = _generator(
        [transient_error, FakeResponse(200, _png_payload())]
    )

    generator.generate("test prompt", "medium", tmp_path)

    assert len(session.calls) == 2
    assert sleeps == [1.0]
    assert session.calls[0]["headers"]["Idempotency-Key"] == session.calls[1][
        "headers"
    ]["Idempotency-Key"]


def test_non_503_http_status_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("IMAGE_API_MAX_ATTEMPTS", raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(generator_module.time, "sleep", sleeps.append)
    generator, session = _generator([FakeResponse(500)])

    with pytest.raises(ImageGenerationError, match="HTTP 500"):
        generator.generate("test prompt", "high", tmp_path)

    assert len(session.calls) == 1
    assert sleeps == []


def test_503_retry_limit_is_bounded_by_environment_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "2")
    sleeps: list[float] = []
    monkeypatch.setattr(generator_module.time, "sleep", sleeps.append)
    generator, session = _generator([FakeResponse(503), FakeResponse(503)])

    with pytest.raises(ImageGenerationError, match="HTTP 503"):
        generator.generate("test prompt", "low", tmp_path)

    assert len(session.calls) == 2
    assert sleeps == [1.0]


def test_retry_attempt_setting_rejects_unsafe_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_MAX_ATTEMPTS", "20")
    generator, session = _generator([])

    with pytest.raises(ValueError, match="from 1 to 3"):
        generator.generate("test prompt", "low", tmp_path)

    assert session.calls == []
