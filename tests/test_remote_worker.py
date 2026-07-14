from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from PIL import Image

from cli import _parser
from image_pipeline.remote_worker import (
    RemoteUpscaleWorker,
    RemoteWorkerProtocolError,
    RemoteWorkerSettings,
    UpscaleClaim,
    _worker_slot_settings,
)


def _png_bytes(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        payload: dict[str, Any] | None = None,
        content: bytes = b"",
    ):
        self.status_code = status_code
        self._payload = payload
        self._content = content

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload

    def iter_content(self, chunk_size: int) -> Any:
        for offset in range(0, len(self._content), chunk_size):
            yield self._content[offset : offset + chunk_size]


class FakeSession:
    def __init__(self, gets: list[FakeResponse | Exception], posts: list[FakeResponse | Exception]):
        self.gets = iter(gets)
        self.posts = iter(posts)
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.get_calls.append({"url": url, **kwargs})
        outcome = next(self.gets)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        image_tuple = kwargs.get("files", {}).get("image")
        captured = {
            "url": url,
            **kwargs,
            "uploaded": image_tuple[1].read() if image_tuple else None,
        }
        self.post_calls.append(captured)
        outcome = next(self.posts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        pass


class FakeUpscaler:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[Path, Path, tuple[int, int]]] = []

    def upscale(
        self, source: Path, output_dir: Path, target: tuple[int, int], fit: str = "cover"
    ) -> Any:
        self.calls.append((source, output_dir, target))
        if self.fail:
            raise RuntimeError("GPU exploded with private diagnostic details")
        output_dir.mkdir(parents=True, exist_ok=True)
        final = output_dir / "final.png"
        final.write_bytes(_png_bytes(target))
        return SimpleNamespace(
            output_image=SimpleNamespace(path=str(final)),
            upscale_seconds=12.5,
            postprocess_seconds=1.25,
            peak_vram_mib=None,
        )


def _settings(tmp_path: Path) -> RemoteWorkerSettings:
    return RemoteWorkerSettings(
        cloud_base_url="https://cloud.example.test",
        worker_token="worker-secret-value",
        worker_id="desktop-4090",
        poll_seconds=4.0,
        work_root=tmp_path,
    )


def _claim(source_url: str = "/internal/upscale/source/task-1") -> FakeResponse:
    return FakeResponse(
        200,
        payload={
            "task_id": "task-1",
            "size": "2k",
            "claim_token": "claim-secret-value",
            "source_url": source_url,
        },
    )


def test_settings_require_cloud_url_and_worker_token_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IMAGE_CLOUD_BASE_URL", raising=False)
    monkeypatch.delenv("IMAGE_UPSCALE_WORKER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="IMAGE_CLOUD_BASE_URL"):
        RemoteWorkerSettings.from_env()

    monkeypatch.setenv("IMAGE_CLOUD_BASE_URL", "https://cloud.example.test/private")
    with pytest.raises(RuntimeError) as caught:
        RemoteWorkerSettings.from_env()
    assert "cloud.example" not in str(caught.value)


def test_worker_concurrency_defaults_to_three_and_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_CLOUD_BASE_URL", "https://cloud.example.test")
    monkeypatch.setenv("IMAGE_UPSCALE_WORKER_TOKEN", "x" * 64)
    monkeypatch.delenv("IMAGE_UPSCALE_WORKER_CONCURRENCY", raising=False)
    assert RemoteWorkerSettings.from_env().concurrency == 3

    monkeypatch.setenv("IMAGE_UPSCALE_WORKER_CONCURRENCY", "5")
    assert RemoteWorkerSettings.from_env().concurrency == 5
    monkeypatch.setenv("IMAGE_UPSCALE_WORKER_CONCURRENCY", "6")
    with pytest.raises(RuntimeError, match="1 to 5"):
        RemoteWorkerSettings.from_env()


def test_worker_pool_uses_unique_ids_and_preserves_global_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    slots = _worker_slot_settings(settings)
    assert [slot.worker_id for slot in slots] == [
        "desktop-4090-1",
        "desktop-4090-2",
        "desktop-4090-3",
    ]
    assert all(slot.concurrency == 1 for slot in slots)
    assert all(slot.worker_token == settings.worker_token for slot in slots)


def test_claim_204_means_no_work_and_sends_bearer(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(204)], [])
    worker = RemoteUpscaleWorker(
        _settings(tmp_path), session=session, upscaler=FakeUpscaler()
    )

    assert worker.run_once() is False
    assert session.get_calls[0]["headers"] == {
        "Authorization": "Bearer worker-secret-value",
        "X-Worker-ID": "desktop-4090",
    }
    assert session.get_calls[0]["allow_redirects"] is False
    assert list(tmp_path.iterdir()) == []


def test_full_claim_download_upscale_submit_and_cleanup(tmp_path: Path) -> None:
    source = _png_bytes((64, 48))
    session = FakeSession([_claim(), FakeResponse(200, content=source)], [FakeResponse(200)])
    upscaler = FakeUpscaler()
    worker = RemoteUpscaleWorker(
        _settings(tmp_path), session=session, upscaler=upscaler
    )

    assert worker.run_once() is True

    assert upscaler.calls[0][2] == (2048, 2048)
    download_headers = session.get_calls[1]["headers"]
    assert download_headers["Authorization"] == "Bearer worker-secret-value"
    assert download_headers["X-Claim-Token"] == "claim-secret-value"
    assert session.get_calls[1]["allow_redirects"] is False
    submitted = session.post_calls[0]
    assert float(submitted["data"].pop("source_download_seconds")) >= 0
    assert submitted["data"] == {
        "task_id": "task-1",
        "claim_token": "claim-secret-value",
        "worker_id": "desktop-4090",
        "upscale_seconds": "12.5",
        "postprocess_seconds": "1.25",
        "peak_vram_mib": "",
    }
    assert submitted["files"]["image"][0] == "result.png"
    assert submitted["files"]["image"][2] == "image/png"
    assert submitted["allow_redirects"] is False
    with Image.open(BytesIO(submitted["uploaded"])) as image:
        assert image.size == (2048, 2048)
    assert list(tmp_path.iterdir()) == []


def test_claim_retries_network_and_5xx_with_bounded_exponential_backoff(
    tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    session = FakeSession(
        [
            FakeResponse(503),
            requests.ConnectionError("contains https://cloud.example.test and secret"),
            FakeResponse(204),
        ],
        [],
    )
    worker = RemoteUpscaleWorker(
        _settings(tmp_path),
        session=session,
        upscaler=FakeUpscaler(),
        sleep=sleeps.append,
    )

    assert worker.run_once() is False
    assert sleeps == [1.0, 2.0]
    assert len(session.get_calls) == 3


def test_submit_retry_reopens_file_and_reuses_same_claim(tmp_path: Path) -> None:
    source = _png_bytes((64, 48))
    session = FakeSession(
        [_claim(), FakeResponse(200, content=source)],
        [FakeResponse(503), requests.ReadTimeout("lost response"), FakeResponse(200)],
    )
    sleeps: list[float] = []
    worker = RemoteUpscaleWorker(
        _settings(tmp_path),
        session=session,
        upscaler=FakeUpscaler(),
        sleep=sleeps.append,
    )

    assert worker.run_once() is True
    assert sleeps == [1.0, 2.0]
    assert len(session.post_calls) == 3
    assert all(call["uploaded"] for call in session.post_calls)
    assert {
        call["data"]["claim_token"] for call in session.post_calls
    } == {"claim-secret-value"}


def test_heartbeat_uses_claim_identity_and_detects_lost_lease(tmp_path: Path) -> None:
    session = FakeSession([], [FakeResponse(200), FakeResponse(409)])
    worker = RemoteUpscaleWorker(
        _settings(tmp_path), session=session, upscaler=FakeUpscaler()
    )
    claim = UpscaleClaim(
        "task-1",
        "2k",
        "claim-secret-value",
        "/internal/upscale/source/task-1",
        lease_seconds=30,
        heartbeat_interval_seconds=5,
    )

    assert worker._heartbeat(claim) is True
    assert worker._heartbeat(claim) is False
    assert all(
        call["url"].endswith("/internal/upscale/heartbeat")
        for call in session.post_calls
    )
    assert session.post_calls[0]["data"] == {
        "task_id": "task-1",
        "claim_token": "claim-secret-value",
        "worker_id": "desktop-4090",
    }


def test_gpu_failure_retains_work_without_leaking_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _png_bytes((64, 48))
    session = FakeSession(
        [_claim(), FakeResponse(200, content=source)], [FakeResponse(200)]
    )
    worker = RemoteUpscaleWorker(
        _settings(tmp_path), session=session, upscaler=FakeUpscaler(fail=True)
    )

    assert worker.run_once() is True

    assert len(list(tmp_path.iterdir())) == 1
    assert len(session.post_calls) == 1
    assert session.post_calls[0]["url"].endswith("/internal/upscale/release")
    assert session.post_calls[0]["data"]["error_code"] == "RuntimeError"
    assert session.post_calls[0]["uploaded"] is None
    stderr = capsys.readouterr().err
    assert "GPU exploded" not in stderr
    assert "private diagnostic" not in stderr
    assert "worker-secret-value" not in stderr
    assert "cloud.example" not in stderr


def test_rejects_cross_origin_source_before_sending_worker_token(tmp_path: Path) -> None:
    session = FakeSession([_claim("https://attacker.example/source")], [])
    worker = RemoteUpscaleWorker(
        _settings(tmp_path), session=session, upscaler=FakeUpscaler()
    )

    claim = worker.claim()
    assert claim is not None
    with pytest.raises(RemoteWorkerProtocolError) as caught:
        worker._download_source(claim, tmp_path / "source.png")
    assert "attacker.example" not in str(caught.value)
    assert "worker-secret-value" not in str(caught.value)
    assert len(session.get_calls) == 1


def test_cli_registers_upscale_worker_command() -> None:
    args = _parser().parse_args(["upscale-worker"])
    assert args.command == "upscale-worker"
