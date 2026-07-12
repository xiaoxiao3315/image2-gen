from __future__ import annotations

import os
import queue
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urljoin, urlsplit

import requests

from .config import PROJECT_ROOT, TARGET_SIZES, Settings
from .image_io import inspect_image
from .models import UpscaleResult
from .upscaler import RealEsrganUpscaler


DEFAULT_POLL_SECONDS = 4.0
DEFAULT_WORKER_CONCURRENCY = 3
MAX_WORKER_CONCURRENCY = 5
MAX_POLL_BACKOFF_SECONDS = 60.0
REQUEST_ATTEMPTS = 3
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class RemoteWorkerError(RuntimeError):
    """A safe-to-display worker error that never includes remote response data."""


class RemoteWorkerAuthenticationError(RemoteWorkerError):
    pass


class RemoteWorkerProtocolError(RemoteWorkerError):
    pass


class _Upscaler(Protocol):
    def upscale(
        self, source: Path, output_dir: Path, target: tuple[int, int], fit: str = "cover"
    ) -> UpscaleResult: ...


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


def _bounded_int_env(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer from 1 to {maximum}") from exc
    if not 1 <= value <= maximum:
        raise RuntimeError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _validate_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("IMAGE_CLOUD_BASE_URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise RuntimeError("IMAGE_CLOUD_BASE_URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise RuntimeError("IMAGE_CLOUD_BASE_URL must not contain a query or fragment")
    return normalized


def _safe_identifier(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
        raise RemoteWorkerProtocolError(f"Cloud returned an invalid {field}")
    return value


@dataclass(frozen=True)
class RemoteWorkerSettings:
    cloud_base_url: str
    worker_token: str
    worker_id: str
    poll_seconds: float
    work_root: Path
    concurrency: int = DEFAULT_WORKER_CONCURRENCY

    @classmethod
    def from_env(cls) -> "RemoteWorkerSettings":
        cloud_base_url = os.getenv("IMAGE_CLOUD_BASE_URL", "").strip()
        if not cloud_base_url:
            raise RuntimeError("Missing IMAGE_CLOUD_BASE_URL")
        worker_token = os.getenv("IMAGE_UPSCALE_WORKER_TOKEN", "")
        if not worker_token:
            raise RuntimeError("Missing IMAGE_UPSCALE_WORKER_TOKEN")
        worker_id = os.getenv("IMAGE_UPSCALE_WORKER_ID", "").strip() or socket.gethostname()
        _safe_identifier(worker_id, "IMAGE_UPSCALE_WORKER_ID")
        work_root = Path(
            os.getenv(
                "IMAGE_UPSCALE_WORK_ROOT",
                PROJECT_ROOT / "remote-worker-data",
            )
        ).resolve()
        return cls(
            cloud_base_url=_validate_base_url(cloud_base_url),
            worker_token=worker_token,
            worker_id=worker_id,
            poll_seconds=_positive_float_env(
                "IMAGE_UPSCALE_POLL_SECONDS", DEFAULT_POLL_SECONDS
            ),
            work_root=work_root,
            concurrency=_bounded_int_env(
                "IMAGE_UPSCALE_WORKER_CONCURRENCY",
                DEFAULT_WORKER_CONCURRENCY,
                MAX_WORKER_CONCURRENCY,
            ),
        )


@dataclass(frozen=True)
class UpscaleClaim:
    task_id: str
    size: str
    claim_token: str
    source_url: str
    lease_seconds: int = 600
    heartbeat_interval_seconds: int = 60


class RemoteUpscaleWorker:
    def __init__(
        self,
        settings: RemoteWorkerSettings,
        *,
        session: requests.Session | Any | None = None,
        heartbeat_session: requests.Session | Any | None = None,
        upscaler: _Upscaler | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self.heartbeat_session = heartbeat_session or (
            requests.Session() if session is None else self.session
        )
        self.upscaler = upscaler or RealEsrganUpscaler(
            Settings.from_env(require_key=False)
        )
        self.sleep = sleep
        self.settings.work_root.mkdir(parents=True, exist_ok=True)

    @property
    def _authorization_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.worker_token}"}

    @property
    def _worker_headers(self) -> dict[str, str]:
        return {
            **self._authorization_headers,
            "X-Worker-ID": self.settings.worker_id,
        }

    def _endpoint(self, path: str) -> str:
        return f"{self.settings.cloud_base_url}{path}"

    def _same_cloud_origin_url(self, source_url: str) -> str:
        absolute = urljoin(f"{self.settings.cloud_base_url}/", source_url)
        expected = urlsplit(self.settings.cloud_base_url)
        actual = urlsplit(absolute)

        def origin(parts: Any) -> tuple[str, str | None, int | None]:
            default_port = 443 if parts.scheme == "https" else 80
            return parts.scheme.lower(), parts.hostname, parts.port or default_port

        if origin(actual) != origin(expected):
            raise RemoteWorkerProtocolError("Cloud returned a cross-origin source URL")
        return absolute

    def _retry_delay(self, attempt: int) -> None:
        self.sleep(float(2 ** (attempt - 1)))

    @staticmethod
    def _close_response(response: Any) -> None:
        close = getattr(response, "close", None)
        if close is not None:
            close()

    @staticmethod
    def _check_authentication(status_code: int) -> None:
        if status_code in {401, 403}:
            raise RemoteWorkerAuthenticationError(
                "Cloud worker authentication was rejected"
            )

    def claim(self) -> UpscaleClaim | None:
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            try:
                response = self.session.get(
                    self._endpoint("/internal/upscale/claim"),
                    headers=self._worker_headers,
                    timeout=(10, 30),
                    allow_redirects=False,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == REQUEST_ATTEMPTS:
                    raise RemoteWorkerError(
                        "Cloud claim request failed after bounded retries"
                    ) from None
                self._retry_delay(attempt)
                continue
            try:
                self._check_authentication(response.status_code)
                if response.status_code == 204:
                    return None
                if 500 <= response.status_code <= 599:
                    if attempt == REQUEST_ATTEMPTS:
                        raise RemoteWorkerError(
                            "Cloud claim request failed after bounded retries"
                        )
                    self._retry_delay(attempt)
                    continue
                if response.status_code != 200:
                    raise RemoteWorkerProtocolError("Cloud claim request was rejected")
                try:
                    payload = response.json()
                    task_id = _safe_identifier(str(payload["task_id"]), "task_id")
                    size = str(payload["size"]).lower()
                    claim_token = str(payload["claim_token"])
                    source_url = str(payload["source_url"])
                    lease_seconds = int(payload.get("lease_seconds", 600))
                    heartbeat_interval_seconds = int(
                        payload.get("heartbeat_interval_seconds", 60)
                    )
                except (KeyError, TypeError, ValueError):
                    raise RemoteWorkerProtocolError(
                        "Cloud returned an invalid claim document"
                    ) from None
            finally:
                self._close_response(response)
            if size not in TARGET_SIZES:
                raise RemoteWorkerProtocolError("Cloud returned an invalid target size")
            if not claim_token or len(claim_token) > 512:
                raise RemoteWorkerProtocolError("Cloud returned an invalid claim token")
            if lease_seconds <= 0 or not 1 <= heartbeat_interval_seconds <= lease_seconds:
                raise RemoteWorkerProtocolError("Cloud returned invalid lease settings")
            return UpscaleClaim(
                task_id,
                size,
                claim_token,
                source_url,
                lease_seconds,
                heartbeat_interval_seconds,
            )
        raise AssertionError("bounded claim retry loop exhausted unexpectedly")

    def _download_source(self, claim: UpscaleClaim, destination: Path) -> float:
        started = time.perf_counter()
        source_url = self._same_cloud_origin_url(claim.source_url)
        headers = {
            **self._authorization_headers,
            "X-Claim-Token": claim.claim_token,
        }
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            destination.unlink(missing_ok=True)
            try:
                response = self.session.get(
                    source_url,
                    headers=headers,
                    timeout=(10, 120),
                    stream=True,
                    allow_redirects=False,
                )
                try:
                    self._check_authentication(response.status_code)
                    if 500 <= response.status_code <= 599:
                        if attempt == REQUEST_ATTEMPTS:
                            raise RemoteWorkerError(
                                "Source download failed after bounded retries"
                            )
                        self._retry_delay(attempt)
                        continue
                    if response.status_code != 200:
                        raise RemoteWorkerProtocolError("Source download was rejected")
                    with destination.open("xb") as output:
                        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                            if chunk:
                                output.write(chunk)
                finally:
                    self._close_response(response)
                inspect_image(destination)
                return round(time.perf_counter() - started, 3)
            except (requests.ConnectionError, requests.Timeout):
                destination.unlink(missing_ok=True)
                if attempt == REQUEST_ATTEMPTS:
                    raise RemoteWorkerError(
                        "Source download failed after bounded retries"
                    ) from None
                self._retry_delay(attempt)
            except RemoteWorkerError:
                raise
            except Exception:
                destination.unlink(missing_ok=True)
                raise RemoteWorkerProtocolError("Downloaded source is not a valid image") from None

    def _heartbeat(self, claim: UpscaleClaim) -> bool:
        try:
            response = self.heartbeat_session.post(
                self._endpoint("/internal/upscale/heartbeat"),
                headers=self._authorization_headers,
                data={
                    "task_id": claim.task_id,
                    "claim_token": claim.claim_token,
                    "worker_id": self.settings.worker_id,
                },
                timeout=(10, 30),
                allow_redirects=False,
            )
        except (requests.ConnectionError, requests.Timeout):
            return True
        try:
            self._check_authentication(response.status_code)
            if response.status_code == 409:
                return False
            if response.status_code != 200:
                return True
            return True
        finally:
            self._close_response(response)

    def _start_lease_keeper(
        self, claim: UpscaleClaim
    ) -> tuple[threading.Event, threading.Event, threading.Thread]:
        stop = threading.Event()
        lost = threading.Event()

        def keep_alive() -> None:
            while not stop.wait(claim.heartbeat_interval_seconds):
                try:
                    active = self._heartbeat(claim)
                except RemoteWorkerAuthenticationError:
                    lost.set()
                    return
                if not active:
                    lost.set()
                    return

        thread = threading.Thread(
            target=keep_alive,
            name=f"upscale-lease-{claim.task_id[:12]}",
            daemon=True,
        )
        thread.start()
        return stop, lost, thread

    def _release(self, claim: UpscaleClaim, error_code: str) -> None:
        safe_error_code = re.sub(r"[^A-Za-z0-9_.-]", "_", error_code)[:64]
        if not safe_error_code or not safe_error_code[0].isalpha():
            safe_error_code = "worker_error"
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            try:
                response = self.session.post(
                    self._endpoint("/internal/upscale/release"),
                    headers=self._authorization_headers,
                    data={
                        "task_id": claim.task_id,
                        "claim_token": claim.claim_token,
                        "worker_id": self.settings.worker_id,
                        "error_code": safe_error_code,
                    },
                    timeout=(10, 30),
                    allow_redirects=False,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt < REQUEST_ATTEMPTS:
                    self._retry_delay(attempt)
                continue
            except Exception:
                # Release is best-effort; the server lease remains the final recovery path.
                return
            try:
                self._check_authentication(response.status_code)
                if response.status_code in {200, 409}:
                    return
                if 500 <= response.status_code <= 599 and attempt < REQUEST_ATTEMPTS:
                    self._retry_delay(attempt)
                    continue
                return
            finally:
                self._close_response(response)

    def _submit(
        self,
        claim: UpscaleClaim,
        result: UpscaleResult,
        source_download_seconds: float,
    ) -> None:
        final_path = Path(result.output_image.path)
        metadata = inspect_image(final_path)
        if (metadata.width, metadata.height) != TARGET_SIZES[claim.size]:
            raise RemoteWorkerProtocolError(
                "Local Pillow verification found an invalid output size"
            )
        fields = {
            "task_id": claim.task_id,
            "claim_token": claim.claim_token,
            "worker_id": self.settings.worker_id,
            "source_download_seconds": str(source_download_seconds),
            "upscale_seconds": str(result.upscale_seconds),
            "postprocess_seconds": str(result.postprocess_seconds),
            "peak_vram_mib": (
                "" if result.peak_vram_mib is None else str(result.peak_vram_mib)
            ),
        }
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            try:
                with final_path.open("rb") as image:
                    response = self.session.post(
                        self._endpoint("/internal/upscale/submit"),
                        headers=self._authorization_headers,
                        data=fields,
                        files={"image": ("result.png", image, "image/png")},
                        timeout=(10, 180),
                        allow_redirects=False,
                    )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == REQUEST_ATTEMPTS:
                    raise RemoteWorkerError(
                        "Result upload failed after bounded retries"
                    ) from None
                self._retry_delay(attempt)
                continue
            try:
                self._check_authentication(response.status_code)
                if 500 <= response.status_code <= 599:
                    if attempt == REQUEST_ATTEMPTS:
                        raise RemoteWorkerError(
                            "Result upload failed after bounded retries"
                        )
                    self._retry_delay(attempt)
                    continue
                if response.status_code != 200:
                    raise RemoteWorkerProtocolError("Result upload was rejected")
                return
            finally:
                self._close_response(response)
        raise AssertionError("bounded submit retry loop exhausted unexpectedly")

    def run_once(self) -> bool:
        claim = self.claim()
        if claim is None:
            return False
        prefix = f"{claim.task_id[:48]}-"
        task_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.settings.work_root))
        source_path = task_dir / "source.png"
        output_dir = task_dir / "output"
        heartbeat_stop, lease_lost, heartbeat_thread = self._start_lease_keeper(claim)
        try:
            source_download_seconds = self._download_source(claim, source_path)
            result = self.upscaler.upscale(
                source_path, output_dir, TARGET_SIZES[claim.size]
            )
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)
            if lease_lost.is_set():
                raise RemoteWorkerProtocolError("Cloud claim lease is no longer active")
            self._submit(claim, result, source_download_seconds)
        except RemoteWorkerAuthenticationError:
            raise
        except Exception as exc:
            self._release(claim, type(exc).__name__)
            print(
                f"Upscale task {claim.task_id} failed ({type(exc).__name__}); "
                "local files retained and the cloud task was released for recovery.",
                file=sys.stderr,
                flush=True,
            )
            return True
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)
        shutil.rmtree(task_dir)
        print(
            f"Upscale task {claim.task_id} completed and submitted.",
            flush=True,
        )
        return True

    def run_forever(self) -> None:
        failure_delay = max(1.0, self.settings.poll_seconds)
        print("Remote upscale worker started.", flush=True)
        while True:
            try:
                claimed = self.run_once()
            except RemoteWorkerAuthenticationError:
                raise
            except RemoteWorkerError as exc:
                print(
                    f"Cloud polling failed ({type(exc).__name__}); retrying with backoff.",
                    file=sys.stderr,
                    flush=True,
                )
                self.sleep(failure_delay)
                failure_delay = min(MAX_POLL_BACKOFF_SECONDS, failure_delay * 2)
                continue
            failure_delay = max(1.0, self.settings.poll_seconds)
            if not claimed:
                self.sleep(self.settings.poll_seconds)

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if close is not None:
            close()
        if self.heartbeat_session is not self.session:
            heartbeat_close = getattr(self.heartbeat_session, "close", None)
            if heartbeat_close is not None:
                heartbeat_close()


def _worker_slot_settings(settings: RemoteWorkerSettings) -> list[RemoteWorkerSettings]:
    if settings.concurrency == 1:
        return [settings]
    slots: list[RemoteWorkerSettings] = []
    for index in range(1, settings.concurrency + 1):
        suffix = f"-{index}"
        worker_id = f"{settings.worker_id[:128 - len(suffix)]}{suffix}"
        slots.append(replace(settings, worker_id=worker_id, concurrency=1))
    return slots


def run_remote_worker() -> None:
    """Run independent claim/download/upscale/upload loops in one process."""
    settings = RemoteWorkerSettings.from_env()
    slots = _worker_slot_settings(settings)
    if len(slots) == 1:
        worker = RemoteUpscaleWorker(slots[0])
        try:
            worker.run_forever()
        finally:
            worker.close()
        return

    failures: queue.Queue[BaseException] = queue.Queue()

    def run_slot(slot: RemoteWorkerSettings) -> None:
        worker = RemoteUpscaleWorker(slot)
        try:
            worker.run_forever()
        except BaseException as exc:
            failures.put(exc)
        finally:
            worker.close()

    threads = [
        threading.Thread(
            target=run_slot,
            args=(slot,),
            name=f"remote-upscale-{index}",
            daemon=True,
        )
        for index, slot in enumerate(slots, start=1)
    ]
    print(f"Remote upscale worker pool starting with concurrency={len(threads)}.", flush=True)
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        try:
            failure = failures.get(timeout=0.5)
        except queue.Empty:
            continue
        raise failure
