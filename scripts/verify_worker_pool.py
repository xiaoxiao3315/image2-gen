from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image


def _wait_for_health(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise RuntimeError("isolated verification service did not become healthy")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _write_source(path: Path, accent: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (512, 384))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (
                (accent[0] + x // 4) % 256,
                (accent[1] + y // 3) % 256,
                (accent[2] + (x + y) // 8) % 256,
            )
    image.save(path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an isolated real two-process GPU worker-pool verification."
    )
    parser.add_argument("--port", type=int, default=18123)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("remote-worker-data") / "elastic-stage1",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    data_root = args.data_root.resolve() / (
        f"run-{int(time.time())}-{secrets.token_hex(4)}"
    )
    source_dir = data_root / "sources"
    image_dir = data_root / "images"
    log_dir = data_root / "logs"
    for directory in (source_dir, image_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    token = secrets.token_urlsafe(48)
    base_url = f"http://127.0.0.1:{args.port}"
    common_env = os.environ.copy()
    common_env.update(
        {
            "IMAGE_SERVICE_DATA": str(data_root),
            "IMAGE_UPSCALE_WORKER_TOKEN": token,
            "IMAGE_UPSCALE_LEASE_SECONDS": "30",
            "IMAGE_UPSCALE_MAX_ATTEMPTS": "3",
            "IMAGE_CLOUD_BASE_URL": base_url,
            "PUBLIC_BASE_URL": base_url,
            "IMAGE_UPSCALE_WORKER_CONCURRENCY": "1",
            "IMAGE_UPSCALE_POLL_SECONDS": "1",
            "IMAGE_UPSCALE_WORK_ROOT": str(data_root / "work"),
            "REALESRGAN_TILE_SIZE": "0",
        }
    )

    os.environ.update(common_env)
    from image_pipeline import service

    store = service.TaskStore(data_root / "tasks.db")
    task_ids: list[str] = []
    for index, accent in enumerate(((20, 40, 80), (100, 30, 10)), start=1):
        task = store.create(f"elastic worker pool verification {index}", "2k")
        source_filename = f"{task['task_id']}.png"
        _write_source(source_dir / source_filename, accent)
        metadata = service.inspect_image(source_dir / source_filename)
        store.update(
            task["task_id"],
            status="awaiting_upscale",
            source_filename=source_filename,
            source_width=metadata.width,
            source_height=metadata.height,
            source_file_bytes=metadata.file_bytes,
            source_sha256=metadata.sha256,
        )
        task_ids.append(task["task_id"])

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    server_log = (log_dir / "server.log").open("wb")
    worker_logs = [
        (log_dir / f"worker-{index}.log").open("wb") for index in (1, 2)
    ]
    processes: list[subprocess.Popen[bytes]] = []
    started = time.perf_counter()
    try:
        server = subprocess.Popen(
            [
                sys.executable,
                str(project_root / "cli.py"),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            cwd=project_root,
            env=common_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        processes.append(server)
        _wait_for_health(base_url, 30)

        for index, log in enumerate(worker_logs, start=1):
            worker_env = common_env.copy()
            worker_env["IMAGE_UPSCALE_WORKER_ID"] = f"stage1-gpu-{index}"
            worker = subprocess.Popen(
                [sys.executable, str(project_root / "cli.py"), "upscale-worker"],
                cwd=project_root,
                env=worker_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            processes.append(worker)

        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            records = [store.get(task_id) for task_id in task_ids]
            if all(record and record["status"] in {"done", "failed"} for record in records):
                break
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("a verification process exited before tasks completed")
            time.sleep(0.5)
        else:
            raise RuntimeError("worker-pool verification timed out")

        records = [store.get(task_id) for task_id in task_ids]
        if not all(record and record["status"] == "done" for record in records):
            raise RuntimeError("one or more verification tasks failed")

        claimed_by = [str(record["claimed_by"]) for record in records if record]
        if set(claimed_by) != {"stage1-gpu-1", "stage1-gpu-2"}:
            raise RuntimeError("tasks were not distributed across both workers")
        if any(int(record["upscale_attempts"] or 0) != 1 for record in records if record):
            raise RuntimeError("a task was claimed more than once")

        results = []
        for record in records:
            assert record is not None
            final_path = image_dir / str(record["image_filename"])
            with Image.open(final_path) as image:
                image.load()
                pixels = list(image.size)
                image_format = image.format
            if pixels != [2048, 2048] or image_format != "PNG":
                raise RuntimeError("Pillow verification found invalid final pixels")
            results.append(
                {
                    "task_id": record["task_id"],
                    "worker_id": record["claimed_by"],
                    "upscale_attempts": record["upscale_attempts"],
                    "actual_pixels": pixels,
                    "format": image_format,
                    "file_bytes": record["file_bytes"],
                    "sha256": record["sha256"],
                }
            )

        report = {
            "status": "passed",
            "worker_processes": 2,
            "task_count": len(results),
            "wall_seconds": round(time.perf_counter() - started, 3),
            "distinct_workers": sorted(set(claimed_by)),
            "duplicate_claims": 0,
            "lost_tasks": 0,
            "results": results,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        for process in reversed(processes):
            _stop_process(process)
        server_log.close()
        for log in worker_logs:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
