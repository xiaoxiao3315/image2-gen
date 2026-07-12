from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


CASES = [
    {
        "name": "rainy-city-night",
        "size": "4k",
        "prompt": "雨后城市夜景，湿润街道倒映霓虹灯，写实摄影，电影感构图，无文字，无标志，无水印",
    },
    {
        "name": "forest-cabin",
        "size": "2k",
        "prompt": "清晨薄雾中的森林木屋，窗户透出温暖灯光，松树和苔藓环绕，写实自然摄影，无文字，无水印",
    },
    {
        "name": "orange-cat-window",
        "size": "2k",
        "prompt": "一只毛茸茸的橘猫坐在阳光窗台上看向窗外，室内温暖柔和，细腻写实摄影，无文字，无水印",
    },
    {
        "name": "future-tech-city",
        "size": "4k",
        "prompt": "未来科技城市全景，层叠摩天楼、空中交通和蓝色能源光带，宏大写实科幻概念艺术，无文字，无标志，无水印",
    },
    {
        "name": "mountain-sunrise",
        "size": "4k",
        "prompt": "高山峡谷日出，金色晨光穿过云海照亮雪峰，广角自然风光摄影，真实细节，无文字，无水印",
    },
]


def validate_image(raw: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        width, height = image.size
    return width, height, hashlib.sha256(raw).hexdigest()


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    attempts: int = 6,
    **kwargs: Any,
) -> httpx.Response:
    """Retry transient tunnel/proxy failures; HTTP responses are returned unchanged."""
    for attempt in range(1, attempts + 1):
        try:
            response = await client.request(method, url, **kwargs)
            if (
                method.upper() in {"GET", "HEAD"}
                and response.status_code in {502, 503, 504}
                and attempt < attempts
            ):
                await response.aclose()
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue
            return response
        except httpx.TransportError:
            if attempt == attempts:
                raise
            await asyncio.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError("unreachable")


async def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    token = os.getenv("IMAGE_SERVICE_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("external-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    submitted: list[dict[str, Any]] = []
    timeout = httpx.Timeout(connect=30, read=180, write=30, pool=30)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        health = await request_with_retry(client, "GET", f"{base_url}/health")
        health.raise_for_status()
        if args.resume_file:
            submitted = json.loads(args.resume_file.read_text(encoding="utf-8"))["tasks"]
            for item in submitted:
                item["submitted_perf"] = None
            print(f"RESUMED {len(submitted)} existing task(s)", flush=True)
        else:
            for case in CASES:
                started = time.perf_counter()
                response = await request_with_retry(
                    client,
                    "POST",
                    f"{base_url}/v1/generate",
                    headers=headers,
                    json={"prompt": case["prompt"], "size": case["size"]},
                )
                if response.status_code != 202:
                    raise RuntimeError(f"submit failed HTTP {response.status_code}")
                payload = response.json()
                submitted.append(
                    {
                        **case,
                        "task_id": payload["task_id"],
                        "submit_status": payload["status"],
                        "submitted_perf": started,
                    }
                )
                print(f"SUBMITTED {case['name']} task={payload['task_id']} size={case['size']}", flush=True)
            (output_dir / "submissions.json").write_text(
                json.dumps(
                    {"base_url": base_url, "tasks": [
                        {key: value for key, value in item.items() if key != "submitted_perf"}
                        for item in submitted
                    ]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        pending = {item["task_id"]: item for item in submitted}
        deadline = time.perf_counter() + args.timeout_seconds
        while pending:
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"timed out with {len(pending)} task(s) pending")
            for task_id, item in list(pending.items()):
                response = await request_with_retry(
                    client,
                    "GET",
                    f"{base_url}/v1/result/{task_id}",
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                status = payload["status"]
                if status == "failed":
                    raise RuntimeError(f"task {task_id} failed: {payload.get('error')}")
                if status != "done":
                    continue
                download_started = time.perf_counter()
                image_response = await request_with_retry(
                    client, "GET", payload["image_url"]
                )
                image_response.raise_for_status()
                download_seconds = time.perf_counter() - download_started
                raw = image_response.content
                width, height, digest = validate_image(raw)
                expected = (2048, 2048) if item["size"] == "2k" else (3840, 2160)
                if (width, height) != expected:
                    raise RuntimeError(
                        f"task {task_id} expected {expected}, downloaded {width}x{height}"
                    )
                local_download = output_dir / f"{item['name']}-{task_id}.png"
                local_download.write_bytes(raw)
                completed_perf = time.perf_counter()
                item.update(
                    {
                        "status": "done",
                        "image_url": payload["image_url"],
                        "server_local_path": payload["local_path"],
                        "downloaded_path": str(local_download.resolve()),
                        "actual_width": width,
                        "actual_height": height,
                        "file_bytes": len(raw),
                        "sha256": digest,
                        "end_to_end_seconds_client": (
                            None
                            if item["submitted_perf"] is None
                            else round(completed_perf - item["submitted_perf"], 3)
                        ),
                        "end_to_end_seconds_server": payload["metrics"]["end_to_end_seconds"],
                        "full_chain_seconds_server_plus_download": round(
                            payload["metrics"]["end_to_end_seconds"] + download_seconds, 3
                        ),
                        "download_seconds": round(download_seconds, 3),
                        "server_metrics": payload["metrics"],
                        "cost": payload["cost"],
                    }
                )
                del pending[task_id]
                print(
                    f"DONE {item['name']} pixels={width}x{height} "
                    f"server_e2e={item['end_to_end_seconds_server']}s",
                    flush=True,
                )
            if pending:
                await asyncio.sleep(args.poll_seconds)

    for item in submitted:
        item.pop("submitted_perf", None)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "tested_public_base_url": base_url,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "results": submitted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        columns = [
            "name",
            "prompt",
            "size",
            "task_id",
            "image_url",
            "server_local_path",
            "downloaded_path",
            "actual_width",
            "actual_height",
            "file_bytes",
            "sha256",
            "end_to_end_seconds_client",
            "download_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(submitted)
    print(f"RESULT_DIR={output_dir.resolve()}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Public-network end-to-end image service test")
    root.add_argument("--base-url", required=True)
    root.add_argument("--output-dir", type=Path)
    root.add_argument("--resume-file", type=Path)
    root.add_argument("--poll-seconds", type=float, default=2)
    root.add_argument("--timeout-seconds", type=float, default=900)
    return root


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
