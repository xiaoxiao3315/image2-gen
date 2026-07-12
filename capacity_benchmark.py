from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import multiprocessing
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from PIL import Image

from image_pipeline.config import PROJECT_ROOT, TARGET_SIZES, Settings
from image_pipeline.image_io import inspect_image
from image_pipeline.upscaler import _cover_resize


DEFAULT_PROMPT = (
    "Capacity benchmark image: a red cube on a plain light gray studio background, "
    "soft light, no text, no logo, no watermark."
)
OVERLOAD_ERRORS = {
    "http_429",
    "http_503",
    "connect_timeout",
    "read_timeout",
    "remote_disconnect",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = PROJECT_ROOT / "capacity-benchmarks" / f"{stamp}-{prefix}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return round(ordered[index], 3)


def numeric_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): numeric_tree(v) for k, v in value.items() if isinstance(v, (int, float, dict))}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def usage_fields(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None
    usage = payload["usage"]
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "input_tokens_details": numeric_tree(usage.get("input_tokens_details")),
        "output_tokens_details": numeric_tree(usage.get("output_tokens_details")),
        "usage_provenance": "provider_reported_unverified",
        "pricing_eligible": False,
        "caveat": "渠道token口径存疑，不能直接套官方image-output-token单价",
    }


def iter_images(value: Any, trail: str = "root") -> Iterable[tuple[str, str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{trail}.{key}"
            lower = str(key).lower()
            if isinstance(item, str):
                if lower in {"b64_json", "image_base64", "base64"}:
                    yield "base64", item, child
                elif item.startswith("data:image/") and ";base64," in item:
                    yield "base64", item.split(",", 1)[1], child
                elif lower in {"url", "image_url"} and item.startswith(("http://", "https://")):
                    yield "url", item, child
            else:
                yield from iter_images(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_images(item, f"{trail}[{index}]")


def validate_image(raw: bytes) -> tuple[int, int, str, str]:
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        width, height = image.size
        image_format = image.format or "unknown"
    return width, height, image_format, hashlib.sha256(raw).hexdigest()


def classify_http(status: int) -> str:
    if status == 429:
        return "http_429"
    if status == 503:
        return "http_503"
    if 400 <= status < 500:
        return "http_4xx_other"
    if status >= 500:
        return "http_5xx_other"
    return "application_error_2xx"


def classify_exception(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "remote_disconnect"
    if isinstance(exc, httpx.ConnectError):
        return "connect_error"
    if isinstance(exc, httpx.DecodingError):
        return "response_decode_error"
    return "other"


async def api_request(
    client: httpx.AsyncClient,
    endpoint: str,
    key: str,
    prompt: str,
    quality: str,
    size: str,
    concurrency: int,
    round_number: int,
    request_index: int,
    ready: dict[str, int],
    condition: asyncio.Condition,
    release: asyncio.Event,
    release_clock: dict[str, float],
) -> dict[str, Any]:
    async with condition:
        ready["count"] += 1
        condition.notify_all()
    await release.wait()
    started = time.perf_counter()
    record: dict[str, Any] = {
        "stage": "api",
        "concurrency": concurrency,
        "round": round_number,
        "request_index": request_index,
        "started_at_utc": utc_now(),
        "start_skew_ms": round((started - release_clock["value"]) * 1000, 3),
        "model": "gpt-image-2",
        "quality": quality,
        "requested_size": size,
        "http_status": None,
        "outcome": "failure",
        "error_category": None,
        "exception_type": None,
        "request_id": None,
        "http_elapsed_s": None,
        "usable_image_elapsed_s": None,
        "response_bytes": None,
        "image_bytes": None,
        "actual_width": None,
        "actual_height": None,
        "image_format": None,
        "image_sha256": None,
        "response_image_trail": None,
        "usage": None,
    }
    payload = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "quality": quality,
        "size": size,
        "n": 1,
    }
    try:
        response = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        http_done = time.perf_counter()
        record["http_status"] = response.status_code
        record["http_elapsed_s"] = round(http_done - started, 3)
        record["response_bytes"] = len(response.content)
        record["request_id"] = (
            response.headers.get("x-request-id")
            or response.headers.get("request-id")
        )
        if not 200 <= response.status_code < 300:
            record["error_category"] = classify_http(response.status_code)
            try:
                error_payload = response.json().get("error", {})
                if isinstance(error_payload, dict):
                    record["provider_error_type"] = error_payload.get("type")
                    record["provider_error_code"] = error_payload.get("code")
            except Exception:
                pass
            return record
        try:
            body = response.json()
        except Exception:
            record["error_category"] = "response_decode_error"
            return record
        record["usage"] = usage_fields(body)
        candidate = next(iter(iter_images(body)), None)
        if candidate is None:
            record["error_category"] = "missing_image"
            return record
        source_kind, value, trail = candidate
        record["response_image_trail"] = trail
        try:
            if source_kind == "base64":
                raw = base64.b64decode(value, validate=False)
            else:
                image_response = await client.get(value)
                image_response.raise_for_status()
                raw = image_response.content
        except Exception as exc:
            record["error_category"] = "url_download_failure" if source_kind == "url" else "invalid_image"
            record["exception_type"] = type(exc).__name__
            return record
        try:
            width, height, image_format, digest = validate_image(raw)
        except Exception as exc:
            record["error_category"] = "invalid_image"
            record["exception_type"] = type(exc).__name__
            return record
        record.update(
            {
                "outcome": "success",
                "error_category": None,
                "image_bytes": len(raw),
                "actual_width": width,
                "actual_height": height,
                "image_format": image_format,
                "image_sha256": digest,
                "usable_image_elapsed_s": round(time.perf_counter() - started, 3),
            }
        )
        return record
    except Exception as exc:
        record["error_category"] = classify_exception(exc)
        record["exception_type"] = type(exc).__name__
        record["usable_image_elapsed_s"] = round(time.perf_counter() - started, 3)
        return record


def failure_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record["outcome"] == "success":
            continue
        category = record.get("error_category") or "unknown"
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def summarize_api_round(
    concurrency: int, round_number: int, records: list[dict[str, Any]], wall_seconds: float
) -> dict[str, Any]:
    success = [r for r in records if r["outcome"] == "success"]
    latencies = [float(r["usable_image_elapsed_s"]) for r in success]
    return {
        "concurrency": concurrency,
        "round": round_number,
        "requests": len(records),
        "successes": len(success),
        "failures": len(records) - len(success),
        "success_rate": round(len(success) / len(records), 4) if records else 0,
        "failure_types": failure_counts(records),
        "p50_seconds": nearest_rank(latencies, 50),
        "p95_seconds_nearest_rank": nearest_rank(latencies, 95),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_images_per_minute": round(len(success) / wall_seconds * 60, 3),
        "max_start_skew_ms": max((r["start_skew_ms"] for r in records), default=None),
        "duplicate_image_hashes": len(success) - len({r["image_sha256"] for r in success}),
    }


async def run_api_round(
    client: httpx.AsyncClient,
    endpoint: str,
    key: str,
    prompt: str,
    quality: str,
    size: str,
    concurrency: int,
    round_number: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ready = {"count": 0}
    condition = asyncio.Condition()
    release = asyncio.Event()
    release_clock = {"value": 0.0}
    tasks = [
        asyncio.create_task(
            api_request(
                client, endpoint, key, prompt, quality, size, concurrency,
                round_number, index, ready, condition, release, release_clock
            )
        )
        for index in range(1, concurrency + 1)
    ]
    async with condition:
        await condition.wait_for(lambda: ready["count"] == concurrency)
    release_clock["value"] = time.perf_counter()
    release.set()
    records = await asyncio.gather(*tasks)
    wall = time.perf_counter() - release_clock["value"]
    return records, summarize_api_round(concurrency, round_number, records, wall)


def aggregate_step(concurrency: int, rounds: list[dict[str, Any]]) -> dict[str, Any]:
    requests = sum(r["requests"] for r in rounds)
    successes = sum(r["successes"] for r in rounds)
    wall = sum(r["wall_seconds"] for r in rounds)
    all_failures: dict[str, int] = {}
    for round_summary in rounds:
        for key, value in round_summary["failure_types"].items():
            all_failures[key] = all_failures.get(key, 0) + value
    rpms = [r["throughput_images_per_minute"] for r in rounds]
    p95s = [r["p95_seconds_nearest_rank"] for r in rounds if r["p95_seconds_nearest_rank"] is not None]
    overload = sum(all_failures.get(name, 0) for name in OVERLOAD_ERRORS)
    stable = (
        (all(r["success_rate"] == 1 for r in rounds) if concurrency in {1, 3} else (
            successes / requests >= 0.95 and all(r["success_rate"] >= 0.90 for r in rounds)
        ))
        and (min(rpms) / max(rpms) >= 0.80 if max(rpms) else False)
        and (max(p95s) < 144 if p95s else False)
        and overload / requests <= 0.05
    )
    collapse = (
        successes / requests < 0.80
        or any(r["success_rate"] < 0.70 for r in rounds)
        or overload / requests >= 0.20
    )
    return {
        "concurrency": concurrency,
        "requests": requests,
        "successes": successes,
        "failures": requests - successes,
        "success_rate": round(successes / requests, 4),
        "failure_types": dict(sorted(all_failures.items())),
        "combined_throughput_images_per_minute": round(successes / wall * 60, 3),
        "conservative_throughput_images_per_minute": round(min(rpms), 3),
        "observed_peak_throughput_images_per_minute": round(max(rpms), 3),
        "round_throughputs": rpms,
        "round_p95_seconds": p95s,
        "stable": stable,
        "collapse": collapse,
    }


async def api_staircase(args: argparse.Namespace) -> int:
    settings = Settings.from_env(require_key=True)
    run_dir = args.run_dir or new_run_dir("api")
    run_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{settings.api_base_url}/images/generations"
    steps = [int(value) for value in args.steps.split(",")]
    metadata = {
        "started_at_utc": utc_now(),
        "base_url": settings.api_base_url,
        "proxy_enabled": bool(settings.api_proxy),
        "key_source": "IMAGE_API_KEY or OPENAI_API_KEY environment variable",
        "key_persisted": False,
        "model": "gpt-image-2",
        "quality": args.quality,
        "size": settings.source_size,
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "steps": steps,
        "rounds_per_step": args.rounds,
        "timeouts_seconds": {"connect": 20, "read": 180, "write": 30, "pool": 10},
        "p95_algorithm": "nearest-rank",
    }
    write_json(run_dir / "api-metadata.json", metadata)
    timeout = httpx.Timeout(connect=20, read=180, write=30, pool=10)
    limits = httpx.Limits(max_connections=max(80, max(steps) + 8), max_keepalive_connections=max(80, max(steps) + 8))
    client_args: dict[str, Any] = {"timeout": timeout, "limits": limits, "http2": False}
    if settings.api_proxy:
        client_args["proxy"] = settings.api_proxy
    step_summaries: list[dict[str, Any]] = []
    platform_streak = 0
    async with httpx.AsyncClient(**client_args) as client:
        for step_index, concurrency in enumerate(steps):
            round_summaries: list[dict[str, Any]] = []
            for round_number in range(1, args.rounds + 1):
                records, summary = await run_api_round(
                    client, endpoint, settings.api_key, args.prompt, args.quality,
                    settings.source_size, concurrency, round_number
                )
                append_jsonl(run_dir / "api-requests.jsonl", records)
                round_summaries.append(summary)
                print(
                    f"API C={concurrency} round={round_number} success={summary['successes']}/{summary['requests']} "
                    f"p50={summary['p50_seconds']} p95={summary['p95_seconds_nearest_rank']} "
                    f"rpm={summary['throughput_images_per_minute']} failures={summary['failure_types']}",
                    flush=True,
                )
                if round_number < args.rounds:
                    await asyncio.sleep(args.round_cooldown)
            aggregate = aggregate_step(concurrency, round_summaries)
            aggregate["rounds"] = round_summaries
            step_summaries.append(aggregate)
            write_json(run_dir / "api-staircase-summary.json", {"steps": step_summaries})
            if aggregate["collapse"]:
                print(f"STOP: collapse detected at concurrency={concurrency}", flush=True)
                break
            previous_stable = next((item for item in reversed(step_summaries[:-1]) if item["stable"]), None)
            if previous_stable and aggregate["stable"]:
                gain = aggregate["conservative_throughput_images_per_minute"] / previous_stable["conservative_throughput_images_per_minute"] - 1
                prev_p95 = max(previous_stable["round_p95_seconds"] or [0])
                current_p95 = max(aggregate["round_p95_seconds"] or [0])
                if gain < 0.05 and prev_p95 and current_p95 / prev_p95 >= 1.25:
                    platform_streak += 1
                else:
                    platform_streak = 0
                aggregate["throughput_gain_vs_previous_stable"] = round(gain, 4)
                aggregate["platform_streak"] = platform_streak
                if platform_streak >= 2:
                    print(f"STOP: sustained throughput platform detected at concurrency={concurrency}", flush=True)
                    break
            if step_index < len(steps) - 1:
                await asyncio.sleep(args.step_cooldown)
    stable_steps = [item for item in step_summaries if item["stable"]]
    recommended = max(
        stable_steps,
        key=lambda item: item["conservative_throughput_images_per_minute"],
        default=None,
    )
    final = {
        "completed_at_utc": utc_now(),
        "steps": step_summaries,
        "recommended_stable_step": recommended,
        "collapse_step": next((item for item in step_summaries if item["collapse"]), None),
        "run_dir": str(run_dir.resolve()),
    }
    write_json(run_dir / "api-staircase-summary.json", final)
    print(f"RESULT_DIR={run_dir.resolve()}", flush=True)
    return 0


async def token_samples(args: argparse.Namespace) -> int:
    settings = Settings.from_env(require_key=True)
    run_dir = args.run_dir or new_run_dir("tokens")
    run_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{settings.api_base_url}/images/generations"
    timeout = httpx.Timeout(connect=20, read=180, write=30, pool=10)
    client_args: dict[str, Any] = {
        "timeout": timeout,
        "limits": httpx.Limits(max_connections=16, max_keepalive_connections=16),
    }
    if settings.api_proxy:
        client_args["proxy"] = settings.api_proxy
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    async with httpx.AsyncClient(**client_args) as client:
        for quality in ("low", "medium", "high"):
            records, summary = await run_api_round(
                client, endpoint, settings.api_key, args.prompt, quality,
                settings.source_size, args.samples, 1
            )
            results.extend(records)
            summary["quality"] = quality
            summaries.append(summary)
            print(f"TOKENS quality={quality} success={summary['successes']}/{summary['requests']}", flush=True)
            await asyncio.sleep(args.cooldown)
    append_jsonl(run_dir / "token-samples.jsonl", results)
    write_json(
        run_dir / "token-summary.json",
        {
            "samples": summaries,
            "caveat": "渠道返回token口径存疑，不能直接套官方$30/1M或image-output-token单价",
        },
    )
    print(f"RESULT_DIR={run_dir.resolve()}", flush=True)
    return 0


async def api_soak(args: argparse.Namespace) -> int:
    """Closed-loop sustained load at one chosen concurrency."""
    settings = Settings.from_env(require_key=True)
    run_dir = args.run_dir or new_run_dir("api-soak")
    run_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{settings.api_base_url}/images/generations"
    timeout = httpx.Timeout(connect=20, read=180, write=30, pool=10)
    client_args: dict[str, Any] = {
        "timeout": timeout,
        "limits": httpx.Limits(
            max_connections=max(80, args.concurrency + 8),
            max_keepalive_connections=max(80, args.concurrency + 8),
        ),
    }
    if settings.api_proxy:
        client_args["proxy"] = settings.api_proxy
    start_event = asyncio.Event()
    ready_condition = asyncio.Condition()
    ready_count = {"count": 0}
    records: list[dict[str, Any]] = []
    soak_clock = {"start": 0.0, "deadline": 0.0}

    async def worker(worker_index: int, client: httpx.AsyncClient) -> None:
        async with ready_condition:
            ready_count["count"] += 1
            ready_condition.notify_all()
        await start_event.wait()
        sequence = 0
        while time.perf_counter() < soak_clock["deadline"]:
            sequence += 1
            dummy_condition = asyncio.Condition()
            dummy_ready = {"count": 0}
            already_released = asyncio.Event()
            already_released.set()
            record = await api_request(
                client,
                endpoint,
                settings.api_key,
                args.prompt,
                args.quality,
                settings.source_size,
                args.concurrency,
                0,
                sequence,
                dummy_ready,
                dummy_condition,
                already_released,
                {"value": time.perf_counter()},
            )
            record["soak_worker"] = worker_index
            record["soak_sequence"] = sequence
            record["completed_offset_seconds"] = round(
                time.perf_counter() - soak_clock["start"], 3
            )
            records.append(record)

    async with httpx.AsyncClient(**client_args) as client:
        tasks = [asyncio.create_task(worker(index, client)) for index in range(1, args.concurrency + 1)]
        async with ready_condition:
            await ready_condition.wait_for(lambda: ready_count["count"] == args.concurrency)
        soak_clock["start"] = time.perf_counter()
        soak_clock["deadline"] = soak_clock["start"] + args.duration_seconds
        start_event.set()
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - soak_clock["start"]
    append_jsonl(run_dir / "api-soak-requests.jsonl", records)
    success = [record for record in records if record["outcome"] == "success"]
    latencies = [record["usable_image_elapsed_s"] for record in success]
    minute_buckets: dict[int, dict[str, int]] = {}
    for record in records:
        bucket = int(float(record["completed_offset_seconds"]) // 60)
        item = minute_buckets.setdefault(bucket, {"requests": 0, "successes": 0})
        item["requests"] += 1
        if record["outcome"] == "success":
            item["successes"] += 1
    summary = {
        "completed_at_utc": utc_now(),
        "concurrency": args.concurrency,
        "requested_duration_seconds": args.duration_seconds,
        "actual_wall_seconds_including_drain": round(wall, 3),
        "requests": len(records),
        "successes": len(success),
        "failures": len(records) - len(success),
        "success_rate": round(len(success) / len(records), 4) if records else 0,
        "failure_types": failure_counts(records),
        "p50_seconds": nearest_rank(latencies, 50),
        "p95_seconds_nearest_rank": nearest_rank(latencies, 95),
        "throughput_images_per_minute": round(len(success) / wall * 60, 3),
        "completion_minute_buckets": [
            {"minute_index": index, **minute_buckets[index]}
            for index in sorted(minute_buckets)
        ],
        "definition": "closed-loop; requests started before deadline were drained and included",
    }
    write_json(run_dir / "api-soak-summary.json", summary)
    print(
        f"SOAK C={args.concurrency} success={summary['successes']}/{summary['requests']} "
        f"wall={summary['actual_wall_seconds_including_drain']} rpm={summary['throughput_images_per_minute']} "
        f"failures={summary['failure_types']}",
        flush=True,
    )
    print(f"RESULT_DIR={run_dir.resolve()}", flush=True)
    return 0


def sample_aggregate_vram(pids: list[int], result: dict[str, Any]) -> None:
    pid_list = ",".join(str(pid) for pid in pids)
    script = rf"""
$targetPids = @({pid_list})
$physicalBytes = 17171480576.0
$maxProcessBytes = 0.0
$maxAdapterBytes = 0.0
$sampleCount = 0
$invalidSamples = 0
$intervalTotal = 0.0
$intervalMax = 0.0
$previousTimestamp = $null
while ($true) {{
  $alive = @($targetPids | Where-Object {{ Get-Process -Id $_ -ErrorAction SilentlyContinue }})
  if ($alive.Count -eq 0) {{ break }}
  try {{
    $timestamp = [DateTime]::UtcNow
    if ($previousTimestamp) {{
      $interval = ($timestamp - $previousTimestamp).TotalSeconds
      $intervalTotal += $interval
      if ($interval -gt $intervalMax) {{ $intervalMax = $interval }}
    }}
    $previousTimestamp = $timestamp
    $counterSet = Get-Counter @(
      '\GPU Process Memory(*)\Dedicated Usage',
      '\GPU Adapter Memory(*)\Dedicated Usage'
    ) -ErrorAction Stop
    $items = @()
    $adapterBytes = 0.0
    foreach ($sample in $counterSet.CounterSamples) {{
      if ($sample.Path -like '*GPU Process Memory*' -and $sample.InstanceName -match '^pid_(\d+)_') {{
        $parsedPid = [int]$Matches[1]
        if ($targetPids -contains $parsedPid) {{
          $items += [PSCustomObject]@{{ Pid=$parsedPid; Bytes=$sample.CookedValue }}
        }}
      }} elseif ($sample.Path -like '*GPU Adapter Memory*') {{
        if ($sample.CookedValue -gt $adapterBytes) {{ $adapterBytes = $sample.CookedValue }}
      }}
    }}
    $current = 0.0
    foreach ($group in ($items | Group-Object Pid)) {{
      $current += ($group.Group | Measure-Object Bytes -Maximum).Maximum
    }}
    $sampleCount += 1
    if ($current -gt $physicalBytes -or ($adapterBytes -gt 0 -and $current -gt ($adapterBytes * 1.10))) {{
      $invalidSamples += 1
    }} else {{
      if ($current -gt $maxProcessBytes) {{ $maxProcessBytes = $current }}
    }}
    if ($adapterBytes -gt $maxAdapterBytes -and $adapterBytes -le ($physicalBytes * 1.10)) {{
      $maxAdapterBytes = $adapterBytes
    }}
  }} catch {{}}
}}
$meanInterval = if ($sampleCount -gt 1) {{ $intervalTotal / ($sampleCount - 1) }} else {{ 0.0 }}
[PSCustomObject]@{{
  peak_process_mib = [math]::Round($maxProcessBytes / 1MB, 3)
  peak_adapter_mib = [math]::Round($maxAdapterBytes / 1MB, 3)
  sample_count = $sampleCount
  invalid_samples = $invalidSamples
  mean_interval_seconds = [math]::Round($meanInterval, 3)
  max_interval_seconds = [math]::Round($intervalMax, 3)
}} | ConvertTo-Json -Compress
"""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=900, check=False
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if lines:
            result.update(json.loads(lines[-1]))
        else:
            result["peak_process_mib"] = None
    except Exception as exc:
        result["peak_vram_mib"] = None
        result["sampler_error"] = type(exc).__name__


def wait_and_postprocess(
    process: subprocess.Popen[str],
    started: float,
    intermediate: Path,
    final: Path,
    target: tuple[int, int],
    index: int,
) -> dict[str, Any]:
    output, _ = process.communicate()
    ncnn_done = time.perf_counter()
    record: dict[str, Any] = {
        "index": index,
        "pid": process.pid,
        "exit_code": process.returncode,
        "outcome": "failure",
        "error_category": None,
        "ncnn_seconds": round(ncnn_done - started, 3),
        "postprocess_seconds": None,
        "total_seconds": None,
        "actual_width": None,
        "actual_height": None,
        "file_bytes": None,
        "sha256": None,
    }
    if process.returncode != 0 or not intermediate.is_file():
        lower = output.lower()
        record["error_category"] = "oom" if "out of memory" in lower or "vkallocatememory" in lower else "ncnn_error"
        record["output_tail"] = output.splitlines()[-10:]
        record["total_seconds"] = round(time.perf_counter() - started, 3)
        return record
    post_started = time.perf_counter()
    try:
        _cover_resize(intermediate, final, target)
        fact = inspect_image(final)
        if (fact.width, fact.height) != target:
            raise RuntimeError("pixel verification mismatch")
        record.update(
            {
                "outcome": "success",
                "postprocess_seconds": round(time.perf_counter() - post_started, 3),
                "total_seconds": round(time.perf_counter() - started, 3),
                "actual_width": fact.width,
                "actual_height": fact.height,
                "file_bytes": fact.file_bytes,
                "sha256": fact.sha256,
            }
        )
    except Exception as exc:
        record["error_category"] = "postprocess_error"
        record["exception_type"] = type(exc).__name__
        record["total_seconds"] = round(time.perf_counter() - started, 3)
    return record


def gpu_worker(
    barrier: Any,
    pid_queue: Any,
    result_queue: Any,
    command: list[str],
    intermediate: str,
    final: str,
    target: tuple[int, int],
    index: int,
) -> None:
    """Spawn-safe worker: barrier release -> ncnn -> Pillow exact-size verification."""
    try:
        barrier.wait(timeout=60)
        started_ns = time.perf_counter_ns()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        pid_queue.put({"index": index, "pid": process.pid, "started_ns": started_ns})
        record = wait_and_postprocess(
            process,
            started_ns / 1_000_000_000,
            Path(intermediate),
            Path(final),
            target,
            index,
        )
        record["worker_started_ns"] = started_ns
        record["worker_finished_ns"] = time.perf_counter_ns()
        result_queue.put(record)
    except Exception as exc:
        result_queue.put(
            {
                "index": index,
                "outcome": "failure",
                "error_category": "worker_error",
                "exception_type": type(exc).__name__,
                "worker_started_ns": None,
                "worker_finished_ns": time.perf_counter_ns(),
            }
        )


def gpu_group(
    settings: Settings,
    source: Path,
    output_dir: Path,
    target_name: str,
    concurrency: int,
    round_number: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = TARGET_SIZES[target_name]
    group_dir = output_dir / target_name / f"m{concurrency}" / f"round-{round_number}"
    group_dir.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(concurrency + 1)
    pid_queue = context.Queue()
    result_queue = context.Queue()
    workers: list[multiprocessing.Process] = []
    for index in range(1, concurrency + 1):
        item_dir = group_dir / f"item-{index:02d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        intermediate = item_dir / "upscaled-x4.png"
        final = item_dir / f"final-{target[0]}x{target[1]}.png"
        command = [
            str(settings.upscaler_exe), "-i", str(source.resolve()), "-o", str(intermediate.resolve()),
            "-s", "4", "-t", str(settings.tile_size), "-m", str(settings.upscaler_models.resolve()),
            "-n", settings.upscaler_model, "-g", str(settings.gpu_id), "-j", "1:2:2", "-f", "png", "-v",
        ]
        worker = context.Process(
            target=gpu_worker,
            args=(
                barrier,
                pid_queue,
                result_queue,
                command,
                str(intermediate),
                str(final),
                target,
                index,
            ),
        )
        worker.start()
        workers.append(worker)
    barrier.wait(timeout=60)
    release_ns = time.perf_counter_ns()
    pid_records: list[dict[str, Any]] = []
    for _ in range(concurrency):
        try:
            pid_records.append(pid_queue.get(timeout=30))
        except queue.Empty:
            break
    vram: dict[str, Any] = {}
    sampler = threading.Thread(
        target=sample_aggregate_vram,
        args=([item["pid"] for item in pid_records], vram),
        daemon=True,
    )
    sampler.start()
    records: list[dict[str, Any]] = []
    for _ in range(concurrency):
        try:
            records.append(result_queue.get(timeout=900))
        except queue.Empty:
            break
    for worker in workers:
        worker.join(timeout=10)
    returned_indexes = {record.get("index") for record in records}
    for index, worker in enumerate(workers, start=1):
        if index not in returned_indexes:
            records.append(
                {
                    "index": index,
                    "outcome": "failure",
                    "error_category": "worker_crash",
                    "worker_exit_code": worker.exitcode,
                    "worker_started_ns": None,
                    "worker_finished_ns": time.perf_counter_ns(),
                }
            )
    finish_ns = max(
        (record.get("worker_finished_ns") or release_ns for record in records),
        default=time.perf_counter_ns(),
    )
    group_wall = (finish_ns - release_ns) / 1_000_000_000
    sampler.join(timeout=30)
    records.sort(key=lambda item: item["index"])
    success = [record for record in records if record["outcome"] == "success"]
    durations = [record["total_seconds"] for record in success]
    errors: dict[str, int] = {}
    for record in records:
        if record["outcome"] != "success":
            name = record.get("error_category") or "unknown"
            errors[name] = errors.get(name, 0) + 1
    summary = {
        "target": target_name,
        "target_pixels": target,
        "concurrency": concurrency,
        "round": round_number,
        "requests": concurrency,
        "successes": len(success),
        "failures": concurrency - len(success),
        "success_rate": round(len(success) / concurrency, 4),
        "failure_types": dict(sorted(errors.items())),
        "total_wall_seconds": round(group_wall, 3),
        "average_per_image_seconds": round(statistics.mean(durations), 3) if durations else None,
        "p50_seconds": nearest_rank(durations, 50),
        "p95_seconds_nearest_rank": nearest_rank(durations, 95),
        "throughput_images_per_minute": round(len(success) / group_wall * 60, 3),
        "peak_vram_mib": vram.get("peak_process_mib"),
        "adapter_peak_vram_mib": vram.get("peak_adapter_mib"),
        "vram_sample_count": vram.get("sample_count"),
        "vram_invalid_samples": vram.get("invalid_samples"),
        "vram_mean_sample_interval_seconds": vram.get("mean_interval_seconds"),
        "vram_max_sample_interval_seconds": vram.get("max_interval_seconds"),
        "peak_vram_source": "central Windows GPU counters; exact ncnn PIDs summed per timestamp",
        "max_launch_skew_ms": round(
            (max(item["started_ns"] for item in pid_records) - min(item["started_ns"] for item in pid_records)) / 1_000_000,
            3,
        ) if pid_records else None,
    }
    return records, summary


def gpu_benchmark(args: argparse.Namespace) -> int:
    settings = Settings.from_env(require_key=False)
    run_dir = args.run_dir or new_run_dir("gpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    source_fact = inspect_image(source)
    levels = [int(value) for value in args.levels.split(",")]
    all_summaries: list[dict[str, Any]] = []
    write_json(
        run_dir / "gpu-metadata.json",
        {
            "started_at_utc": utc_now(),
            "source": source_fact.__dict__,
            "levels": levels,
            "rounds": args.rounds,
            "model": settings.upscaler_model,
            "gpu_id": settings.gpu_id,
            "tile_size": settings.tile_size,
        },
    )
    for target_name in ("2k", "4k"):
        warmup_records, warmup_summary = gpu_group(
            settings, source, run_dir / "gpu-artifacts", target_name, 1, 0
        )
        write_json(run_dir / f"gpu-warmup-{target_name}.json", {"summary": warmup_summary, "records": warmup_records})
        print(f"GPU warmup target={target_name} wall={warmup_summary['total_wall_seconds']}", flush=True)
        for concurrency in levels:
            level_summaries: list[dict[str, Any]] = []
            for round_number in range(1, args.rounds + 1):
                records, summary = gpu_group(
                    settings, source, run_dir / "gpu-artifacts", target_name, concurrency, round_number
                )
                append_jsonl(run_dir / "gpu-requests.jsonl", [
                    {"target": target_name, "concurrency": concurrency, "round": round_number, **record}
                    for record in records
                ])
                all_summaries.append(summary)
                level_summaries.append(summary)
                print(
                    f"GPU target={target_name} M={concurrency} round={round_number} "
                    f"success={summary['successes']}/{summary['requests']} wall={summary['total_wall_seconds']} "
                    f"rpm={summary['throughput_images_per_minute']} vram={summary['peak_vram_mib']} "
                    f"failures={summary['failure_types']}",
                    flush=True,
                )
                time.sleep(args.cooldown)
            level_failures = sum(item["failures"] for item in level_summaries)
            level_requests = sum(item["requests"] for item in level_summaries)
            has_oom = any(item["failure_types"].get("oom", 0) for item in level_summaries)
            if has_oom or level_failures / level_requests >= 0.10:
                print(f"STOP GPU {target_name}: failure threshold at M={concurrency}", flush=True)
                break
    by_target: dict[str, Any] = {}
    for target_name in ("2k", "4k"):
        target_rows = [row for row in all_summaries if row["target"] == target_name]
        levels_summary: list[dict[str, Any]] = []
        for concurrency in sorted({row["concurrency"] for row in target_rows}):
            rows = [row for row in target_rows if row["concurrency"] == concurrency]
            throughputs = [row["throughput_images_per_minute"] for row in rows]
            mean_throughput = statistics.mean(throughputs)
            cv = statistics.pstdev(throughputs) / mean_throughput if len(throughputs) > 1 and mean_throughput else 0
            stable = (
                all(row["success_rate"] == 1 for row in rows)
                and all((row["max_launch_skew_ms"] or 0) <= 250 for row in rows)
                and cv <= 0.10
            )
            levels_summary.append(
                {
                    "concurrency": concurrency,
                    "rounds": rows,
                    "stable": stable,
                    "throughput_cv": round(cv, 4),
                    "median_throughput_images_per_minute": round(statistics.median(throughputs), 3),
                    "conservative_throughput_images_per_minute": min(row["throughput_images_per_minute"] for row in rows),
                    "peak_vram_mib": max((row["peak_vram_mib"] or 0) for row in rows) or None,
                }
            )
        stable_levels = [row for row in levels_summary if row["stable"]]
        recommended = max(stable_levels, key=lambda row: row["conservative_throughput_images_per_minute"], default=None)
        max_stable_concurrency = max((row["concurrency"] for row in stable_levels), default=None)
        peak_stable = max((row["median_throughput_images_per_minute"] for row in stable_levels), default=0)
        production = next(
            (row for row in stable_levels if row["median_throughput_images_per_minute"] >= peak_stable * 0.95),
            None,
        )
        by_target[target_name] = {
            "levels": levels_summary,
            "maximum_stable_concurrency": max_stable_concurrency,
            "highest_conservative_throughput_level": recommended,
            "recommended_production_level": production,
        }
    write_json(
        run_dir / "gpu-summary.json",
        {"completed_at_utc": utc_now(), "targets": by_target, "all_rounds": all_summaries},
    )
    print(f"RESULT_DIR={run_dir.resolve()}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Capacity benchmark for gpt-image-2 + local Real-ESRGAN")
    commands = root.add_subparsers(dest="command", required=True)
    api = commands.add_parser("api")
    api.add_argument("--steps", default="1,3,5,8,12,16,24,32")
    api.add_argument("--rounds", type=int, default=2)
    api.add_argument("--quality", choices=("low", "medium", "high"), default="low")
    api.add_argument("--prompt", default=DEFAULT_PROMPT)
    api.add_argument("--round-cooldown", type=float, default=5)
    api.add_argument("--step-cooldown", type=float, default=10)
    api.add_argument("--run-dir", type=Path)

    tokens = commands.add_parser("tokens")
    tokens.add_argument("--samples", type=int, default=2)
    tokens.add_argument("--prompt", default=DEFAULT_PROMPT)
    tokens.add_argument("--cooldown", type=float, default=5)
    tokens.add_argument("--run-dir", type=Path)

    soak = commands.add_parser("soak")
    soak.add_argument("--concurrency", type=int, required=True)
    soak.add_argument("--duration-seconds", type=float, default=300)
    soak.add_argument("--quality", choices=("low", "medium", "high"), default="low")
    soak.add_argument("--prompt", default=DEFAULT_PROMPT)
    soak.add_argument("--run-dir", type=Path)

    gpu = commands.add_parser("gpu")
    gpu.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "runs" / "20260711T064118Z-low-4k-bef1003d" / "source.png",
    )
    gpu.add_argument("--levels", default="1,2,4,6,8")
    gpu.add_argument("--rounds", type=int, default=3)
    gpu.add_argument("--cooldown", type=float, default=3)
    gpu.add_argument("--run-dir", type=Path)
    return root


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    if args.command == "api":
        return asyncio.run(api_staircase(args))
    if args.command == "tokens":
        return asyncio.run(token_samples(args))
    if args.command == "soak":
        return asyncio.run(api_soak(args))
    if args.command == "gpu":
        return gpu_benchmark(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
