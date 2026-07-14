from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from image_pipeline.config import TARGET_SIZES, Settings
from image_pipeline.upscaler import RealEsrganUpscaler


def _edge_mean(image: Image.Image, axis: str, position: int) -> float:
    if axis == "x":
        before = image.crop((position - 1, 0, position, image.height))
        after = image.crop((position, 0, position + 1, image.height))
    else:
        before = image.crop((0, position - 1, image.width, position))
        after = image.crop((0, position, image.width, position + 1))
    return sum(ImageStat.Stat(ImageChops.difference(before, after)).mean) / 3


def seam_score(path: Path, source_tile: int, scale: int = 4) -> dict[str, float | int | None]:
    """Compare gradient energy on expected tile joins with nearby control lines."""
    if source_tile <= 0:
        return {"boundaries": 0, "boundary_mean": None, "control_mean": None, "ratio": None}
    spacing = source_tile * scale
    boundary: list[float] = []
    control: list[float] = []
    with Image.open(path) as source:
        image = source.convert("RGB")
        for axis, limit in (("x", image.width), ("y", image.height)):
            for position in range(spacing, limit, spacing):
                if 3 <= position < limit - 3:
                    boundary.append(_edge_mean(image, axis, position))
                    control.extend(
                        (
                            _edge_mean(image, axis, position - 2),
                            _edge_mean(image, axis, position + 2),
                        )
                    )
    boundary_mean = sum(boundary) / len(boundary) if boundary else None
    control_mean = sum(control) / len(control) if control else None
    return {
        "boundaries": len(boundary),
        "boundary_mean": round(boundary_mean, 6) if boundary_mean is not None else None,
        "control_mean": round(control_mean, 6) if control_mean is not None else None,
        "ratio": (
            round(boundary_mean / control_mean, 6)
            if boundary_mean is not None and control_mean
            else None
        ),
    }


class AdapterMemoryMonitor:
    def __init__(self, root: Path):
        self.stop_path = root / ".stop-gpu-monitor"
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.stop_path.unlink(missing_ok=True)
        escaped = str(self.stop_path).replace("'", "''")
        script = rf"""
$stopPath = '{escaped}'
$first = $null
$maxBytes = 0.0
while (-not (Test-Path -LiteralPath $stopPath)) {{
  try {{
    $values = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
      ForEach-Object {{ $_.CookedValue }}
    if ($values) {{
      $current = ($values | Measure-Object -Sum).Sum
      if ($null -eq $first) {{ $first = $current }}
      if ($current -gt $maxBytes) {{ $maxBytes = $current }}
    }}
  }} catch {{}}
  Start-Sleep -Milliseconds 200
}}
$firstMiB = if ($null -eq $first) {{ 0 }} else {{ $first / 1MB }}
[Console]::WriteLine(
  ([string]::Format(
    [Globalization.CultureInfo]::InvariantCulture,
    '{{0:F3}}|{{1:F3}}',
    $firstMiB,
    $maxBytes / 1MB
  ))
)
"""
        self.process = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        time.sleep(0.5)

    def stop(self) -> dict[str, float | None]:
        if self.process is None:
            return {"adapter_baseline_mib": None, "adapter_peak_mib": None, "adapter_delta_mib": None}
        self.stop_path.touch()
        stdout, _ = self.process.communicate(timeout=30)
        self.stop_path.unlink(missing_ok=True)
        try:
            baseline, peak = (float(value) for value in stdout.strip().split("|", 1))
        except (ValueError, TypeError):
            return {"adapter_baseline_mib": None, "adapter_peak_mib": None, "adapter_delta_mib": None}
        return {
            "adapter_baseline_mib": round(baseline, 3),
            "adapter_peak_mib": round(peak, 3),
            "adapter_delta_mib": round(max(0.0, peak - baseline), 3),
        }


def run_one(source: Path, output: Path, target: str, tile: int) -> dict[str, object]:
    settings = replace(Settings.from_env(require_key=False), tile_size=tile)
    result = RealEsrganUpscaler(settings).upscale(source, output, TARGET_SIZES[target])
    record = asdict(result)
    record["tile_size"] = tile
    record["intermediate_seam_score"] = seam_score(
        Path(result.intermediate_image.path), tile
    )
    return record


def run_concurrent(
    source: Path, output: Path, target: str, tile: int, concurrency: int
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    barrier = threading.Barrier(concurrency)
    monitor = AdapterMemoryMonitor(output)

    def work(index: int) -> dict[str, object]:
        barrier.wait()
        return run_one(source, output / f"worker-{index:02d}", target, tile)

    monitor.start()
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(work, range(1, concurrency + 1)))
    finally:
        memory = monitor.stop()
    return {
        "source": str(source.resolve()),
        "target": target,
        "tile_size": tile,
        "concurrency": concurrency,
        "wall_seconds": round(time.perf_counter() - started, 3),
        **memory,
        "workers": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Real-ESRGAN tile seams and concurrency")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target", choices=("2k", "4k"), default="4k")
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.tile < 0:
        parser.error("--tile must be non-negative")
    if not 1 <= args.concurrency <= 5:
        parser.error("--concurrency must be from 1 to 5")
    args.output_root.mkdir(parents=True, exist_ok=True)
    record = run_concurrent(
        args.source.resolve(), args.output_root.resolve(), args.target, args.tile, args.concurrency
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
