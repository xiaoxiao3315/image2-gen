from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import Settings
from .image_io import inspect_image
from .models import UpscaleResult


class UpscaleError(RuntimeError):
    pass


@dataclass
class _VramSample:
    peak_mib: float | None = None
    error: str | None = None


def _sample_windows_vram(pid: int, result: _VramSample) -> None:
    """Sample per-process dedicated GPU memory via Windows performance counters."""
    script = rf"""
$targetPid = {pid}
$maxBytes = 0.0
while (Get-Process -Id $targetPid -ErrorAction SilentlyContinue) {{
  try {{
    $values = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
      Where-Object {{ $_.InstanceName -like ('pid_' + $targetPid + '_*') }} |
      ForEach-Object {{ $_.CookedValue }}
    if ($values) {{
      $current = ($values | Measure-Object -Maximum).Maximum
      if ($current -gt $maxBytes) {{ $maxBytes = $current }}
    }}
  }} catch {{}}
  Start-Sleep -Milliseconds 200
}}
[Console]::WriteLine([math]::Round($maxBytes / 1MB, 3).ToString([Globalization.CultureInfo]::InvariantCulture))
"""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if lines:
            value = float(lines[-1])
            result.peak_mib = value if value > 0 else None
    except Exception as exc:  # Metric failure must not fail image delivery.
        result.error = type(exc).__name__


def _cover_resize(source: Path, output: Path, target: tuple[int, int]) -> None:
    target_width, target_height = target
    with Image.open(source) as image:
        image.load()
        width, height = image.size
        source_ratio = width / height
        target_ratio = target_width / target_height
        if source_ratio > target_ratio:
            crop_width = max(1, round(height * target_ratio))
            left = (width - crop_width) // 2
            box = (left, 0, left + crop_width, height)
        else:
            crop_height = max(1, round(width / target_ratio))
            top = (height - crop_height) // 2
            box = (0, top, width, top + crop_height)
        cropped = image.crop(box)
        final = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        if final.mode not in {"RGB", "RGBA"}:
            final = final.convert("RGB")
        output.parent.mkdir(parents=True, exist_ok=True)
        final.save(output, format="PNG", optimize=False)


class RealEsrganUpscaler:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _validate_install(self) -> None:
        if not self.settings.upscaler_exe.is_file():
            raise UpscaleError(f"Real-ESRGAN executable not found: {self.settings.upscaler_exe}")
        model_param = self.settings.upscaler_models / f"{self.settings.upscaler_model}.param"
        model_bin = self.settings.upscaler_models / f"{self.settings.upscaler_model}.bin"
        if not model_param.is_file() or not model_bin.is_file():
            raise UpscaleError(f"Real-ESRGAN model files not found for {self.settings.upscaler_model}")

    def upscale(
        self,
        source: Path,
        output_dir: Path,
        target: tuple[int, int],
        fit: str = "cover",
    ) -> UpscaleResult:
        self._validate_install()
        if fit != "cover":
            raise ValueError("Only fit=cover is currently supported")
        input_image = inspect_image(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / "upscaled-x4.png"
        final_path = output_dir / f"final-{target[0]}x{target[1]}.png"

        command = [
            str(self.settings.upscaler_exe),
            "-i",
            str(source.resolve()),
            "-o",
            str(intermediate_path.resolve()),
            "-s",
            "4",
            "-t",
            str(self.settings.tile_size),
            "-m",
            str(self.settings.upscaler_models.resolve()),
            "-n",
            self.settings.upscaler_model,
            "-g",
            str(self.settings.gpu_id),
            "-j",
            "1:2:2",
            "-f",
            "png",
            "-v",
        ]

        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        vram_sample = _VramSample()
        sampler = threading.Thread(
            target=_sample_windows_vram, args=(process.pid, vram_sample), daemon=True
        )
        sampler.start()
        output_text, _ = process.communicate()
        upscale_seconds = time.perf_counter() - started
        sampler.join(timeout=15)

        if process.returncode != 0 or not intermediate_path.is_file():
            safe_tail = "\n".join(output_text.splitlines()[-20:])
            raise UpscaleError(
                f"Real-ESRGAN exited with {process.returncode}. Output tail:\n{safe_tail}"
            )
        device_match = re.search(r"\[0 ([^\]]+)\]", output_text)
        device = device_match.group(1) if device_match else f"GPU {self.settings.gpu_id}"
        intermediate_image = inspect_image(intermediate_path)

        post_started = time.perf_counter()
        _cover_resize(intermediate_path, final_path, target)
        postprocess_seconds = time.perf_counter() - post_started
        output_image = inspect_image(final_path)
        if (output_image.width, output_image.height) != target:
            raise UpscaleError(
                f"Pillow verification failed: expected {target}, got "
                f"{output_image.width}x{output_image.height}"
            )

        return UpscaleResult(
            model=self.settings.upscaler_model,
            device=device,
            gpu_id=self.settings.gpu_id,
            input_image=input_image,
            intermediate_image=intermediate_image,
            output_image=output_image,
            upscale_seconds=round(upscale_seconds, 3),
            postprocess_seconds=round(postprocess_seconds, 3),
            total_seconds=round(upscale_seconds + postprocess_seconds, 3),
            peak_vram_mib=vram_sample.peak_mib,
            peak_vram_source=(
                "Windows GPU Process Memory/Dedicated Usage performance counter"
                if vram_sample.peak_mib is not None
                else "unavailable (NVML failed; Windows counter produced no sample)"
            ),
            fit=fit,
        )

    def upscale_batch(
        self,
        sources: list[Path],
        output_root: Path,
        target: tuple[int, int],
        fit: str = "cover",
    ) -> list[UpscaleResult]:
        """Process a batch serially to keep GPU memory usage predictable."""
        results: list[UpscaleResult] = []
        output_root.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(sources, start=1):
            item_dir = output_root / f"{index:04d}-{source.stem}"
            results.append(self.upscale(source, item_dir, target, fit=fit))
        return results
