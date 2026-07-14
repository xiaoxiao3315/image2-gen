from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_SIZES: dict[str, tuple[int, int]] = {
    "2k": (2048, 2048),
    "4k": (3840, 2160),
}
TIERS = ("low", "medium", "high")


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _nonnegative_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalize_api_base_url(value: str) -> str:
    """Accept either a host root or an OpenAI-compatible /v1 base URL."""
    base = value.strip().rstrip("/")
    if not base:
        raise ValueError("IMAGE_API_BASE_URL must not be empty")
    return base if base.lower().endswith("/v1") else f"{base}/v1"


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    api_key: str
    api_proxy: str | None
    api_connect_timeout_seconds: float
    api_timeout_seconds: float
    output_root: Path
    model: str
    source_size: str
    upscaler_exe: Path
    upscaler_models: Path
    upscaler_model: str
    gpu_id: int
    tile_size: int
    gpu_power_watts_estimate: float
    electricity_cny_per_kwh: float
    api_cost_cny: dict[str, float | None]

    @classmethod
    def from_env(cls, require_key: bool = True) -> "Settings":
        key = os.getenv("IMAGE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if require_key and not key:
            raise RuntimeError(
                "Missing IMAGE_API_KEY (OPENAI_API_KEY is accepted as a fallback). "
                "Set it in the environment; never put the key in a file or CLI argument."
            )

        tool_root = PROJECT_ROOT / "tools" / "realesrgan-ncnn-vulkan"
        proxy = os.getenv("IMAGE_API_PROXY", "").strip()
        return cls(
            api_base_url=_normalize_api_base_url(
                os.getenv("IMAGE_API_BASE_URL", "https://your-image-api-endpoint/v1")
            ),
            api_key=key,
            api_proxy=proxy or None,
            api_connect_timeout_seconds=float(
                os.getenv("IMAGE_API_CONNECT_TIMEOUT_SECONDS", "20")
            ),
            api_timeout_seconds=float(os.getenv("IMAGE_API_TIMEOUT_SECONDS", "300")),
            output_root=Path(os.getenv("IMAGE_OUTPUT_ROOT", PROJECT_ROOT / "runs")),
            model=os.getenv("IMAGE_MODEL", "gpt-image-2"),
            source_size=os.getenv("IMAGE_SOURCE_SIZE", "1536x1024"),
            upscaler_exe=Path(
                os.getenv(
                    "REALESRGAN_EXE", tool_root / "realesrgan-ncnn-vulkan.exe"
                )
            ),
            upscaler_models=Path(
                os.getenv("REALESRGAN_MODELS", tool_root / "models")
            ),
            upscaler_model=os.getenv("REALESRGAN_MODEL", "realesrgan-x4plus"),
            gpu_id=int(os.getenv("REALESRGAN_GPU_ID", "0")),
            # 0 asks Real-ESRGAN NCNN/Vulkan to process the whole image. This
            # avoids visible seams caused by stitching small tiles on GPUs
            # with enough VRAM. Operators can still set a positive tile size.
            tile_size=_nonnegative_int("REALESRGAN_TILE_SIZE", 0),
            gpu_power_watts_estimate=float(os.getenv("GPU_POWER_WATTS", "175")),
            electricity_cny_per_kwh=float(
                os.getenv("ELECTRICITY_CNY_PER_KWH", "0.60")
            ),
            api_cost_cny={
                tier: _optional_float(f"IMAGE_API_COST_CNY_{tier.upper()}")
                for tier in TIERS
            },
        )
