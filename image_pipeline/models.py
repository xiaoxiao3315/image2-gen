from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageMetadata:
    path: str
    width: int
    height: int
    format: str | None
    mode: str
    file_bytes: int
    sha256: str


@dataclass(frozen=True)
class GenerationResult:
    requested_model: str
    requested_quality: str
    requested_size: str
    request_body: dict[str, Any]
    status_code: int
    request_id: str | None
    api_seconds: float
    download_seconds: float
    total_seconds: float
    response_image_trail: str
    usage: dict[str, Any] | None
    image: ImageMetadata


@dataclass(frozen=True)
class UpscaleResult:
    model: str
    device: str
    gpu_id: int
    input_image: ImageMetadata
    intermediate_image: ImageMetadata
    output_image: ImageMetadata
    upscale_seconds: float
    postprocess_seconds: float
    total_seconds: float
    peak_vram_mib: float | None
    peak_vram_source: str
    fit: str


@dataclass(frozen=True)
class CostResult:
    api_cost_cny: float | None
    api_cost_status: str
    local_upscale_cost_cny_estimate: float
    local_cost_basis: str
    total_cost_cny: float | None
    target_cny: float
    target_status: str


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    run_dir: str
    tier: str
    target: str
    target_pixels: tuple[int, int]
    generation: GenerationResult
    upscale: UpscaleResult
    cost: CostResult
    total_seconds: float
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def path_string(path: Path) -> str:
    return str(path.resolve())
