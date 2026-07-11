from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .generator import ImageGenerationError
from .pipeline import ImagePipeline
from .upscaler import UpscaleError


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32000)
    tier: str = Field(default="low", pattern="^(low|medium|high)$")
    target: str = Field(default="4k", pattern="^(2k|4k)$")


class BatchGenerateRequest(BaseModel):
    items: list[GenerateRequest] = Field(min_length=1, max_length=32)


app = FastAPI(title="gpt-image-2 + Real-ESRGAN Pipeline", version="1.0.0")
_pipeline_lock = threading.Lock()


@app.get("/health")
def health() -> dict[str, object]:
    settings = Settings.from_env(require_key=False)
    return {
        "status": "ok",
        "api_key_configured": bool(settings.api_key),
        "model": settings.model,
        "upscaler_installed": settings.upscaler_exe.is_file(),
        "targets": ["2k", "4k"],
        "tiers": ["low", "medium", "high"],
    }


@app.post("/v1/generate")
def generate(request: GenerateRequest) -> dict[str, object]:
    try:
        with _pipeline_lock:
            result = ImagePipeline().generate_and_upscale(
                request.prompt, request.tier, request.target
            )
        return result.to_dict()
    except (ValueError, RuntimeError, ImageGenerationError, UpscaleError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/generate/batch")
def generate_batch(request: BatchGenerateRequest) -> dict[str, object]:
    """Run items serially so concurrent requests cannot overcommit GPU memory."""
    results: list[dict[str, object]] = []
    try:
        with _pipeline_lock:
            pipeline = ImagePipeline()
            for item in request.items:
                results.append(
                    pipeline.generate_and_upscale(
                        item.prompt, item.tier, item.target
                    ).to_dict()
                )
        return {"count": len(results), "results": results}
    except (ValueError, RuntimeError, ImageGenerationError, UpscaleError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Batch stopped after {len(results)} completed item(s): {exc}",
        ) from exc
