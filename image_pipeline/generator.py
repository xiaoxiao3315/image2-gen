from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from .config import Settings, TIERS
from .image_io import extension_for_image, inspect_image
from .models import GenerationResult


class ImageGenerationError(RuntimeError):
    pass


def _iter_images(value: Any, trail: str = "root") -> Iterable[tuple[str, str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{trail}.{key}"
            key_lower = str(key).lower()
            if isinstance(item, str):
                if key_lower in {"b64_json", "image_base64", "base64"}:
                    yield "base64", item, child
                elif item.startswith("data:image/") and ";base64," in item:
                    yield "base64", item.split(",", 1)[1], child
                elif key_lower in {"url", "image_url"} and item.startswith(
                    ("http://", "https://")
                ):
                    yield "url", item, child
            else:
                yield from _iter_images(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_images(item, f"{trail}[{index}]")


class GptImageGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        if settings.api_proxy:
            self.session.proxies.update(
                {"http": settings.api_proxy, "https": settings.api_proxy}
            )

    def generate(self, prompt: str, quality: str, output_dir: Path) -> GenerationResult:
        if quality not in TIERS:
            raise ValueError(f"quality must be one of {', '.join(TIERS)}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        output_dir.mkdir(parents=True, exist_ok=True)

        request_body = {
            "model": self.settings.model,
            "prompt": prompt,
            "quality": quality,
            "size": self.settings.source_size,
            "n": 1,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.settings.api_base_url}/images/generations"

        api_started = time.perf_counter()
        try:
            response = self.session.post(
                endpoint,
                headers=headers,
                json=request_body,
                timeout=self.settings.api_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ImageGenerationError(f"Image API request failed: {type(exc).__name__}") from exc
        api_seconds = time.perf_counter() - api_started

        if response.status_code != 200:
            # Deliberately avoid persisting or echoing arbitrary gateway response bodies.
            raise ImageGenerationError(
                f"Image API returned HTTP {response.status_code}; response body omitted for secret safety"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ImageGenerationError("Image API returned non-JSON HTTP 200") from exc

        candidate = next(iter(_iter_images(payload)), None)
        if candidate is None:
            raise ImageGenerationError("Image API HTTP 200 response contained no image URL or base64")

        kind, value, trail = candidate
        download_started = time.perf_counter()
        if kind == "base64":
            try:
                raw = base64.b64decode(value, validate=False)
            except Exception as exc:
                raise ImageGenerationError("Image base64 could not be decoded") from exc
        else:
            try:
                image_response = self.session.get(
                    value, timeout=self.settings.api_timeout_seconds
                )
                image_response.raise_for_status()
                raw = image_response.content
            except requests.RequestException as exc:
                raise ImageGenerationError("Returned image URL could not be downloaded") from exc
        download_seconds = time.perf_counter() - download_started

        try:
            extension = extension_for_image(raw)
        except Exception as exc:
            raise ImageGenerationError("Returned bytes are not a Pillow-decodable image") from exc
        image_path = output_dir / f"source{extension}"
        image_path.write_bytes(raw)
        image = inspect_image(image_path)

        request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("request-id")
        )
        return GenerationResult(
            requested_model=self.settings.model,
            requested_quality=quality,
            requested_size=self.settings.source_size,
            request_body=request_body,
            status_code=response.status_code,
            request_id=request_id,
            api_seconds=round(api_seconds, 3),
            download_seconds=round(download_seconds, 3),
            total_seconds=round(api_seconds + download_seconds, 3),
            response_image_trail=trail,
            image=image,
        )
