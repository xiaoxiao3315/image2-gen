from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from .config import Settings, TIERS
from .image_io import extension_for_image, inspect_image
from .models import GenerationResult


class ImageGenerationError(RuntimeError):
    pass


class NativeBatchUnsupported(ImageGenerationError):
    """The provider explicitly rejected image generation with n > 1."""


class NativeBatchIncomplete(ImageGenerationError):
    """HTTP 200 returned some, but fewer than the requested native batch."""

    def __init__(
        self,
        message: str,
        partial_results: list[GenerationResult],
        requested_count: int,
    ) -> None:
        super().__init__(message)
        self.partial_results = partial_results
        self.requested_count = requested_count


DEFAULT_API_MAX_ATTEMPTS = 3
MAX_API_MAX_ATTEMPTS = 3
API_RETRY_BASE_SECONDS = 1.0


def _api_max_attempts() -> int:
    raw = os.getenv("IMAGE_API_MAX_ATTEMPTS", str(DEFAULT_API_MAX_ATTEMPTS)).strip()
    try:
        attempts = int(raw)
    except ValueError as exc:
        raise ValueError("IMAGE_API_MAX_ATTEMPTS must be an integer from 1 to 3") from exc
    if not 1 <= attempts <= MAX_API_MAX_ATTEMPTS:
        raise ValueError("IMAGE_API_MAX_ATTEMPTS must be an integer from 1 to 3")
    return attempts


def _retry_delay_seconds(failed_attempt: int) -> float:
    return API_RETRY_BASE_SECONDS * (2 ** (failed_attempt - 1))


def _numeric_tree(value: Any) -> Any:
    """Keep provider usage numbers without persisting arbitrary response text."""
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, item in value.items()
            if (cleaned := _numeric_tree(item)) is not None
        }
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _safe_usage(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None
    cleaned = _numeric_tree(payload["usage"])
    if not isinstance(cleaned, dict):
        return None
    cleaned["usage_provenance"] = "provider_reported_unverified"
    cleaned["pricing_eligible"] = False
    cleaned["caveat"] = "provider token口径未经验证，不能直接套官方图像token单价"
    return cleaned


def _pop_image(value: Any, trail: str = "root") -> tuple[str, str, str] | None:
    """Remove and return one candidate so processed base64 is released promptly."""
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{trail}.{key}"
            key_lower = str(key).lower()
            if isinstance(item, str):
                if key_lower in {"b64_json", "image_base64", "base64"}:
                    value[key] = None
                    return "base64", item, child
                if item.startswith("data:image/") and ";base64," in item:
                    value[key] = None
                    return "base64", item.split(",", 1)[1], child
                if key_lower in {"url", "image_url"} and item.startswith(
                    ("http://", "https://")
                ):
                    value[key] = None
                    return "url", item, child
            else:
                candidate = _pop_image(item, child)
                if candidate is not None:
                    return candidate
    elif isinstance(value, list):
        for index, item in enumerate(value):
            candidate = _pop_image(item, f"{trail}[{index}]")
            if candidate is not None:
                return candidate
    return None


def _native_batch_is_explicitly_unsupported(response: requests.Response) -> bool:
    """Inspect an error only to classify it; never return or log its body."""
    if response.status_code not in {400, 422}:
        return False
    try:
        body = json.dumps(response.json(), ensure_ascii=False)
    except (ValueError, TypeError):
        body = str(getattr(response, "text", ""))
    normalized = " ".join(body.lower().split())[:20_000]
    mentions_n = bool(
        re.search(
            r"(?:parameter|field|argument|param)?\s*['\"`]?n['\"`]?\s*"
            r"(?:is\s*)?(?:unsupported|not supported|must (?:be|equal)|only supports?)",
            normalized,
        )
        or re.search(
            r"(?:unsupported|not supported|only supports?|must (?:be|equal))[^.]{0,80}"
            r"(?:parameter|field|argument|param)?\s*['\"`]?n['\"`]?\b",
            normalized,
        )
        or re.search(r"\bn\s*(?:=|must be)\s*1\b", normalized)
    )
    return mentions_n


class GptImageGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        if settings.api_proxy:
            self.session.proxies.update(
                {"http": settings.api_proxy, "https": settings.api_proxy}
            )

    def generate(
        self,
        prompt: str,
        quality: str,
        output_dir: Path,
        idempotency_key: str | None = None,
    ) -> GenerationResult:
        return self.generate_many(
            prompt,
            quality,
            output_dir,
            n=1,
            idempotency_key=idempotency_key,
        )[0]

    def generate_many(
        self,
        prompt: str,
        quality: str,
        output_dir: Path,
        n: int,
        idempotency_key: str | None = None,
    ) -> list[GenerationResult]:
        if quality not in TIERS:
            raise ValueError(f"quality must be one of {', '.join(TIERS)}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 1 <= n <= 5:
            raise ValueError("n must be an integer from 1 to 5")
        if idempotency_key is not None and not re.fullmatch(
            r"[A-Za-z0-9_.:-]{8,200}", idempotency_key
        ):
            raise ValueError("idempotency_key must be 8-200 safe ASCII characters")
        output_dir.mkdir(parents=True, exist_ok=True)

        request_body = {
            "model": self.settings.model,
            "prompt": prompt,
            "quality": quality,
            "size": self.settings.source_size,
            "n": n,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            # Reuse this value for every retry of this logical generation request.
            # This is the primary safeguard against duplicate work/billing when a
            # timeout leaves the provider-side outcome unknown.
            "Idempotency-Key": idempotency_key or uuid.uuid4().hex,
        }
        endpoint = f"{self.settings.api_base_url}/images/generations"

        api_started = time.perf_counter()
        max_attempts = _api_max_attempts()
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.post(
                    endpoint,
                    headers=headers,
                    json=request_body,
                    timeout=(
                        self.settings.api_connect_timeout_seconds,
                        self.settings.api_timeout_seconds,
                    ),
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= max_attempts:
                    raise ImageGenerationError(
                        f"Image API request failed after {attempt} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
                time.sleep(_retry_delay_seconds(attempt))
                continue
            except requests.RequestException as exc:
                # Other request failures are not known-transient and must not be replayed.
                raise ImageGenerationError(
                    f"Image API request failed: {type(exc).__name__}"
                ) from exc

            if response.status_code != 503 or attempt >= max_attempts:
                break
            # Do not inspect or log the gateway response body. A 503 is the only
            # HTTP status replayed; the same Idempotency-Key is retained above.
            time.sleep(_retry_delay_seconds(attempt))
        api_seconds = time.perf_counter() - api_started

        if response.status_code != 200:
            if n > 1 and _native_batch_is_explicitly_unsupported(response):
                raise NativeBatchUnsupported(
                    "Image API explicitly does not support native n > 1"
                )
            # Deliberately avoid persisting or echoing arbitrary gateway response bodies.
            raise ImageGenerationError(
                f"Image API returned HTTP {response.status_code}; response body omitted for secret safety"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ImageGenerationError("Image API returned non-JSON HTTP 200") from exc

        request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("request-id")
        )
        usage = _safe_usage(payload)
        # requests keeps the complete response bytes in addition to the decoded JSON.
        # Drop that duplicate before decoding potentially large base64 candidates.
        if hasattr(response, "_content"):
            response._content = b""

        results: list[GenerationResult] = []
        for index in range(n):
            candidate = _pop_image(payload)
            if candidate is None:
                raise NativeBatchIncomplete(
                    f"Image API HTTP 200 response contained only {len(results)} of {n} images",
                    partial_results=results,
                    requested_count=n,
                )
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
                        value,
                        timeout=self.settings.api_timeout_seconds,
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
            stem = "source" if n == 1 else f"source-{index + 1}"
            image_path = output_dir / f"{stem}{extension}"
            image_path.write_bytes(raw)
            del raw
            image = inspect_image(image_path)
            results.append(
                GenerationResult(
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
                    usage=usage,
                    image=image,
                )
            )
        return results
