from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .models import ImageMetadata, path_string


def inspect_image(path: Path) -> ImageMetadata:
    """Decode the saved file with Pillow and report facts from the file itself."""
    path = path.resolve()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        image_format = image.format
        mode = image.mode
    raw = path.read_bytes()
    return ImageMetadata(
        path=path_string(path),
        width=width,
        height=height,
        format=image_format,
        mode=mode,
        file_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def extension_for_image(raw: bytes) -> str:
    from io import BytesIO

    with Image.open(BytesIO(raw)) as image:
        image.load()
        return {
            "PNG": ".png",
            "JPEG": ".jpg",
            "WEBP": ".webp",
        }.get(image.format or "", ".img")
