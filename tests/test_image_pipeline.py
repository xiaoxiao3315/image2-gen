from pathlib import Path

from PIL import Image

from image_pipeline.image_io import inspect_image
import pytest

from image_pipeline.config import Settings, _normalize_api_base_url
from image_pipeline.upscaler import _cover_resize


def test_inspect_image_reads_real_pixels(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (321, 123), "navy").save(path)
    fact = inspect_image(path)
    assert (fact.width, fact.height) == (321, 123)
    assert fact.file_bytes == path.stat().st_size
    assert len(fact.sha256) == 64


def test_cover_resize_outputs_exact_4k(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "final.png"
    Image.new("RGB", (1536, 1024), "orange").save(source)
    _cover_resize(source, output, (3840, 2160))
    fact = inspect_image(output)
    assert (fact.width, fact.height) == (3840, 2160)


def test_cover_resize_outputs_exact_2k(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "final.png"
    Image.new("RGB", (1536, 1024), "green").save(source)
    _cover_resize(source, output, (2048, 2048))
    fact = inspect_image(output)
    assert (fact.width, fact.height) == (2048, 2048)


def test_base_url_accepts_root_or_v1() -> None:
    assert _normalize_api_base_url("https://example.test") == "https://example.test/v1"
    assert _normalize_api_base_url("https://example.test/v1/") == "https://example.test/v1"


def test_realesrgan_tile_defaults_to_full_frame_and_remains_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REALESRGAN_TILE_SIZE", raising=False)
    assert Settings.from_env(require_key=False).tile_size == 0
    monkeypatch.setenv("REALESRGAN_TILE_SIZE", "512")
    assert Settings.from_env(require_key=False).tile_size == 512
    monkeypatch.setenv("REALESRGAN_TILE_SIZE", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        Settings.from_env(require_key=False)
