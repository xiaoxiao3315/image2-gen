from pathlib import Path

from PIL import Image

from image_pipeline.image_io import inspect_image
from image_pipeline.config import _normalize_api_base_url
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
