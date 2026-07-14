import io

from PIL import Image

from capacity_benchmark import classify_http, nearest_rank, validate_image


def test_nearest_rank_percentiles() -> None:
    assert nearest_rank([1, 2, 3, 4, 5], 50) == 3
    assert nearest_rank([1, 2, 3, 4, 5], 95) == 5
    assert nearest_rank([], 95) is None


def test_http_failure_categories() -> None:
    assert classify_http(429) == "http_429"
    assert classify_http(503) == "http_503"
    assert classify_http(401) == "http_4xx_other"


def test_validate_image_reads_real_pixels() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (37, 19), "red").save(buffer, format="PNG")
    width, height, image_format, digest = validate_image(buffer.getvalue())
    assert (width, height) == (37, 19)
    assert image_format == "PNG"
    assert len(digest) == 64
