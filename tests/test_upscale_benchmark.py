from pathlib import Path

from PIL import Image, ImageDraw

from scripts.benchmark_upscale_tiles import seam_score


def test_seam_score_detects_expected_tile_boundary(tmp_path: Path) -> None:
    path = tmp_path / "grid.png"
    image = Image.new("RGB", (24, 24), "gray")
    draw = ImageDraw.Draw(image)
    draw.line((8, 0, 8, 23), fill="white")
    draw.line((0, 8, 23, 8), fill="white")
    image.save(path)

    score = seam_score(path, source_tile=2, scale=4)
    assert score["boundaries"] == 4
    assert float(score["boundary_mean"] or 0) > float(score["control_mean"] or 0)
    assert seam_score(path, source_tile=0)["boundaries"] == 0
