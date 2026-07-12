from pathlib import Path


def test_frontend_has_bounded_batch_controls_and_multi_result_rendering() -> None:
    html = (Path(__file__).parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="image-count"' in html
    assert 'id="request-concurrency"' in html
    assert html.count('min="1" max="5"') >= 2
    assert 'id="result-grid"' in html
    assert "JSON.stringify({ prompt, size, count, concurrency })" in html
    assert "task_ids" in html and "result_urls" in html
    assert "下载第 ${index + 1} 张 PNG" in html
