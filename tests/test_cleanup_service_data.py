from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.cleanup_service_data import _safe_child, cleanup


def test_cleanup_removes_only_expired_terminal_tasks(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "sources").mkdir()
    database = tmp_path / "tasks.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE tasks (task_id TEXT PRIMARY KEY,batch_id TEXT,status TEXT,created_at REAL,"
            "completed_at REAL,image_filename TEXT,source_filename TEXT)"
        )
        connection.executemany(
            "INSERT INTO tasks VALUES(?,?,?,?,?,?,?)",
            [
                ("old", "old-batch", "done", 1, 10, "old.png", "old-source.png"),
                ("new", "new-batch", "done", 900_000, 900_000, "new.png", None),
                ("active", "active-batch", "awaiting_upscale", 1, None, None, "active.png"),
                ("mixed-old", "mixed-batch", "done", 1, 10, "mixed-old.png", None),
                ("mixed-active", "mixed-batch", "processing", 1, None, None, None),
                ("split-old", "split-batch", "done", 1, 10, "split-old.png", None),
                ("split-new", "split-batch", "done", 900_000, 900_000, "split-new.png", None),
            ],
        )
        connection.execute(
            "CREATE TABLE api_idempotency (batch_id TEXT PRIMARY KEY)"
        )
        connection.executemany(
            "INSERT INTO api_idempotency(batch_id) VALUES(?)",
            [
                ("old-batch",),
                ("new-batch",),
                ("active-batch",),
                ("mixed-batch",),
                ("split-batch",),
            ],
        )
    for relative in (
        "images/old.png",
        "sources/old-source.png",
        "images/new.png",
        "sources/active.png",
        "images/mixed-old.png",
        "images/split-old.png",
        "images/split-new.png",
    ):
        path = tmp_path / relative
        path.write_bytes(b"x")

    assert cleanup(tmp_path, 7, now=1_000_000) == 1

    with sqlite3.connect(database) as connection:
        ids = {row[0] for row in connection.execute("SELECT task_id FROM tasks")}
        idempotent_batches = {
            row[0] for row in connection.execute("SELECT batch_id FROM api_idempotency")
        }
    assert ids == {
        "new",
        "active",
        "mixed-old",
        "mixed-active",
        "split-old",
        "split-new",
    }
    assert idempotent_batches == {
        "new-batch",
        "active-batch",
        "mixed-batch",
        "split-batch",
    }
    assert not (tmp_path / "images/old.png").exists()
    assert not (tmp_path / "sources/old-source.png").exists()
    assert (tmp_path / "images/new.png").is_file()
    assert (tmp_path / "sources/active.png").is_file()
    assert (tmp_path / "images/mixed-old.png").is_file()
    assert (tmp_path / "images/split-old.png").is_file()
    assert (tmp_path / "images/split-new.png").is_file()


def test_safe_child_rejects_traversal_and_does_not_resolve_final_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"keep")
    assert _safe_child(root, "../outside.png") is None
    assert _safe_child(root, "subdir/file.png") is None
    assert _safe_child(root, r"subdir\file.png") is None

    link = root / "safe-name.png"
    try:
        link.symlink_to(outside)
    except OSError:
        return  # Symlink creation may require privileges on older Windows hosts.
    candidate = _safe_child(root, link.name)
    assert candidate == link
    assert candidate is not None
    candidate.unlink()
    assert outside.is_file()
