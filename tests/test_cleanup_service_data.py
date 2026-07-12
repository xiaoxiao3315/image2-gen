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
            "CREATE TABLE tasks (task_id TEXT PRIMARY KEY,status TEXT,created_at REAL,"
            "completed_at REAL,image_filename TEXT,source_filename TEXT)"
        )
        connection.executemany(
            "INSERT INTO tasks VALUES(?,?,?,?,?,?)",
            [
                ("old", "done", 1, 10, "old.png", "old-source.png"),
                ("new", "done", 900_000, 900_000, "new.png", None),
                ("active", "awaiting_upscale", 1, None, None, "active.png"),
            ],
        )
    for relative in ("images/old.png", "sources/old-source.png", "images/new.png", "sources/active.png"):
        path = tmp_path / relative
        path.write_bytes(b"x")

    assert cleanup(tmp_path, 7, now=1_000_000) == 1

    with sqlite3.connect(database) as connection:
        ids = {row[0] for row in connection.execute("SELECT task_id FROM tasks")}
    assert ids == {"new", "active"}
    assert not (tmp_path / "images/old.png").exists()
    assert not (tmp_path / "sources/old-source.png").exists()
    assert (tmp_path / "images/new.png").is_file()
    assert (tmp_path / "sources/active.png").is_file()


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
