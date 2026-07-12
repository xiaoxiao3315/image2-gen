from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path


def _safe_child(root: Path, filename: str | None) -> Path | None:
    if (
        not filename
        or filename in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", filename)
    ):
        return None
    # Do not resolve the final path: if it was replaced with a symlink, unlink
    # the symlink itself rather than the file it points to.
    return root.resolve() / filename


def cleanup(data_root: Path, older_than_days: float, *, now: float | None = None) -> int:
    if older_than_days <= 0:
        raise ValueError("older_than_days must be positive")
    root = data_root.resolve()
    database = root / "tasks.db"
    if not database.is_file():
        return 0
    cutoff = (time.time() if now is None else now) - older_than_days * 86_400
    images = root / "images"
    sources = root / "sources"
    removed = 0
    with sqlite3.connect(database, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        source_column = (
            "source_filename" if "source_filename" in columns else "NULL AS source_filename"
        )
        rows = connection.execute(
            f"SELECT task_id,image_filename,{source_column} FROM tasks "
            "WHERE status IN ('done','failed') "
            "AND COALESCE(completed_at,created_at) < ?",
            (cutoff,),
        ).fetchall()
        for row in rows:
            for path in (
                _safe_child(images, row["image_filename"]),
                _safe_child(sources, row["source_filename"]),
            ):
                if path is not None and (path.is_file() or path.is_symlink()):
                    path.unlink(missing_ok=True)
            connection.execute("DELETE FROM tasks WHERE task_id=?", (row["task_id"],))
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete expired image tasks and files")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--older-than-days", type=float, default=7.0)
    args = parser.parse_args()
    removed = cleanup(args.data_root, args.older_than_days)
    print(f"expired_tasks_removed={removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
