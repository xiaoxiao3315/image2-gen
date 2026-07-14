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
        batch_column = "batch_id" if "batch_id" in columns else "NULL AS batch_id"
        if "batch_id" in columns:
            rows = connection.execute(
                f"SELECT task_id,image_filename,{source_column},{batch_column} FROM tasks "
                "WHERE (batch_id IN ("
                "SELECT batch_id FROM tasks WHERE batch_id IS NOT NULL GROUP BY batch_id "
                "HAVING SUM(CASE WHEN status NOT IN ('done','failed') THEN 1 ELSE 0 END)=0 "
                "AND MAX(COALESCE(completed_at,created_at)) < ?"
                ") OR (batch_id IS NULL AND status IN ('done','failed') "
                "AND COALESCE(completed_at,created_at) < ?))",
                (cutoff, cutoff),
            ).fetchall()
        else:
            rows = connection.execute(
                f"SELECT task_id,image_filename,{source_column},{batch_column} FROM tasks "
                "WHERE status IN ('done','failed') "
                "AND COALESCE(completed_at,created_at) < ?",
                (cutoff,),
            ).fetchall()
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        event_columns = (
            {
                str(item[1])
                for item in connection.execute("PRAGMA table_info(task_events)").fetchall()
            }
            if "task_events" in tables
            else set()
        )
        attempt_columns = (
            {
                str(item[1])
                for item in connection.execute(
                    "PRAGMA table_info(generation_attempts)"
                ).fetchall()
            }
            if "generation_attempts" in tables
            else set()
        )
        # Clear each persisted file reference immediately after a successful unlink.
        # If a later artifact fails to unlink, the retained task stays discoverable
        # without pointing at an already-deleted file.
        for row in rows:
            task_id = str(row["task_id"])
            artifact_columns = (
                ("source_filename", _safe_child(sources, row["source_filename"])),
                ("image_filename", _safe_child(images, row["image_filename"])),
            )
            for column, path in artifact_columns:
                if path is not None and (path.is_file() or path.is_symlink()):
                    path.unlink(missing_ok=True)
                    if column in columns:
                        connection.execute(
                            f"UPDATE tasks SET {column}=NULL WHERE task_id=?",
                            (task_id,),
                        )
                        connection.commit()
            if "task_id" in event_columns:
                connection.execute("DELETE FROM task_events WHERE task_id=?", (task_id,))
            if "task_id" in attempt_columns:
                connection.execute(
                    "DELETE FROM generation_attempts WHERE task_id=?", (task_id,)
                )
            connection.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
            connection.commit()
            removed += 1
        if "batch_id" in attempt_columns and "batch_id" in columns:
            connection.execute(
                "DELETE FROM generation_attempts WHERE batch_id IS NOT NULL AND NOT EXISTS ("
                "SELECT 1 FROM tasks WHERE tasks.batch_id=generation_attempts.batch_id)"
            )
        if "api_idempotency" in tables and "batch_id" in columns:
            connection.execute(
                "DELETE FROM api_idempotency "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM tasks WHERE tasks.batch_id=api_idempotency.batch_id"
                ")"
            )
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
