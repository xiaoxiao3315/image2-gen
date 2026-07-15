from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ERROR_CATEGORIES = {
    "connect_timeout",
    "read_timeout",
    "connection_error",
    "remote_disconnect",
    "http_503",
    "http_429",
    "http_4xx",
    "http_5xx",
    "invalid_response",
    "image_download_failure",
    "unknown",
}
_TERMINAL_STATUSES = {"done", "failed"}
_GENERATION_MODES = {
    "single",
    "single_fallback",
    "native_n",
    "native_n_partial",
}
_UPSCALE_QUEUE_REASONS = {"lease_expired", "worker_release"}
_UPSCALE_OUTCOMES = {"success", "retry", "failed", "lease_expired"}


def hash_label(value: str | None) -> str | None:
    """Return an irreversible label without persisting the source value."""
    normalized = (value or "").strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_error(
    error: BaseException | None = None,
    *,
    http_status: int | None = None,
    phase: str | None = None,
) -> str:
    """Classify failures without retaining exception text or response data."""
    if phase == "image_download":
        return "image_download_failure"
    if phase in {"json", "candidate", "image_decode", "validation"}:
        return "invalid_response"
    if http_status == 503:
        return "http_503"
    if http_status == 429:
        return "http_429"
    if http_status is not None and 400 <= http_status < 500:
        return "http_4xx"
    if http_status is not None and http_status >= 500:
        return "http_5xx"
    if error is None:
        return "unknown"

    def classify_exception(candidate: BaseException) -> str:
        name = type(candidate).__name__.lower()
        if "connecttimeout" in name:
            return "connect_timeout"
        if "readtimeout" in name or name == "timeout":
            return "read_timeout"
        if any(
            part in name
            for part in ("chunked", "remotedisconnected", "protocolerror")
        ):
            return "remote_disconnect"
        if "connection" in name:
            return "connection_error"
        return "unknown"

    category = classify_exception(error)
    if category != "unknown":
        return category

    current = error
    seen = {id(error)}
    for _ in range(2):
        cause = current.__cause__
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        category = classify_exception(cause)
        if category != "unknown":
            return category
        current = cause
    return "unknown"


def safe_error_summary(
    error: BaseException | None = None,
    *,
    category: str | None = None,
    http_status: int | None = None,
) -> str:
    """Build a bounded summary from type/category/status only, never exception text."""
    parts = [category or normalize_error(error, http_status=http_status)]
    if http_status is not None:
        parts.append(f"http_{int(http_status)}")
    if error is not None:
        type_name = re.sub(r"[^A-Za-z0-9_.-]", "", type(error).__name__)[:80]
        if type_name:
            parts.append(type_name)
    return ":".join(parts)[:160]


def _project_details(
    event_type: str,
    details: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return only the canonical detail fields defined for this lifecycle event."""
    if type(details) is not dict:
        return {}

    if event_type == "generation_queued":
        reason = details.get("reason")
        return (
            {"reason": "restart_recovery"}
            if type(reason) is str and reason == "restart_recovery"
            else {}
        )
    if event_type == "generation_completed":
        mode = details.get("mode")
        return (
            {"mode": mode}
            if type(mode) is str and mode in _GENERATION_MODES
            else {}
        )
    if event_type == "upscale_queued":
        reason = details.get("reason")
        return (
            {"reason": reason}
            if type(reason) is str and reason in _UPSCALE_QUEUE_REASONS
            else {}
        )
    if event_type == "upscale_finished":
        outcome = details.get("outcome")
        return (
            {"outcome": outcome}
            if type(outcome) is str and outcome in _UPSCALE_OUTCOMES
            else {}
        )
    if event_type == "terminal_failed":
        stage = details.get("stage")
        category = details.get("category")
        if (
            stage == "generation"
            and type(category) is str
            and category in ERROR_CATEGORIES
        ):
            return {"category": category, "stage": stage}
        if stage == "upscale" and category == "worker_failure":
            return {"category": category, "stage": stage}
    return {}


def _safe_details(
    event_type: str,
    details: Mapping[str, Any] | None,
) -> str | None:
    projected = _project_details(event_type, details)
    if not projected:
        return None
    return json.dumps(projected, sort_keys=True, separators=(",", ":"))


def _decode_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return round(ordered[rank - 1], 3)


def _percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value)) and value >= 0]
    return {
        "count": len(finite),
        "p50": _nearest_rank(finite, 50),
        "p95": _nearest_rank(finite, 95),
        "p99": _nearest_rank(finite, 99),
    }


def parse_utc_timestamp(value: str) -> float:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


class TelemetryStore:
    """Short-timeout, append-only SQLite telemetry that never owns business truth."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 0.005,
        initialize: bool = True,
    ):
        self.path = Path(path)
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.failure_count = 0
        self.last_error_type: str | None = None
        self._failure_lock = threading.Lock()
        self._initialize_lock = threading.Lock()
        self.enabled = self._try_initialize() if initialize else self.path.is_file()

    def _record_failure(self, error: BaseException) -> bool:
        with self._failure_lock:
            self.failure_count += 1
            self.last_error_type = type(error).__name__[:80]
        return False

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=self.timeout_seconds,
            )
        else:
            connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        return connection

    def _try_initialize(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_events (
                        event_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at REAL NOT NULL,
                        duration_seconds REAL,
                        attempt_no INTEGER,
                        worker_id_hash TEXT,
                        details_json TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS generation_attempts (
                        attempt_id TEXT PRIMARY KEY,
                        batch_id TEXT,
                        task_id TEXT,
                        task_ids_json TEXT NOT NULL,
                        attempt_no INTEGER NOT NULL,
                        request_kind TEXT NOT NULL,
                        requested_n INTEGER NOT NULL,
                        started_at REAL NOT NULL,
                        finished_at REAL,
                        duration_seconds REAL,
                        http_status INTEGER,
                        outcome TEXT,
                        error_category TEXT,
                        will_retry INTEGER,
                        backoff_seconds REAL,
                        error_summary TEXT,
                        provider_request_id TEXT,
                        route_label_hash TEXT
                    )
                    """
                )
                event_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(task_events)")
                }
                attempt_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(generation_attempts)")
                }
                required_event_columns = {
                    "event_id": "TEXT",
                    "task_id": "TEXT",
                    "event_type": "TEXT",
                    "occurred_at": "REAL",
                    "duration_seconds": "REAL",
                    "attempt_no": "INTEGER",
                    "worker_id_hash": "TEXT",
                    "details_json": "TEXT",
                }
                required_attempt_columns = {
                    "attempt_id": "TEXT",
                    "batch_id": "TEXT",
                    "task_id": "TEXT",
                    "task_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                    "attempt_no": "INTEGER NOT NULL DEFAULT 1",
                    "request_kind": "TEXT NOT NULL DEFAULT 'legacy'",
                    "requested_n": "INTEGER NOT NULL DEFAULT 1",
                    "started_at": "REAL NOT NULL DEFAULT 0",
                    "finished_at": "REAL",
                    "duration_seconds": "REAL",
                    "http_status": "INTEGER",
                    "outcome": "TEXT",
                    "error_category": "TEXT",
                    "will_retry": "INTEGER",
                    "backoff_seconds": "REAL",
                    "error_summary": "TEXT",
                    "provider_request_id": "TEXT",
                    "route_label_hash": "TEXT",
                }
                for column, definition in required_event_columns.items():
                    if column not in event_columns:
                        connection.execute(
                            f"ALTER TABLE task_events ADD COLUMN {column} {definition}"
                        )
                for column, definition in required_attempt_columns.items():
                    if column not in attempt_columns:
                        connection.execute(
                            f"ALTER TABLE generation_attempts ADD COLUMN {column} {definition}"
                        )
                connection.executescript(
                    """
                    CREATE INDEX IF NOT EXISTS idx_task_events_task_time
                        ON task_events(task_id, occurred_at, event_id);
                    CREATE INDEX IF NOT EXISTS idx_task_events_time_type
                        ON task_events(occurred_at, event_type);
                    CREATE INDEX IF NOT EXISTS idx_task_events_type_time
                        ON task_events(event_type, occurred_at);
                    CREATE INDEX IF NOT EXISTS idx_generation_attempts_started
                        ON generation_attempts(started_at, attempt_id);
                    CREATE INDEX IF NOT EXISTS idx_generation_attempts_task
                        ON generation_attempts(task_id, started_at, attempt_id);
                    CREATE INDEX IF NOT EXISTS idx_generation_attempts_batch
                        ON generation_attempts(batch_id, started_at, attempt_id);
                    CREATE INDEX IF NOT EXISTS idx_generation_attempts_failure
                        ON generation_attempts(error_category, started_at);
                    """
                )
            return True
        except Exception as exc:
            return self._record_failure(exc)

    def _ensure_enabled(self) -> bool:
        if self.enabled:
            return True
        with self._initialize_lock:
            if not self.enabled:
                self.enabled = self._try_initialize()
        return self.enabled

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "failure_count": self.failure_count,
            "last_error_type": self.last_error_type,
        }

    def seconds_since_latest_event(
        self, task_id: str, event_type: str, *, occurred_at: float
    ) -> float | None:
        try:
            if not self._ensure_enabled():
                return None
            with self._connect(readonly=True) as connection:
                row = connection.execute(
                    "SELECT occurred_at FROM task_events WHERE task_id=? AND event_type=? "
                    "ORDER BY occurred_at DESC,event_id DESC LIMIT 1",
                    (task_id, event_type),
                ).fetchone()
            if row is None:
                return None
            return max(0.0, float(occurred_at) - float(row["occurred_at"]))
        except Exception as exc:
            self._record_failure(exc)
            return None

    def try_event(
        self,
        task_id: str,
        event_type: str,
        *,
        occurred_at: float | None = None,
        duration_seconds: float | None = None,
        attempt_no: int | None = None,
        worker_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        try:
            if not self._ensure_enabled():
                return False
            duration = None if duration_seconds is None else max(0.0, float(duration_seconds))
            safe_event_type = re.sub(r"[^A-Za-z0-9_.-]", "", event_type)[:80]
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO task_events(event_id,task_id,event_type,occurred_at,"
                    "duration_seconds,attempt_no,worker_id_hash,details_json) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        str(task_id),
                        safe_event_type,
                        time.time() if occurred_at is None else float(occurred_at),
                        duration,
                        attempt_no,
                        hash_label(worker_id),
                        _safe_details(event_type, details),
                    ),
                )
            return True
        except Exception as exc:
            return self._record_failure(exc)

    def next_attempt_no(self, task_ids: Sequence[str]) -> int:
        """Return a shared next ordinal for tasks participating in one physical request."""
        try:
            if not self.enabled or not task_ids:
                return 1
            wanted = {str(task_id) for task_id in task_ids}
            maximum = 0
            with self._connect(readonly=True) as connection:
                rows = connection.execute(
                    "SELECT task_id,task_ids_json,attempt_no FROM generation_attempts "
                    "ORDER BY started_at,attempt_id"
                ).fetchall()
            for row in rows:
                attributed = set(_decode_json(row["task_ids_json"], []))
                if row["task_id"]:
                    attributed.add(str(row["task_id"]))
                if attributed & wanted:
                    maximum = max(maximum, int(row["attempt_no"]))
            return maximum + 1
        except Exception as exc:
            self._record_failure(exc)
            return 1

    def try_start_attempt(
        self,
        *,
        task_ids: Sequence[str],
        batch_id: str | None,
        attempt_no: int,
        request_kind: str,
        requested_n: int,
        started_at: float,
        route_label: str | None = None,
    ) -> str | None:
        try:
            if not self._ensure_enabled():
                return None
            normalized_ids = sorted({str(task_id) for task_id in task_ids})
            if not normalized_ids:
                return None
            attempt_id = uuid.uuid4().hex
            single_task = normalized_ids[0] if len(normalized_ids) == 1 else None
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO generation_attempts(attempt_id,batch_id,task_id,task_ids_json,"
                    "attempt_no,request_kind,requested_n,started_at,route_label_hash) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        attempt_id,
                        batch_id,
                        single_task,
                        json.dumps(normalized_ids, separators=(",", ":")),
                        int(attempt_no),
                        re.sub(r"[^A-Za-z0-9_.-]", "", request_kind)[:40],
                        int(requested_n),
                        float(started_at),
                        hash_label(route_label),
                    ),
                )
            return attempt_id
        except Exception as exc:
            self._record_failure(exc)
            return None

    def try_finish_attempt(
        self,
        attempt_id: str | None,
        *,
        finished_at: float,
        duration_seconds: float,
        http_status: int | None,
        outcome: str,
        error_category: str | None,
        will_retry: bool,
        backoff_seconds: float | None,
        error_summary: str | None,
        provider_request_id: str | None,
    ) -> bool:
        if not attempt_id:
            return False
        try:
            if not self._ensure_enabled():
                return False
            category = error_category if error_category in ERROR_CATEGORIES else None
            safe_request_id = hash_label(provider_request_id)
            safe_summary = None
            if error_summary:
                safe_summary = re.sub(
                    r"[^A-Za-z0-9_.:-]", "_", str(error_summary)
                )[:160]
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE generation_attempts SET finished_at=?,duration_seconds=?,http_status=?,"
                    "outcome=?,error_category=?,will_retry=?,backoff_seconds=?,error_summary=?,"
                    "provider_request_id=? WHERE attempt_id=? AND finished_at IS NULL",
                    (
                        float(finished_at),
                        max(0.0, float(duration_seconds)),
                        http_status,
                        re.sub(r"[^A-Za-z0-9_.-]", "", outcome)[:40],
                        category,
                        1 if will_retry else 0,
                        None if backoff_seconds is None else max(0.0, float(backoff_seconds)),
                        safe_summary,
                        safe_request_id,
                        attempt_id,
                    ),
                )
            return cursor.rowcount == 1
        except Exception as exc:
            return self._record_failure(exc)

    def try_reconcile_open_attempts(
        self,
        *,
        occurred_at: float | None = None,
        restart_task_ids: Sequence[str] = (),
    ) -> set[str]:
        """Best-effort closure for attempts whose business task already advanced."""
        reconciled_tasks: set[str] = set()
        try:
            if not self._ensure_enabled():
                return reconciled_tasks
            now = time.time() if occurred_at is None else float(occurred_at)
            restart_tasks = {str(task_id) for task_id in restart_task_ids}
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT attempt_id,task_id,task_ids_json,started_at FROM generation_attempts "
                    "WHERE finished_at IS NULL"
                ).fetchall()
                for row in rows:
                    attributed = set(_decode_json(row["task_ids_json"], []))
                    if row["task_id"]:
                        attributed.add(str(row["task_id"]))
                    if not attributed:
                        continue
                    placeholders = ",".join("?" for _ in attributed)
                    task_rows = connection.execute(
                        f"SELECT task_id,status FROM tasks WHERE task_id IN ({placeholders})",
                        sorted(attributed),
                    ).fetchall()
                    if not task_rows or any(
                        str(task["status"]) == "processing" for task in task_rows
                    ):
                        continue
                    try:
                        started_at = float(row["started_at"])
                    except (TypeError, ValueError):
                        started_at = now
                    task_ids = {str(task["task_id"]) for task in task_rows}
                    restart_interrupted = bool(task_ids & restart_tasks)
                    connection.execute(
                        "UPDATE generation_attempts SET finished_at=?,duration_seconds=?,"
                        "outcome=?,error_category='unknown',will_retry=?,error_summary=? "
                        "WHERE attempt_id=? AND finished_at IS NULL",
                        (
                            now,
                            max(0.0, now - started_at),
                            "interrupted" if restart_interrupted else "telemetry_interrupted",
                            1 if restart_interrupted else 0,
                            (
                                "process_interrupted"
                                if restart_interrupted
                                else "finish_fact_unavailable"
                            ),
                            row["attempt_id"],
                        ),
                    )
                    reconciled_tasks.update(task_ids)
            return reconciled_tasks
        except Exception as exc:
            self._record_failure(exc)
            return reconciled_tasks

    def try_close_interrupted_attempts(
        self, recovered_task_ids: Sequence[str], *, occurred_at: float | None = None
    ) -> bool:
        try:
            if not self.enabled or not recovered_task_ids:
                return True
            recovered = {str(value) for value in recovered_task_ids}
            now = time.time() if occurred_at is None else float(occurred_at)
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT attempt_id,task_id,task_ids_json,started_at "
                    "FROM generation_attempts WHERE finished_at IS NULL"
                ).fetchall()
                matching = []
                for row in rows:
                    attributed = set(_decode_json(row["task_ids_json"], []))
                    if row["task_id"]:
                        attributed.add(str(row["task_id"]))
                    if attributed & recovered:
                        matching.append((row["attempt_id"], float(row["started_at"])))
                connection.executemany(
                    "UPDATE generation_attempts SET finished_at=?,duration_seconds=?,"
                    "outcome='interrupted',error_category='unknown',will_retry=1,"
                    "error_summary='process_interrupted' WHERE attempt_id=? AND finished_at IS NULL",
                    [(now, max(0.0, now - started), attempt_id) for attempt_id, started in matching],
                )
            return True
        except Exception as exc:
            return self._record_failure(exc)

    def task_timeline(self, task_id: str) -> dict[str, Any]:
        with self._connect(readonly=True) as connection:
            task = connection.execute(
                "SELECT task_id,batch_id,batch_index,batch_size,size,status,created_at,started_at,"
                "completed_at,upscale_attempts FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise KeyError("task not found")
            try:
                events = connection.execute(
                    "SELECT event_id,event_type,occurred_at,duration_seconds,attempt_no,"
                    "worker_id_hash,details_json FROM task_events WHERE task_id=? "
                    "ORDER BY occurred_at,event_id",
                    (task_id,),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table: task_events" not in str(exc):
                    raise
                events = []
            try:
                raw_attempts = connection.execute(
                    "SELECT * FROM generation_attempts WHERE task_id=? OR batch_id=? "
                    "ORDER BY started_at,attempt_id",
                    (task_id, task["batch_id"]),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "generation_attempts" not in str(exc):
                    raise
                raw_attempts = []

        attempts = []
        for row in raw_attempts:
            attributed = set(_decode_json(row["task_ids_json"], []))
            if row["task_id"]:
                attributed.add(str(row["task_id"]))
            if task_id not in attributed:
                continue
            item = dict(row)
            item.pop("task_ids_json", None)
            attempts.append(item)
        event_items = []
        for row in events:
            item = dict(row)
            raw_details = _decode_json(item.pop("details_json"), {})
            item["details"] = _project_details(item["event_type"], raw_details)
            event_items.append(item)

        combined = [
            {"kind": "event", "at": item["occurred_at"], **item}
            for item in event_items
        ] + [
            {"kind": "generation_attempt", "at": item["started_at"], **item}
            for item in attempts
        ]
        combined.sort(key=lambda item: (float(item["at"]), str(item.get("event_id") or item.get("attempt_id"))))
        event_types = [item["event_type"] for item in event_items]
        diagnostics: list[str] = []
        if "accepted" not in event_types:
            diagnostics.append("missing_accepted_event")
        if task["status"] in _TERMINAL_STATUSES and not any(
            event in event_types for event in ("delivery_completed", "terminal_failed")
        ):
            diagnostics.append("missing_terminal_event")
        if any(item["finished_at"] is None for item in attempts):
            diagnostics.append("open_generation_attempt")
        return {
            "task": dict(task),
            "events": event_items,
            "generation_attempts": attempts,
            "timeline": combined,
            "integrity": {"ok": not diagnostics, "diagnostics": diagnostics},
        }

    def window_stats(self, since: float, until: float) -> dict[str, Any]:
        if until <= since:
            raise ValueError("until must be after since")
        with self._connect(readonly=True) as connection:
            arrivals = connection.execute(
                "SELECT task_id,status,created_at,started_at,completed_at FROM tasks "
                "WHERE created_at>=? AND created_at<? ORDER BY created_at,task_id",
                (since, until),
            ).fetchall()
            completed = connection.execute(
                "SELECT task_id,status,created_at,started_at,completed_at FROM tasks "
                "WHERE completed_at>=? AND completed_at<? AND status IN ('done','failed') "
                "ORDER BY completed_at,task_id",
                (since, until),
            ).fetchall()
            queue_task_rows = connection.execute(
                "SELECT task_id,created_at FROM tasks WHERE created_at<?",
                (until,),
            ).fetchall()
            try:
                queue_state_events = connection.execute(
                    "SELECT task_id,event_type,occurred_at,event_id FROM task_events "
                    "WHERE event_type IN ('generation_queued','generation_started') "
                    "AND occurred_at<? ORDER BY task_id,occurred_at,event_id",
                    (until,),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table: task_events" not in str(exc):
                    raise
                queue_state_events = []
            try:
                events = connection.execute(
                    "SELECT task_id,event_type,occurred_at,duration_seconds,details_json "
                    "FROM task_events WHERE occurred_at>=? AND occurred_at<? "
                    "ORDER BY occurred_at,event_id",
                    (since, until),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table: task_events" not in str(exc):
                    raise
                events = []
            try:
                attempts = connection.execute(
                    "SELECT task_id,task_ids_json,duration_seconds,backoff_seconds,error_category "
                    "FROM generation_attempts WHERE finished_at>=? AND finished_at<? "
                    "ORDER BY finished_at,attempt_id",
                    (since, until),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table: generation_attempts" not in str(exc):
                    raise
                attempts = []

        delivered_rows = [row for row in completed if row["status"] == "done"]
        failed_rows = [row for row in completed if row["status"] == "failed"]
        delivered = len(delivered_rows)
        failed = len(failed_rows)
        terminal = delivered + failed
        latest_queue_state: dict[str, sqlite3.Row] = {}
        for row in queue_state_events:
            latest_queue_state[str(row["task_id"])] = row
        created_by_task = {
            str(row["task_id"]): float(row["created_at"])
            for row in queue_task_rows
        }
        generation_queue = [
            float(row["occurred_at"] - created_by_task[str(row["task_id"])])
            for row in events
            if row["event_type"] == "generation_started"
            and str(row["task_id"]) in created_by_task
        ]
        end_to_end = [
            float(row["completed_at"] - row["created_at"])
            for row in completed
        ]
        oldest_queued_age = None
        queued_created = [
            created_by_task[task_id]
            for task_id, row in latest_queue_state.items()
            if row["event_type"] == "generation_queued"
            and task_id in created_by_task
        ]
        if queued_created:
            oldest_queued_age = round(max(0.0, until - min(queued_created)), 3)

        event_durations: dict[str, list[float]] = {
            "upscale_queue": [],
            "upscale_execution": [],
            "delivery": [],
        }
        for row in events:
            if row["duration_seconds"] is None:
                continue
            event_type = str(row["event_type"])
            if event_type == "upscale_started":
                event_durations["upscale_queue"].append(float(row["duration_seconds"]))
            elif event_type == "upscale_finished":
                event_durations["upscale_execution"].append(float(row["duration_seconds"]))
            elif event_type == "delivery_completed":
                event_durations["delivery"].append(float(row["duration_seconds"]))

        attempt_durations: list[float] = []
        backoffs: list[float] = []
        failure_categories: dict[str, int] = {}
        for row in attempts:
            attributed = set(_decode_json(row["task_ids_json"], []))
            if row["task_id"]:
                attributed.add(str(row["task_id"]))
            if row["duration_seconds"] is not None:
                attempt_durations.append(float(row["duration_seconds"]))
            if row["backoff_seconds"] is not None:
                backoffs.append(float(row["backoff_seconds"]))
            if row["error_category"]:
                category = str(row["error_category"])
                failure_categories[category] = failure_categories.get(category, 0) + 1

        request_count = len(arrivals)
        window_seconds = until - since
        return {
            "window": {"since": since, "until": until, "seconds": round(window_seconds, 3)},
            "requests": request_count,
            "delivered": delivered,
            "terminal_failures": failed,
            "success_rate": None if terminal == 0 else round(delivered / terminal, 6),
            "failure_rate": None if terminal == 0 else round(failed / terminal, 6),
            "throughput_delivered_per_hour": round(delivered / window_seconds * 3600, 6),
            "oldest_queued_age_seconds": oldest_queued_age,
            "failure_categories": dict(sorted(failure_categories.items())),
            "latency_seconds": {
                "generation_queue": _percentiles(generation_queue),
                "generation_attempt": _percentiles(attempt_durations),
                "retry_backoff": _percentiles(backoffs),
                "upscale_queue": _percentiles(event_durations["upscale_queue"]),
                "upscale_execution": _percentiles(event_durations["upscale_execution"]),
                "delivery": _percentiles(event_durations["delivery"]),
                "end_to_end": _percentiles(end_to_end),
            },
        }
