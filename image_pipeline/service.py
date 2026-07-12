from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import queue
import re
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field, field_validator

from .config import PROJECT_ROOT, Settings
from .generator import (
    GptImageGenerator,
    NativeBatchIncomplete,
    NativeBatchUnsupported,
)
from .image_io import inspect_image


SERVICE_ROOT = Path(os.getenv("IMAGE_SERVICE_DATA", PROJECT_ROOT / "service-data")).resolve()
PUBLIC_IMAGE_DIR = (SERVICE_ROOT / "images").resolve()
SOURCE_IMAGE_DIR = (SERVICE_ROOT / "sources").resolve()
DATABASE_PATH = (SERVICE_ROOT / "tasks.db").resolve()
WEB_INDEX_PATH = (PROJECT_ROOT / "web" / "index.html").resolve()
PUBLIC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PIXELS = {"2k": (2048, 2048), "4k": (3840, 2160)}
DEFAULT_LEASE_SECONDS = 15 * 60
DEFAULT_UPSCALE_MAX_ATTEMPTS = 3
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_BATCH_SIZE = 5
MAX_GENERATION_CONCURRENCY = 8
DEFAULT_GENERATION_MIN_WORKERS = 1
DEFAULT_GENERATION_MAX_WORKERS = 3


class GenerateTaskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32000)
    size: str = Field(default="4k", validation_alias=AliasChoices("size", "target"))
    count: int = Field(default=1, ge=1, le=MAX_BATCH_SIZE)
    concurrency: int = Field(default=3, ge=1, le=MAX_BATCH_SIZE)

    @field_validator("size")
    @classmethod
    def normalize_size(cls, value: str) -> str:
        normalized = value.strip().lower().replace("×", "x")
        aliases = {
            "2k": "2k",
            "2048x2048": "2k",
            "4k": "4k",
            "3840x2160": "4k",
        }
        if normalized not in aliases:
            raise ValueError("size must be 2k/2048x2048 or 4k/3840x2160")
        return aliases[normalized]


class OpenAIImageGenerationRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = Field(default=1, ge=1, le=MAX_BATCH_SIZE)
    size: str = Field(default="2048x2048")
    response_format: str = "url"
    user: str = Field(min_length=8, max_length=128)

    @field_validator("size")
    @classmethod
    def normalize_size(cls, value: str) -> str:
        return GenerateTaskRequest.normalize_size(value)

    @field_validator("response_format")
    @classmethod
    def require_url_response(cls, value: str) -> str:
        if value != "url":
            raise ValueError("response_format must be url")
        return value


class IdempotencyConflict(RuntimeError):
    pass


def _utc_iso(timestamp: float | None = None) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp or time.time(), tz=timezone.utc).isoformat()


def _scrub_error(value: object) -> str:
    text = str(value)
    for secret_value in (
        os.getenv("IMAGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
        os.getenv("IMAGE_API_BASE_URL", "").strip().rstrip("/"),
    ):
        if secret_value:
            text = text.replace(secret_value, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED]", text)[:500]


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _bounded_int_env(name: str, default: int, maximum: int) -> int:
    value = _positive_int_env(name, default)
    if value > maximum:
        raise RuntimeError(f"{name} must be an integer from 1 to {maximum}")
    return value


class WeightedLimiter:
    """A weighted semaphore with observable usage for tests and health checks."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._used = 0
        self._peak = 0
        self._condition = threading.Condition()

    @property
    def current(self) -> int:
        with self._condition:
            return self._used

    @property
    def peak(self) -> int:
        with self._condition:
            return self._peak

    @contextmanager
    def slot(self, weight: int) -> Iterator[None]:
        if not 1 <= weight <= self.capacity:
            raise ValueError(f"weight must be from 1 to {self.capacity}")
        with self._condition:
            while self._used + weight > self.capacity:
                self._condition.wait()
            self._used += weight
            self._peak = max(self._peak, self._used)
        try:
            yield
        finally:
            with self._condition:
                self._used -= weight
                self._condition.notify_all()


GLOBAL_GENERATION_LIMITER = WeightedLimiter(MAX_GENERATION_CONCURRENCY)


def _cost_metrics(
    file_bytes: int, gpu_seconds: float, usage: dict[str, Any] | None
) -> dict[str, Any]:
    """Legacy reporting helper retained for callers of the former single-node service."""
    gigabytes = file_bytes / 1_000_000_000
    gpu_low = gpu_seconds / 3600 * 3.0
    gpu_high = gpu_seconds / 3600 * 5.0
    return {
        "api_cost_cny": None,
        "api_cost_source": "unavailable",
        "provider_reported_cost": None,
        "provider_usage": usage,
        "gpu_cloud_cny_range": [round(gpu_low, 6), round(gpu_high, 6)],
        "storage_cny_first_month": round(gigabytes * 0.12, 6),
        "download_traffic_cny_once": round(gigabytes * 0.50, 6),
    }


class TaskStore:
    """SQLite task store shared by the public API and outbound GPU workers."""

    _MIGRATION_COLUMNS = {
        "source_filename": "TEXT",
        "source_width": "INTEGER",
        "source_height": "INTEGER",
        "source_file_bytes": "INTEGER",
        "source_sha256": "TEXT",
        "claim_token": "TEXT",
        "claimed_at": "REAL",
        "lease_expires_at": "REAL",
        "submitted_claim_token": "TEXT",
        "batch_id": "TEXT",
        "batch_index": "INTEGER",
        "batch_size": "INTEGER",
        "requested_concurrency": "INTEGER",
        "claimed_by": "TEXT",
        "lease_heartbeat_at": "REAL",
        "upscale_attempts": "INTEGER",
        "generation_idempotency_key": "TEXT",
    }

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    size TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    error TEXT,
                    image_filename TEXT,
                    local_path TEXT,
                    width INTEGER,
                    height INTEGER,
                    file_bytes INTEGER,
                    sha256 TEXT,
                    manifest_path TEXT,
                    metrics_json TEXT,
                    cost_json TEXT
                )
                """
            )
            existing = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            for column, definition in self._MIGRATION_COLUMNS.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")
            connection.execute(
                "UPDATE tasks SET batch_id=task_id WHERE batch_id IS NULL OR batch_id=''"
            )
            connection.execute(
                "UPDATE tasks SET batch_index=0 WHERE batch_index IS NULL"
            )
            connection.execute(
                "UPDATE tasks SET batch_size=1 WHERE batch_size IS NULL"
            )
            connection.execute(
                "UPDATE tasks SET requested_concurrency=1 WHERE requested_concurrency IS NULL"
            )
            connection.execute(
                "UPDATE tasks SET upscale_attempts=0 WHERE upscale_attempts IS NULL"
            )
            connection.execute(
                "UPDATE tasks SET generation_idempotency_key=lower(hex(randomblob(16))) "
                "WHERE generation_idempotency_key IS NULL OR generation_idempotency_key=''"
            )
            # Generation calls interrupted by a process restart are safe to enqueue again;
            # an active GPU claim is only recovered after its lease expires.
            connection.execute(
                "UPDATE tasks SET status='queued', started_at=NULL "
                "WHERE status IN ('running', 'processing')"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_created "
                "ON tasks(status, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_batch_index "
                "ON tasks(batch_id, batch_index)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_upscale_lease "
                "ON tasks(status, lease_expires_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS upscale_workers (
                    worker_id TEXT PRIMARY KEY,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    state TEXT NOT NULL,
                    current_task_id TEXT,
                    completed_count INTEGER NOT NULL DEFAULT 0,
                    released_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS api_idempotency (
                    tenant_scope_hash TEXT NOT NULL,
                    idempotency_key_hash TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(tenant_scope_hash, idempotency_key_hash)
                )
                """
            )

    def create(self, prompt: str, size: str) -> dict[str, Any]:
        return self.create_batch(prompt, size, count=1, requested_concurrency=1)[0]

    def create_batch(
        self, prompt: str, size: str, count: int, requested_concurrency: int
    ) -> list[dict[str, Any]]:
        if not 1 <= count <= MAX_BATCH_SIZE:
            raise ValueError(f"count must be from 1 to {MAX_BATCH_SIZE}")
        if not 1 <= requested_concurrency <= MAX_BATCH_SIZE:
            raise ValueError(
                f"requested_concurrency must be from 1 to {MAX_BATCH_SIZE}"
            )
        batch_id = uuid.uuid4().hex
        created_at = time.time()
        records = [
            {
                "task_id": uuid.uuid4().hex,
                "batch_id": batch_id,
                "batch_index": index,
                "batch_size": count,
                "requested_concurrency": requested_concurrency,
                "generation_idempotency_key": uuid.uuid4().hex,
                "status": "queued",
                "created_at": created_at,
            }
            for index in range(count)
        ]
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO tasks(task_id,prompt,size,status,created_at,batch_id,batch_index,"
                "batch_size,requested_concurrency,generation_idempotency_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        record["task_id"],
                        prompt,
                        size,
                        "queued",
                        created_at,
                        batch_id,
                        record["batch_index"],
                        count,
                        requested_concurrency,
                        record["generation_idempotency_key"],
                    )
                    for record in records
                ],
            )
        return records

    def create_batch_idempotent(
        self,
        prompt: str,
        size: str,
        count: int,
        requested_concurrency: int,
        tenant_scope: str,
        idempotency_key: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        if not 1 <= count <= MAX_BATCH_SIZE:
            raise ValueError(f"count must be from 1 to {MAX_BATCH_SIZE}")
        if not 1 <= requested_concurrency <= MAX_BATCH_SIZE:
            raise ValueError(f"requested_concurrency must be from 1 to {MAX_BATCH_SIZE}")
        tenant_hash = hashlib.sha256(tenant_scope.encode("utf-8")).hexdigest()
        key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        request_hash = hashlib.sha256(
            json.dumps(
                {
                    "prompt": prompt,
                    "size": size,
                    "count": count,
                    "concurrency": requested_concurrency,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_hash,batch_id FROM api_idempotency "
                "WHERE tenant_scope_hash=? AND idempotency_key_hash=?",
                (tenant_hash, key_hash),
            ).fetchone()
            if existing is not None:
                if not secrets.compare_digest(str(existing["request_hash"]), request_hash):
                    connection.rollback()
                    raise IdempotencyConflict(
                        "idempotency key was already used with a different request"
                    )
                rows = connection.execute(
                    "SELECT * FROM tasks WHERE batch_id=? ORDER BY batch_index",
                    (existing["batch_id"],),
                ).fetchall()
                if rows:
                    indexes = [int(row["batch_index"]) for row in rows]
                    if len(rows) != count or indexes != list(range(count)):
                        connection.rollback()
                        raise IdempotencyConflict(
                            "idempotent batch ledger is incomplete; refusing unsafe replay"
                        )
                    connection.commit()
                    return [dict(row) for row in rows], True
                # Retention cleanup may have deleted every task in this batch.
                # Remove the orphaned ledger row and recreate the logical request
                # atomically instead of returning reused=True with an empty batch.
                connection.execute(
                    "DELETE FROM api_idempotency "
                    "WHERE tenant_scope_hash=? AND idempotency_key_hash=?",
                    (tenant_hash, key_hash),
                )

            batch_id = uuid.uuid4().hex
            created_at = time.time()
            records = [
                {
                    "task_id": uuid.uuid4().hex,
                    "batch_id": batch_id,
                    "batch_index": index,
                    "batch_size": count,
                    "requested_concurrency": requested_concurrency,
                    "generation_idempotency_key": uuid.uuid4().hex,
                    "status": "queued",
                    "created_at": created_at,
                }
                for index in range(count)
            ]
            connection.executemany(
                "INSERT INTO tasks(task_id,prompt,size,status,created_at,batch_id,batch_index,"
                "batch_size,requested_concurrency,generation_idempotency_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        record["task_id"],
                        prompt,
                        size,
                        "queued",
                        created_at,
                        batch_id,
                        record["batch_index"],
                        count,
                        requested_concurrency,
                        record["generation_idempotency_key"],
                    )
                    for record in records
                ],
            )
            connection.execute(
                "INSERT INTO api_idempotency(tenant_scope_hash,idempotency_key_hash,"
                "request_hash,batch_id,created_at) VALUES(?,?,?,?,?)",
                (tenant_hash, key_hash, request_hash, batch_id, created_at),
            )
            connection.commit()
            return records, False
        finally:
            connection.close()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def get_batch(self, batch_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE batch_id=? ORDER BY batch_index", (batch_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def queued_batch_ids(self) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT batch_id,MIN(created_at) FROM tasks WHERE status='queued' "
                "GROUP BY batch_id ORDER BY MIN(created_at)"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def queued_ids(self) -> list[str]:
        """Compatibility helper for callers that inspect individual queued tasks."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT task_id FROM tasks WHERE status='queued' ORDER BY created_at,batch_index"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status").fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def operational_stats(self, now: float | None = None) -> dict[str, Any]:
        from datetime import datetime, timezone

        now = time.time() if now is None else now
        current = datetime.fromtimestamp(now, tz=timezone.utc)
        day_start = current.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        with self.connect() as connection:
            status_rows = connection.execute(
                "SELECT status,COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
            today = connection.execute(
                "SELECT "
                "SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done_count,"
                "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count,"
                "AVG(CASE WHEN status='done' THEN completed_at-created_at END) AS avg_seconds "
                "FROM tasks WHERE completed_at>=?",
                (day_start,),
            ).fetchone()
        counts = {str(row["status"]): int(row["count"]) for row in status_rows}
        done = int(today["done_count"] or 0)
        failed = int(today["failed_count"] or 0)
        terminal = done + failed
        return {
            "queue_length": counts.get("queued", 0),
            "generating": counts.get("processing", 0),
            "awaiting_upscale": counts.get("awaiting_upscale", 0),
            "upscaling": counts.get("upscaling", 0),
            "done_total": counts.get("done", 0),
            "failed_total": counts.get("failed", 0),
            "today_done": done,
            "today_failed": failed,
            "today_success_rate": None if terminal == 0 else round(done / terminal, 6),
            "today_average_end_to_end_seconds": (
                None if today["avg_seconds"] is None else round(float(today["avg_seconds"]), 3)
            ),
        }

    def update(self, task_id: str, **values: Any) -> None:
        if not values:
            return
        columns = ",".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE tasks SET {columns} WHERE task_id=?",
                [*values.values(), task_id],
            )

    def begin_generation(self, task_id: str, started_at: float) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status='processing',started_at=?,error=NULL "
                "WHERE task_id=? AND status='queued'",
                (started_at, task_id),
            )
        return cursor.rowcount == 1

    def begin_batch_generation(
        self, batch_id: str, started_at: float
    ) -> list[dict[str, Any]]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM tasks WHERE batch_id=? AND status='queued' "
                "ORDER BY batch_index",
                (batch_id,),
            ).fetchall()
            if not rows:
                connection.commit()
                return []
            task_ids = [str(row["task_id"]) for row in rows]
            placeholders = ",".join("?" for _ in task_ids)
            connection.execute(
                f"UPDATE tasks SET status='processing',started_at=?,error=NULL "
                f"WHERE task_id IN ({placeholders}) AND status='queued'",
                (started_at, *task_ids),
            )
            connection.commit()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _touch_worker(
        connection: sqlite3.Connection,
        worker_id: str,
        state: str,
        current_task_id: str | None,
        now: float,
    ) -> None:
        connection.execute(
            "INSERT INTO upscale_workers(worker_id,first_seen_at,last_seen_at,state,current_task_id) "
            "VALUES(?,?,?,?,?) ON CONFLICT(worker_id) DO UPDATE SET "
            "last_seen_at=excluded.last_seen_at,state=excluded.state,"
            "current_task_id=excluded.current_task_id",
            (worker_id, now, now, state, current_task_id),
        )

    def claim_upscale(
        self,
        lease_seconds: int,
        worker_id: str = "legacy-worker",
        now: float | None = None,
    ) -> dict[str, Any] | None:
        now = time.time() if now is None else now
        claim_token = secrets.token_urlsafe(32)
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE tasks SET status='awaiting_upscale',claim_token=NULL,"
                "claimed_at=NULL,lease_expires_at=NULL,claimed_by=NULL,"
                "lease_heartbeat_at=NULL "
                "WHERE status='upscaling' AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at<=?",
                (now,),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE status='awaiting_upscale' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                self._touch_worker(connection, worker_id, "idle", None, now)
                connection.commit()
                return None
            lease_expires_at = now + lease_seconds
            cursor = connection.execute(
                "UPDATE tasks SET status='upscaling',claim_token=?,claimed_at=?,"
                "lease_expires_at=?,claimed_by=?,lease_heartbeat_at=?,"
                "upscale_attempts=COALESCE(upscale_attempts,0)+1 "
                "WHERE task_id=? AND status='awaiting_upscale'",
                (
                    claim_token,
                    now,
                    lease_expires_at,
                    worker_id,
                    now,
                    row["task_id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            self._touch_worker(connection, worker_id, "busy", str(row["task_id"]), now)
            connection.commit()
            result = dict(row)
            result.update(
                status="upscaling",
                claim_token=claim_token,
                claimed_at=now,
                lease_expires_at=lease_expires_at,
                claimed_by=worker_id,
                lease_heartbeat_at=now,
                upscale_attempts=int(row["upscale_attempts"] or 0) + 1,
            )
            return result
        finally:
            connection.close()

    def renew_upscale_lease(
        self,
        task_id: str,
        claim_token: str,
        worker_id: str,
        lease_seconds: int,
        now: float | None = None,
    ) -> float | None:
        now = time.time() if now is None else now
        lease_expires_at = now + lease_seconds
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE tasks SET lease_expires_at=?,lease_heartbeat_at=? "
                "WHERE task_id=? AND status='upscaling' AND claim_token=? AND claimed_by=?",
                (lease_expires_at, now, task_id, claim_token, worker_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            self._touch_worker(connection, worker_id, "busy", task_id, now)
            connection.commit()
            return lease_expires_at
        finally:
            connection.close()

    def release_upscale(
        self,
        task_id: str,
        claim_token: str,
        worker_id: str,
        error_code: str,
        max_attempts: int,
        now: float | None = None,
    ) -> str | None:
        """Release a failed claim immediately or fail it after bounded attempts."""
        now = time.time() if now is None else now
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT upscale_attempts FROM tasks WHERE task_id=? AND status='upscaling' "
                "AND claim_token=? AND claimed_by=?",
                (task_id, claim_token, worker_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            attempts = int(row["upscale_attempts"] or 0)
            terminal = attempts >= max_attempts
            next_status = "failed" if terminal else "awaiting_upscale"
            connection.execute(
                "UPDATE tasks SET status=?,completed_at=?,error=?,claim_token=NULL,"
                "claimed_at=NULL,lease_expires_at=NULL,claimed_by=NULL,"
                "lease_heartbeat_at=NULL WHERE task_id=?",
                (
                    next_status,
                    now if terminal else None,
                    f"upscale worker failure: {error_code}" if terminal else None,
                    task_id,
                ),
            )
            self._touch_worker(connection, worker_id, "idle", None, now)
            connection.execute(
                "UPDATE upscale_workers SET released_count=released_count+1 WHERE worker_id=?",
                (worker_id,),
            )
            connection.commit()
            return next_status
        finally:
            connection.close()

    def worker_snapshot(
        self, active_window_seconds: int = 120, now: float | None = None
    ) -> list[dict[str, Any]]:
        now = time.time() if now is None else now
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT worker_id,first_seen_at,last_seen_at,state,current_task_id,"
                "completed_count,released_count FROM upscale_workers "
                "WHERE last_seen_at>=? ORDER BY worker_id",
                (now - active_window_seconds,),
            ).fetchall()
        return [dict(row) for row in rows]


GeneratorFactory = Callable[[Settings], GptImageGenerator]


class TaskManager:
    def __init__(
        self,
        store: TaskStore,
        source_dir: Path = SOURCE_IMAGE_DIR,
        generator_factory: GeneratorFactory = GptImageGenerator,
        generation_limiter: WeightedLimiter | None = None,
    ):
        self.store = store
        self.source_dir = source_dir
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.generator_factory = generator_factory
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.stop_event = threading.Event()
        self.workers: list[threading.Thread] = []
        self._worker_lock = threading.Lock()
        self._busy_workers = 0
        self._worker_sequence = 0
        self.controller: threading.Thread | None = None
        self._native_batch_condition = threading.Condition()
        self._native_batch_supported: bool | None = None
        self._native_batch_probe_in_progress = False
        legacy_max = os.getenv("IMAGE_GENERATION_WORKERS", "").strip()
        default_max = (
            _bounded_int_env(
                "IMAGE_GENERATION_WORKERS",
                DEFAULT_GENERATION_MAX_WORKERS,
                MAX_GENERATION_CONCURRENCY,
            )
            if legacy_max
            else DEFAULT_GENERATION_MAX_WORKERS
        )
        self.max_worker_count = _bounded_int_env(
            "IMAGE_GENERATION_MAX_WORKERS",
            default_max,
            MAX_GENERATION_CONCURRENCY,
        )
        self.min_worker_count = _bounded_int_env(
            "IMAGE_GENERATION_MIN_WORKERS",
            DEFAULT_GENERATION_MIN_WORKERS,
            self.max_worker_count,
        )
        self.idle_retire_seconds = _positive_int_env(
            "IMAGE_GENERATION_IDLE_RETIRE_SECONDS", 30
        )
        self.shutdown_timeout_seconds = _positive_int_env(
            "IMAGE_GENERATION_SHUTDOWN_TIMEOUT_SECONDS", 990
        )
        # A worker consumes one queued batch, while a native batch can represent
        # up to five logical image generations. Keep the elastic worker ceiling
        # separate from the process-wide logical image concurrency ceiling so a
        # legal n=5 request still works with the conservative three-worker default.
        self.generation_limiter = generation_limiter or WeightedLimiter(
            MAX_GENERATION_CONCURRENCY
        )

    @property
    def worker_count(self) -> int:
        """Compatibility name for the configured generation ceiling."""
        return self.max_worker_count

    @property
    def active_worker_count(self) -> int:
        with self._worker_lock:
            return sum(worker.is_alive() for worker in self.workers)

    @property
    def busy_worker_count(self) -> int:
        with self._worker_lock:
            return self._busy_workers

    def _start_worker_locked(self) -> None:
        self._worker_sequence += 1
        worker = threading.Thread(
            target=self._worker,
            name=f"image-generation-worker-{self._worker_sequence}",
            daemon=True,
        )
        self.workers.append(worker)
        worker.start()

    def _ensure_capacity(self) -> None:
        if self.stop_event.is_set():
            return
        with self._worker_lock:
            self.workers = [worker for worker in self.workers if worker.is_alive()]
            active = len(self.workers)
            outstanding = self.queue.qsize() + self._busy_workers
            desired = min(
                self.max_worker_count,
                max(self.min_worker_count, outstanding),
            )
            for _ in range(max(0, desired - active)):
                self._start_worker_locked()

    def _controller(self) -> None:
        while not self.stop_event.wait(0.5):
            self._ensure_capacity()

    def start(self) -> None:
        if self.controller and self.controller.is_alive():
            return
        with self._worker_lock:
            if any(worker.is_alive() for worker in self.workers):
                raise RuntimeError(
                    "cannot restart generation manager while old workers are still active"
                )
        self.stop_event.clear()
        for batch_id in self.store.queued_batch_ids():
            self.queue.put(batch_id)
        self._ensure_capacity()
        self.controller = threading.Thread(
            target=self._controller,
            name="image-generation-elastic-controller",
            daemon=True,
        )
        self.controller.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self._worker_lock:
            workers = list(self.workers)
        for _ in workers:
            self.queue.put(None)
        deadline = time.monotonic() + self.shutdown_timeout_seconds
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if self.controller:
            self.controller.join(timeout=max(0.0, deadline - time.monotonic()))
            self.controller = None
        with self._worker_lock:
            self.workers = [worker for worker in self.workers if worker.is_alive()]
            if not self.workers:
                self._busy_workers = 0
                # Queued work remains durable in SQLite. Drop stale in-memory
                # batch IDs and wake-up sentinels so a same-process test restart
                # cannot double-enqueue or immediately retire fresh workers.
                self.queue = queue.Queue()

    def submit(self, prompt: str, size: str) -> dict[str, Any]:
        return self.submit_batch(prompt, size, count=1, requested_concurrency=1)[0]

    def submit_batch(
        self, prompt: str, size: str, count: int, requested_concurrency: int
    ) -> list[dict[str, Any]]:
        records = self.store.create_batch(
            prompt,
            size,
            count=count,
            requested_concurrency=requested_concurrency,
        )
        self.queue.put(records[0]["batch_id"])
        self._ensure_capacity()
        return records

    def submit_batch_idempotent(
        self,
        prompt: str,
        size: str,
        count: int,
        requested_concurrency: int,
        tenant_scope: str,
        idempotency_key: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        records, reused = self.store.create_batch_idempotent(
            prompt,
            size,
            count,
            requested_concurrency,
            tenant_scope,
            idempotency_key,
        )
        if not reused:
            self.queue.put(records[0]["batch_id"])
            self._ensure_capacity()
        return records, reused

    def _worker(self) -> None:
        current = threading.current_thread()
        try:
            while not self.stop_event.is_set():
                try:
                    batch_id = self.queue.get(timeout=self.idle_retire_seconds)
                except queue.Empty:
                    with self._worker_lock:
                        active = sum(worker.is_alive() for worker in self.workers)
                    if active > self.min_worker_count and self.queue.empty():
                        return
                    continue
                if batch_id is None:
                    self.queue.task_done()
                    return
                with self._worker_lock:
                    self._busy_workers += 1
                try:
                    self._run_batch(batch_id)
                finally:
                    with self._worker_lock:
                        self._busy_workers -= 1
                    self.queue.task_done()
        finally:
            with self._worker_lock:
                self.workers = [worker for worker in self.workers if worker is not current]

    def _run_task(self, task_id: str) -> None:
        """Compatibility entry point used by older callers and focused tests."""
        task = self.store.get(task_id)
        if task:
            self._run_batch(str(task["batch_id"]))

    def _native_batch_route(self) -> str:
        """Choose native/fallback while allowing only one initial capability probe."""
        with self._native_batch_condition:
            while (
                self._native_batch_supported is None
                and self._native_batch_probe_in_progress
            ):
                self._native_batch_condition.wait()
            if self._native_batch_supported is False:
                return "fallback"
            if self._native_batch_supported is True:
                return "native"
            self._native_batch_probe_in_progress = True
            return "probe"

    def _finish_native_batch_probe(self, supported: bool | None) -> None:
        with self._native_batch_condition:
            if supported is not None:
                self._native_batch_supported = supported
            self._native_batch_probe_in_progress = False
            self._native_batch_condition.notify_all()

    def _remember_native_batch_unsupported(self) -> None:
        with self._native_batch_condition:
            self._native_batch_supported = False
            self._native_batch_probe_in_progress = False
            self._native_batch_condition.notify_all()

    def _run_batch(self, batch_id: str) -> None:
        started_at = time.time()
        tasks = self.store.begin_batch_generation(batch_id, started_at)
        if not tasks:
            return
        quality = os.getenv("IMAGE_FIXED_QUALITY", "low").strip().lower()
        try:
            settings = Settings.from_env(require_key=True)
        except Exception as exc:
            for task in tasks:
                self._fail_task(task["task_id"], exc)
            return
        if len(tasks) == 1:
            task = tasks[0]
            work_dir = self.source_dir / f".{task['task_id']}-{uuid.uuid4().hex}.tmp"
            try:
                with self.generation_limiter.slot(1):
                    generation = self.generator_factory(settings).generate(
                        task["prompt"],
                        quality,
                        work_dir,
                        idempotency_key=task["generation_idempotency_key"],
                    )
                self._persist_generation(task, generation, started_at, "single")
            except Exception as exc:
                self._fail_task(task["task_id"], exc)
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
            return

        native_dir = self.source_dir / f".batch-{batch_id}-{uuid.uuid4().hex}.tmp"
        try:
            route = self._native_batch_route()
            if route == "fallback":
                self._run_fallback(tasks, settings, quality, started_at)
                return
            try:
                with self.generation_limiter.slot(len(tasks)):
                    generations = self.generator_factory(settings).generate_many(
                        tasks[0]["prompt"],
                        quality,
                        native_dir,
                        n=len(tasks),
                        idempotency_key=tasks[0]["generation_idempotency_key"],
                    )
            except NativeBatchIncomplete as exc:
                self._remember_native_batch_unsupported()
                partial_count = len(exc.partial_results)
                if partial_count >= len(tasks):
                    raise RuntimeError(
                        "incomplete native batch reported an invalid partial result count"
                    ) from exc
                for task, generation in zip(
                    tasks[:partial_count], exc.partial_results, strict=True
                ):
                    try:
                        self._persist_generation(
                            task, generation, started_at, "native_n_partial"
                        )
                    except Exception as persist_exc:
                        self._fail_task(task["task_id"], persist_exc)
                self._run_fallback(
                    tasks[partial_count:], settings, quality, started_at
                )
                return
            except NativeBatchUnsupported:
                self._remember_native_batch_unsupported()
                self._run_fallback(tasks, settings, quality, started_at)
                return
            except Exception:
                if route == "probe":
                    self._finish_native_batch_probe(None)
                raise
            if route == "probe":
                self._finish_native_batch_probe(True)
            if len(generations) != len(tasks):
                raise RuntimeError("native batch returned an unexpected result count")
            for task, generation in zip(tasks, generations, strict=True):
                try:
                    self._persist_generation(task, generation, started_at, "native_n")
                except Exception as exc:
                    self._fail_task(task["task_id"], exc)
        except Exception as exc:
            for task in tasks:
                if (self.store.get(task["task_id"]) or {}).get("status") == "processing":
                    self._fail_task(task["task_id"], exc)
        finally:
            shutil.rmtree(native_dir, ignore_errors=True)

    def _run_fallback(
        self,
        tasks: list[dict[str, Any]],
        settings: Settings,
        quality: str,
        started_at: float,
    ) -> None:
        requested = int(tasks[0]["requested_concurrency"] or 1)
        worker_count = min(
            requested,
            self.worker_count,
            len(tasks),
            MAX_GENERATION_CONCURRENCY,
        )

        def generate_one(task: dict[str, Any]) -> None:
            work_dir = self.source_dir / f".{task['task_id']}-{uuid.uuid4().hex}.tmp"
            try:
                with self.generation_limiter.slot(1):
                    generation = self.generator_factory(settings).generate(
                        task["prompt"],
                        quality,
                        work_dir,
                        idempotency_key=task["generation_idempotency_key"],
                    )
                self._persist_generation(task, generation, started_at, "single_fallback")
            except Exception as exc:
                self._fail_task(task["task_id"], exc)
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"image-fallback-{tasks[0]['batch_id'][:8]}",
        ) as executor:
            futures = [executor.submit(generate_one, task) for task in tasks]
            for future in as_completed(futures):
                future.result()

    def _persist_generation(
        self,
        task: dict[str, Any],
        generation: Any,
        started_at: float,
        generation_mode: str,
    ) -> None:
        task_id = str(task["task_id"])
        generated_path = Path(generation.image.path)
        source_filename = f"{task_id}{generated_path.suffix.lower()}"
        source_path = self.source_dir / source_filename
        os.replace(generated_path, source_path)
        image = inspect_image(source_path)
        metrics = {
            "model": generation.requested_model,
            "quality": generation.requested_quality,
            "requested_source_size": generation.requested_size,
            "provider_request_id": generation.request_id,
            "usage": generation.usage,
            "generation_mode": generation_mode,
            "batch_id": task["batch_id"],
            "batch_index": task["batch_index"],
            "batch_size": task["batch_size"],
            "requested_concurrency": task["requested_concurrency"],
            "source_pixels": [image.width, image.height],
            "source_bytes": image.file_bytes,
            "api_seconds": generation.api_seconds,
            "download_seconds": generation.download_seconds,
            "generation_total_seconds": generation.total_seconds,
            "queue_wait_seconds": round(started_at - task["created_at"], 3),
        }
        self.store.update(
            task_id,
            status="awaiting_upscale",
            source_filename=source_filename,
            source_width=image.width,
            source_height=image.height,
            source_file_bytes=image.file_bytes,
            source_sha256=image.sha256,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
        )

    def _fail_task(self, task_id: str, exc: Exception) -> None:
        self.store.update(
            task_id,
            status="failed",
            completed_at=time.time(),
            error=f"{type(exc).__name__}: {_scrub_error(exc)}",
        )


store = TaskStore(DATABASE_PATH)
manager = TaskManager(store)


@asynccontextmanager
async def lifespan(_: FastAPI) -> Iterator[None]:
    manager.start()
    try:
        yield
    finally:
        manager.stop()


app = FastAPI(
    title="Image Generation Service",
    version="4.0.0",
    lifespan=lifespan,
)
app.mount("/images", StaticFiles(directory=PUBLIC_IMAGE_DIR), name="public_images")


def _bearer_value(authorization: str | None) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:]
    return ""


def _secret_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return bool(normalized) and (
        (normalized.startswith("<") and normalized.endswith(">"))
        or normalized.startswith(("change-me", "replace-me", "set-with-"))
    )


def _service_token_ready() -> bool:
    token = os.getenv("IMAGE_SERVICE_TOKEN", "")
    return bool(token) and not _secret_placeholder(token)


def _service_auth_required() -> bool:
    return os.getenv("IMAGE_REQUIRE_SERVICE_AUTH", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("IMAGE_SERVICE_TOKEN", "")
    if _secret_placeholder(expected) or (_service_auth_required() and not expected):
        raise HTTPException(status_code=503, detail="service authentication is not configured")
    if expected and not secrets.compare_digest(_bearer_value(authorization), expected):
        raise HTTPException(status_code=401, detail="invalid service token")


def _worker_token_ready() -> bool:
    worker_token = os.getenv("IMAGE_UPSCALE_WORKER_TOKEN", "")
    service_token = os.getenv("IMAGE_SERVICE_TOKEN", "")
    return not _secret_placeholder(worker_token) and len(worker_token) >= 32 and (
        not service_token or not secrets.compare_digest(worker_token, service_token)
    )


def require_worker_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("IMAGE_UPSCALE_WORKER_TOKEN", "")
    if not _worker_token_ready():
        raise HTTPException(status_code=503, detail="upscale worker authentication is not configured")
    if not secrets.compare_digest(_bearer_value(authorization), expected):
        raise HTTPException(status_code=401, detail="invalid worker token")


def _litellm_backend_token_ready() -> bool:
    token = os.getenv("IMAGE_LITELLM_BACKEND_TOKEN", "")
    other_tokens = {
        os.getenv("IMAGE_SERVICE_TOKEN", ""),
        os.getenv("IMAGE_UPSCALE_WORKER_TOKEN", ""),
    }
    return not _secret_placeholder(token) and len(token) >= 32 and token not in other_tokens


def require_litellm_backend_token(
    authorization: str | None = Header(default=None),
) -> None:
    expected = os.getenv("IMAGE_LITELLM_BACKEND_TOKEN", "")
    if not _litellm_backend_token_ready():
        raise HTTPException(status_code=503, detail="gateway backend authentication is not configured")
    if not secrets.compare_digest(_bearer_value(authorization), expected):
        raise HTTPException(status_code=401, detail="invalid gateway backend token")


def _validated_worker_id(value: str | None) -> str:
    worker_id = (value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", worker_id):
        raise HTTPException(status_code=422, detail="invalid worker id")
    return worker_id


def public_base_url(request: Request) -> str:
    configured = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    forwarded_host = request.headers.get("x-forwarded-host", "").split(",", 1)[0].strip()
    scheme = forwarded_proto or request.url.scheme
    host = forwarded_host or request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}".rstrip("/")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(WEB_INDEX_PATH, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict[str, Any]:
    settings = Settings.from_env(require_key=False)
    return {
        "status": "ok",
        "version": app.version,
        "role": "cloud",
        "api_key_configured": bool(settings.api_key),
        "fixed_quality": os.getenv("IMAGE_FIXED_QUALITY", "low"),
        "generation_worker_count": manager.active_worker_count,
        "generation_workers_active": manager.active_worker_count,
        "generation_workers_busy": manager.busy_worker_count,
        "generation_workers_min": manager.min_worker_count,
        "generation_workers_max": manager.max_worker_count,
        "generation_concurrency_hard_limit": MAX_GENERATION_CONCURRENCY,
        "generation_slots_in_use": manager.generation_limiter.current,
        "task_counts": store.counts(),
        "service_auth_enabled": _service_token_ready(),
        "service_auth_required": _service_auth_required(),
        "upscale_worker_auth_configured": _worker_token_ready(),
        "litellm_backend_auth_configured": _litellm_backend_token_ready(),
    }


@app.post("/v1/generate", status_code=status.HTTP_202_ACCEPTED)
def submit_generation(
    payload: GenerateTaskRequest,
    request: Request,
    _: None = Depends(require_service_token),
) -> dict[str, Any]:
    tasks = manager.submit_batch(
        payload.prompt,
        payload.size,
        count=payload.count,
        requested_concurrency=payload.concurrency,
    )
    base_url = public_base_url(request)
    response: dict[str, Any] = {
        "batch_id": tasks[0]["batch_id"],
        "count": payload.count,
        "concurrency": payload.concurrency,
        "task_ids": [task["task_id"] for task in tasks],
        "result_urls": [
            f"{base_url}/v1/result/{task['task_id']}" for task in tasks
        ],
        "batch_result_url": f"{base_url}/v1/batch/{tasks[0]['batch_id']}",
        "status": "queued",
        "size": payload.size,
    }
    if payload.count == 1:
        response["task_id"] = tasks[0]["task_id"]
        response["result_url"] = response["result_urls"][0]
    return response


def _task_result_payload(task: dict[str, Any], request: Request) -> dict[str, Any]:
    response: dict[str, Any] = {
        "task_id": task["task_id"],
        "batch_id": task["batch_id"],
        "batch_index": task["batch_index"],
        "batch_size": task["batch_size"],
        "status": task["status"],
        "prompt": task["prompt"],
        "size": task["size"],
        "created_at": _utc_iso(task["created_at"]),
        "started_at": _utc_iso(task["started_at"]) if task["started_at"] else None,
        "completed_at": _utc_iso(task["completed_at"]) if task["completed_at"] else None,
    }
    if task["status"] == "failed":
        response["error"] = task["error"]
    if task["status"] == "done":
        response.update(
            {
                "image_url": f"{public_base_url(request)}/images/{task['image_filename']}",
                "actual_pixels": [task["width"], task["height"]],
                "file_bytes": task["file_bytes"],
                "sha256": task["sha256"],
                "metrics": json.loads(task["metrics_json"] or "{}"),
                "cost": json.loads(task["cost_json"] or "{}"),
            }
        )
    return response


@app.get("/v1/result/{task_id}", name="get_result")
def get_result(
    task_id: str,
    request: Request,
    _: None = Depends(require_service_token),
) -> dict[str, Any]:
    task = store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return _task_result_payload(task, request)


@app.get("/v1/batch/{batch_id}")
def get_batch_result(
    batch_id: str,
    request: Request,
    _: None = Depends(require_service_token),
) -> dict[str, Any]:
    tasks = store.get_batch(batch_id)
    if not tasks:
        raise HTTPException(status_code=404, detail="batch not found")
    counts: dict[str, int] = {}
    for task in tasks:
        task_status = str(task["status"])
        counts[task_status] = counts.get(task_status, 0) + 1
    terminal = counts.get("done", 0) + counts.get("failed", 0)
    if counts.get("done", 0) == len(tasks):
        batch_status = "done"
    elif counts.get("failed", 0) == len(tasks):
        batch_status = "failed"
    elif terminal == len(tasks):
        batch_status = "partial_failed"
    elif counts.get("queued", 0) == len(tasks):
        batch_status = "queued"
    else:
        batch_status = "processing"
    return {
        "batch_id": batch_id,
        "status": batch_status,
        "count": len(tasks),
        "concurrency": tasks[0]["requested_concurrency"],
        "summary": counts,
        "results": [_task_result_payload(task, request) for task in tasks],
    }


async def _wait_for_terminal_batch(
    batch_id: str, timeout_seconds: int
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        tasks = await asyncio.to_thread(store.get_batch, batch_id)
        if tasks and all(task["status"] in {"done", "failed"} for task in tasks):
            return tasks
        await asyncio.sleep(0.5)
    raise HTTPException(status_code=504, detail="image generation timed out")


@app.post("/v1/images/generations", include_in_schema=False)
async def openai_image_generation_backend(
    payload: OpenAIImageGenerationRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    _: None = Depends(require_litellm_backend_token),
) -> dict[str, Any]:
    private_model = os.getenv(
        "IMAGE_LITELLM_PRIVATE_MODEL", "image-pipeline-private"
    ).strip()
    public_alias = os.getenv("IMAGE_PUBLIC_MODEL_ALIAS", "image-gen").strip()
    if not private_model or not public_alias:
        raise HTTPException(status_code=503, detail="gateway model mapping is not configured")
    if not secrets.compare_digest(payload.model, private_model):
        raise HTTPException(status_code=403, detail="private model mapping rejected")
    if idempotency_key is not None and not 8 <= len(idempotency_key) <= 200:
        raise HTTPException(status_code=422, detail="invalid Idempotency-Key")

    if idempotency_key:
        try:
            tasks, _ = await asyncio.to_thread(
                manager.submit_batch_idempotent,
                payload.prompt,
                payload.size,
                payload.n,
                min(payload.n, MAX_BATCH_SIZE),
                payload.user,
                idempotency_key,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
    else:
        tasks = await asyncio.to_thread(
            manager.submit_batch,
            payload.prompt,
            payload.size,
            payload.n,
            min(payload.n, MAX_BATCH_SIZE),
        )

    timeout_seconds = _positive_int_env("IMAGE_LITELLM_SYNC_TIMEOUT_SECONDS", 600)
    completed = await _wait_for_terminal_batch(
        str(tasks[0]["batch_id"]), timeout_seconds
    )
    failed = [task for task in completed if task["status"] != "done"]
    if failed:
        raise HTTPException(status_code=502, detail="one or more images failed to generate")
    return {
        "created": int(time.time()),
        "model": public_alias,
        "data": [
            {
                "url": f"{public_base_url(request)}/images/{task['image_filename']}"
            }
            for task in completed
        ],
    }


@app.get("/internal/gateway/stats", include_in_schema=False)
def gateway_stats(
    _: None = Depends(require_litellm_backend_token),
) -> dict[str, Any]:
    stats = store.operational_stats()
    active_gpu_workers = store.worker_snapshot()
    stats.update(
        generation_workers_active=manager.active_worker_count,
        generation_workers_busy=manager.busy_worker_count,
        generation_workers_min=manager.min_worker_count,
        generation_workers_max=manager.max_worker_count,
        generation_slots_in_use=manager.generation_limiter.current,
        generation_slots_limit=manager.generation_limiter.capacity,
        active_gpu_workers=len(active_gpu_workers),
        busy_gpu_workers=sum(
            1 for worker in active_gpu_workers if worker["state"] == "busy"
        ),
    )
    return stats


@app.get("/internal/upscale/claim", response_model=None)
def claim_upscale(
    request: Request,
    x_worker_id: str | None = Header(default=None, alias="X-Worker-ID"),
    _: None = Depends(require_worker_token),
) -> Response | dict[str, Any]:
    # Missing IDs are accepted only for rolling upgrades from the pre-pool worker.
    worker_id = _validated_worker_id(x_worker_id or "legacy-worker")
    lease_seconds = _positive_int_env(
        "IMAGE_UPSCALE_LEASE_SECONDS", DEFAULT_LEASE_SECONDS
    )
    task = store.claim_upscale(lease_seconds, worker_id=worker_id)
    if task is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return {
        "task_id": task["task_id"],
        "size": task["size"],
        "claim_token": task["claim_token"],
        "lease_expires_at": _utc_iso(task["lease_expires_at"]),
        "lease_seconds": lease_seconds,
        "heartbeat_interval_seconds": max(2, min(60, lease_seconds // 3)),
        "source_url": f"{public_base_url(request)}/internal/upscale/source/{task['task_id']}",
    }


@app.post("/internal/upscale/heartbeat")
def heartbeat_upscale(
    task_id: str = Form(...),
    claim_token: str = Form(...),
    worker_id: str = Form(...),
    _: None = Depends(require_worker_token),
) -> dict[str, Any]:
    worker_id = _validated_worker_id(worker_id)
    lease_seconds = _positive_int_env(
        "IMAGE_UPSCALE_LEASE_SECONDS", DEFAULT_LEASE_SECONDS
    )
    lease_expires_at = store.renew_upscale_lease(
        task_id,
        claim_token,
        worker_id,
        lease_seconds,
    )
    if lease_expires_at is None:
        raise HTTPException(status_code=409, detail="claim is no longer active")
    return {
        "task_id": task_id,
        "status": "upscaling",
        "lease_expires_at": _utc_iso(lease_expires_at),
    }


@app.post("/internal/upscale/release")
def release_upscale(
    task_id: str = Form(...),
    claim_token: str = Form(...),
    worker_id: str = Form(...),
    error_code: str = Form(default="worker_error"),
    _: None = Depends(require_worker_token),
) -> dict[str, Any]:
    worker_id = _validated_worker_id(worker_id)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", error_code):
        raise HTTPException(status_code=422, detail="invalid worker error code")
    next_status = store.release_upscale(
        task_id,
        claim_token,
        worker_id,
        error_code,
        _bounded_int_env(
            "IMAGE_UPSCALE_MAX_ATTEMPTS", DEFAULT_UPSCALE_MAX_ATTEMPTS, 10
        ),
    )
    if next_status is None:
        raise HTTPException(status_code=409, detail="claim is no longer active")
    return {"task_id": task_id, "status": next_status}


@app.get("/internal/upscale/workers")
def list_upscale_workers(
    _: None = Depends(require_worker_token),
) -> dict[str, Any]:
    workers = store.worker_snapshot()
    return {"active_workers": len(workers), "workers": workers}


@app.get("/internal/upscale/source/{task_id}")
def download_source(
    task_id: str,
    x_claim_token: str | None = Header(default=None, alias="X-Claim-Token"),
    _: None = Depends(require_worker_token),
) -> FileResponse:
    task = store.get(task_id)
    if not task or task["status"] != "upscaling" or not task["claim_token"]:
        raise HTTPException(status_code=404, detail="claimed source not found")
    if not x_claim_token or not secrets.compare_digest(x_claim_token, task["claim_token"]):
        raise HTTPException(status_code=403, detail="invalid claim token")
    path = SOURCE_IMAGE_DIR / task["source_filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="source image not found")
    return FileResponse(path, media_type="application/octet-stream")


def _done_response(task: dict[str, Any], request: Request) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "status": "done",
        "image_url": f"{public_base_url(request)}/images/{task['image_filename']}",
        "actual_pixels": [task["width"], task["height"]],
        "file_bytes": task["file_bytes"],
        "sha256": task["sha256"],
    }


@app.post("/internal/upscale/submit")
def submit_upscale(
    request: Request,
    task_id: str = Form(...),
    claim_token: str = Form(...),
    worker_id: str | None = Form(default=None),
    source_download_seconds: float | None = Form(default=None),
    upscale_seconds: float | None = Form(default=None),
    postprocess_seconds: float | None = Form(default=None),
    peak_vram_mib: float | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
    _: None = Depends(require_worker_token),
) -> dict[str, Any]:
    uploaded = image or file
    if uploaded is None:
        raise HTTPException(status_code=422, detail="missing uploaded image")
    task = store.get(task_id)
    idempotent_done = bool(
        task
        and task["status"] == "done"
        and task["submitted_claim_token"]
        and secrets.compare_digest(claim_token, task["submitted_claim_token"])
    )
    if not idempotent_done:
        if not task or task["status"] != "upscaling" or not task["claim_token"]:
            raise HTTPException(status_code=409, detail="task is not actively claimed")
        if not secrets.compare_digest(claim_token, task["claim_token"]):
            raise HTTPException(status_code=403, detail="invalid claim token")
    claimed_by = str(task.get("claimed_by") or "legacy-worker") if task else "legacy-worker"
    effective_worker_id = _validated_worker_id(worker_id or claimed_by)
    if claimed_by != "legacy-worker" and not secrets.compare_digest(
        effective_worker_id, claimed_by
    ):
        raise HTTPException(status_code=403, detail="claim belongs to another worker")
    numeric_metrics = {
        "source_download_seconds": source_download_seconds,
        "upscale_seconds": upscale_seconds,
        "postprocess_seconds": postprocess_seconds,
        "peak_vram_mib": peak_vram_mib,
    }
    if any(
        value is not None
        and (not math.isfinite(value) or value < 0 or value > 86_400)
        for value in numeric_metrics.values()
    ):
        raise HTTPException(status_code=422, detail="invalid worker metrics")

    maximum = _positive_int_env("IMAGE_UPSCALE_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)
    temp_path = PUBLIC_IMAGE_DIR / f".{task_id}-{uuid.uuid4().hex}.upload"
    received = 0
    try:
        with temp_path.open("xb") as output:
            while chunk := uploaded.file.read(UPLOAD_CHUNK_BYTES):
                received += len(chunk)
                if received > maximum:
                    raise HTTPException(status_code=413, detail="uploaded image is too large")
                output.write(chunk)
        try:
            metadata = inspect_image(temp_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="uploaded file is not a valid PNG image") from exc
        expected = TARGET_PIXELS[task["size"]]
        if metadata.format != "PNG":
            raise HTTPException(status_code=422, detail="uploaded file must be PNG")
        if (metadata.width, metadata.height) != expected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Pillow pixel verification failed: expected {expected[0]}x{expected[1]}, "
                    f"got {metadata.width}x{metadata.height}"
                ),
            )

        image_filename = f"{task_id}.png"
        final_path = PUBLIC_IMAGE_DIR / image_filename
        connection = store.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current_row = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            current = dict(current_row) if current_row else None
            if (
                current
                and current["status"] == "done"
                and current["submitted_claim_token"]
                and secrets.compare_digest(claim_token, current["submitted_claim_token"])
            ):
                connection.commit()
                return _done_response(current, request)
            if (
                not current
                or current["status"] != "upscaling"
                or not current["claim_token"]
                or not secrets.compare_digest(claim_token, current["claim_token"])
            ):
                connection.rollback()
                raise HTTPException(status_code=409, detail="claim is no longer active")
            os.replace(temp_path, final_path)
            completed_at = time.time()
            metrics = json.loads(current["metrics_json"] or "{}")
            metrics.update(
                worker_id=effective_worker_id,
                **{key: value for key, value in numeric_metrics.items() if value is not None},
                final_pixels=[metadata.width, metadata.height],
                final_bytes=metadata.file_bytes,
                final_sha256=metadata.sha256,
                upscale_claimed_at=_utc_iso(current["claimed_at"]),
                completed_at=_utc_iso(completed_at),
                remote_stage_seconds=round(completed_at - current["claimed_at"], 3),
                end_to_end_seconds=round(completed_at - current["created_at"], 3),
            )
            connection.execute(
                "UPDATE tasks SET status='done',completed_at=?,image_filename=?,"
                "local_path=NULL,width=?,height=?,file_bytes=?,sha256=?,manifest_path=NULL,"
                "metrics_json=?,submitted_claim_token=?,claim_token=NULL,"
                "lease_expires_at=NULL,lease_heartbeat_at=NULL "
                "WHERE task_id=?",
                (
                    completed_at,
                    image_filename,
                    metadata.width,
                    metadata.height,
                    metadata.file_bytes,
                    metadata.sha256,
                    json.dumps(metrics, ensure_ascii=False),
                    claim_token,
                    task_id,
                ),
            )
            self_worker_id = str(current["claimed_by"] or effective_worker_id)
            if self_worker_id == "legacy-worker":
                self_worker_id = effective_worker_id
            TaskStore._touch_worker(
                connection, self_worker_id, "idle", None, completed_at
            )
            connection.execute(
                "UPDATE upscale_workers SET completed_count=completed_count+1 "
                "WHERE worker_id=?",
                (self_worker_id,),
            )
            connection.commit()
        finally:
            connection.close()
        try:
            (SOURCE_IMAGE_DIR / current["source_filename"]).unlink(missing_ok=True)
        except OSError:
            # Delivery is already committed; the scheduled cleanup job will retry.
            pass
        return _done_response(store.get(task_id) or {}, request)
    finally:
        uploaded.file.close()
        temp_path.unlink(missing_ok=True)
