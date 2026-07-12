from __future__ import annotations

import io
import json
import sqlite3
import threading
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import image_pipeline.service as service
from image_pipeline.models import GenerationResult


WORKER_TOKEN = "worker-test-token-with-at-least-32-bytes"
WORKER_HEADERS = {"Authorization": f"Bearer {WORKER_TOKEN}"}


def _png_bytes(size: tuple[int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(output, format="PNG")
    return output.getvalue()


def _claimed_task(store: service.TaskStore, source_dir: Path, size: str = "2k") -> dict:
    task = store.create("test prompt", size)
    source_filename = f"{task['task_id']}.png"
    (source_dir / source_filename).write_bytes(_png_bytes((32, 24)))
    store.update(task["task_id"], status="awaiting_upscale", source_filename=source_filename)
    claimed = store.claim_upscale(lease_seconds=60)
    assert claimed is not None
    return claimed


def _install_test_store(monkeypatch, tmp_path: Path) -> service.TaskStore:
    store = service.TaskStore(tmp_path / "tasks.db")
    source_dir = tmp_path / "sources"
    public_dir = tmp_path / "images"
    source_dir.mkdir()
    public_dir.mkdir()
    monkeypatch.setattr(service, "store", store)
    monkeypatch.setattr(service, "SOURCE_IMAGE_DIR", source_dir)
    monkeypatch.setattr(service, "PUBLIC_IMAGE_DIR", public_dir)
    monkeypatch.setenv("IMAGE_UPSCALE_WORKER_TOKEN", WORKER_TOKEN)
    return store


def test_schema_migrates_existing_database_without_dropping_rows(tmp_path: Path) -> None:
    database = tmp_path / "old.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE tasks (task_id TEXT PRIMARY KEY,prompt TEXT NOT NULL,size TEXT NOT NULL,"
            "status TEXT NOT NULL,created_at REAL NOT NULL,started_at REAL,completed_at REAL,"
            "error TEXT,image_filename TEXT,local_path TEXT,width INTEGER,height INTEGER,"
            "file_bytes INTEGER,sha256 TEXT,manifest_path TEXT,metrics_json TEXT,cost_json TEXT)"
        )
        connection.execute(
            "INSERT INTO tasks(task_id,prompt,size,status,created_at) VALUES('kept','p','2k','processing',1)"
        )
    store = service.TaskStore(database)
    row = store.get("kept")
    assert row is not None
    assert row["status"] == "queued"
    assert "lease_expires_at" in row


def test_claim_is_atomic_and_expired_lease_is_recovered(tmp_path: Path) -> None:
    store = service.TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "2k")
    store.update(task["task_id"], status="awaiting_upscale", source_filename="source.png")
    results: list[dict | None] = []
    barrier = threading.Barrier(2)

    def claim() -> None:
        barrier.wait()
        results.append(store.claim_upscale(lease_seconds=10, now=100))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result is not None for result in results) == 1
    first = next(result for result in results if result is not None)
    assert store.claim_upscale(lease_seconds=10, now=109) is None
    reclaimed = store.claim_upscale(lease_seconds=10, now=110)
    assert reclaimed is not None
    assert reclaimed["task_id"] == first["task_id"]
    assert reclaimed["claim_token"] != first["claim_token"]


def test_two_workers_claim_distinct_tasks_renew_and_release(tmp_path: Path) -> None:
    store = service.TaskStore(tmp_path / "tasks.db")
    tasks = [store.create(f"p-{index}", "2k") for index in range(2)]
    for task in tasks:
        store.update(
            task["task_id"],
            status="awaiting_upscale",
            source_filename=f"{task['task_id']}.png",
        )

    claims: list[dict | None] = []
    barrier = threading.Barrier(2)

    def claim(worker_id: str) -> None:
        barrier.wait()
        claims.append(
            store.claim_upscale(lease_seconds=30, worker_id=worker_id, now=100)
        )

    threads = [
        threading.Thread(target=claim, args=("gpu-a",)),
        threading.Thread(target=claim, args=("gpu-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    claimed = [item for item in claims if item is not None]
    assert len(claimed) == 2
    assert len({item["task_id"] for item in claimed}) == 2
    assert {item["claimed_by"] for item in claimed} == {"gpu-a", "gpu-b"}
    assert {row["state"] for row in store.worker_snapshot(now=100)} == {"busy"}

    first = claimed[0]
    renewed = store.renew_upscale_lease(
        first["task_id"],
        first["claim_token"],
        first["claimed_by"],
        lease_seconds=30,
        now=110,
    )
    assert renewed == 140
    assert (
        store.renew_upscale_lease(
            first["task_id"],
            first["claim_token"],
            "wrong-worker",
            lease_seconds=30,
            now=111,
        )
        is None
    )
    assert (
        store.release_upscale(
            first["task_id"],
            first["claim_token"],
            first["claimed_by"],
            "UpscaleError",
            max_attempts=3,
            now=112,
        )
        == "awaiting_upscale"
    )
    recovered = store.claim_upscale(30, worker_id="gpu-c", now=113)
    assert recovered is not None
    assert recovered["task_id"] == first["task_id"]
    assert recovered["claim_token"] != first["claim_token"]


def test_upscale_release_fails_after_bounded_attempts(tmp_path: Path) -> None:
    store = service.TaskStore(tmp_path / "tasks.db")
    task = store.create("p", "4k")
    store.update(task["task_id"], status="awaiting_upscale", source_filename="source.png")
    claimed = store.claim_upscale(30, worker_id="gpu-a", now=100)
    assert claimed is not None

    assert (
        store.release_upscale(
            claimed["task_id"],
            claimed["claim_token"],
            "gpu-a",
            "UpscaleError",
            max_attempts=1,
            now=101,
        )
        == "failed"
    )
    failed = store.get(task["task_id"])
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "upscale worker failure: UpscaleError"


def test_internal_auth_claim_and_source_download(monkeypatch, tmp_path: Path) -> None:
    store = _install_test_store(monkeypatch, tmp_path)
    claimed = _claimed_task(store, tmp_path / "sources")
    client = TestClient(service.app)

    assert client.get("/internal/upscale/claim").status_code == 401
    assert client.get("/internal/upscale/claim", headers=WORKER_HEADERS).status_code == 204
    source_url = f"/internal/upscale/source/{claimed['task_id']}"
    assert client.get(source_url, headers=WORKER_HEADERS).status_code == 403
    valid_headers = {**WORKER_HEADERS, "X-Claim-Token": claimed["claim_token"]}
    response = client.get(source_url, headers=valid_headers)
    assert response.status_code == 200
    assert response.content.startswith(b"\x89PNG")


def test_worker_api_records_identity_and_renews_lease(monkeypatch, tmp_path: Path) -> None:
    store = _install_test_store(monkeypatch, tmp_path)
    task = store.create("pool test", "2k")
    source_filename = f"{task['task_id']}.png"
    (tmp_path / "sources" / source_filename).write_bytes(_png_bytes((32, 24)))
    store.update(
        task["task_id"], status="awaiting_upscale", source_filename=source_filename
    )
    client = TestClient(service.app)
    headers = {**WORKER_HEADERS, "X-Worker-ID": "gpu-pool-a"}

    claimed = client.get("/internal/upscale/claim", headers=headers)
    assert claimed.status_code == 200
    payload = claimed.json()
    stored = store.get(task["task_id"])
    assert stored is not None
    assert stored["claimed_by"] == "gpu-pool-a"
    assert stored["upscale_attempts"] == 1

    heartbeat = client.post(
        "/internal/upscale/heartbeat",
        headers=WORKER_HEADERS,
        data={
            "task_id": task["task_id"],
            "claim_token": payload["claim_token"],
            "worker_id": "gpu-pool-a",
        },
    )
    assert heartbeat.status_code == 200
    workers = client.get("/internal/upscale/workers", headers=WORKER_HEADERS)
    assert workers.status_code == 200
    assert workers.json()["active_workers"] == 1
    assert workers.json()["workers"][0]["worker_id"] == "gpu-pool-a"

    wrong_worker = client.post(
        "/internal/upscale/release",
        headers=WORKER_HEADERS,
        data={
            "task_id": task["task_id"],
            "claim_token": payload["claim_token"],
            "worker_id": "gpu-pool-b",
            "error_code": "UpscaleError",
        },
    )
    assert wrong_worker.status_code == 409


def test_submit_validates_real_pixels_and_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    store = _install_test_store(monkeypatch, tmp_path)
    claimed = _claimed_task(store, tmp_path / "sources", "2k")
    client = TestClient(service.app)
    data = {
        "task_id": claimed["task_id"],
        "claim_token": claimed["claim_token"],
        "worker_id": "desktop-4090",
        "source_download_seconds": "1.25",
        "upscale_seconds": "12.5",
        "postprocess_seconds": "1.0",
        "peak_vram_mib": "1898.5",
    }

    wrong = client.post(
        "/internal/upscale/submit",
        headers=WORKER_HEADERS,
        data=data,
        files={"image": ("wrong.png", _png_bytes((64, 64)), "image/png")},
    )
    assert wrong.status_code == 422
    assert store.get(claimed["task_id"])["status"] == "upscaling"

    png = _png_bytes((2048, 2048))
    first = client.post(
        "/internal/upscale/submit",
        headers=WORKER_HEADERS,
        data=data,
        files={"image": ("final.png", png, "image/png")},
    )
    assert first.status_code == 200
    assert first.json()["actual_pixels"] == [2048, 2048]
    stored_metrics = json.loads(store.get(claimed["task_id"])["metrics_json"])
    assert stored_metrics["worker_id"] == "desktop-4090"
    assert stored_metrics["source_download_seconds"] == 1.25
    assert stored_metrics["upscale_seconds"] == 12.5
    assert stored_metrics["remote_stage_seconds"] >= 0
    second = client.post(
        "/internal/upscale/submit",
        headers=WORKER_HEADERS,
        data=data,
        files={"file": ("final.png", png, "image/png")},
    )
    assert second.status_code == 200
    assert second.json() == first.json()
    assert not (tmp_path / "sources" / claimed["source_filename"]).exists()
    final = tmp_path / "images" / f"{claimed['task_id']}.png"
    with Image.open(final) as image:
        image.load()
        assert image.size == (2048, 2048)


def test_submit_enforces_upload_limit_even_for_idempotent_retry(
    monkeypatch, tmp_path: Path
) -> None:
    store = _install_test_store(monkeypatch, tmp_path)
    claimed = _claimed_task(store, tmp_path / "sources", "2k")
    client = TestClient(service.app)
    data = {"task_id": claimed["task_id"], "claim_token": claimed["claim_token"]}
    png = _png_bytes((2048, 2048))
    first = client.post(
        "/internal/upscale/submit",
        headers=WORKER_HEADERS,
        data=data,
        files={"image": ("final.png", png, "image/png")},
    )
    assert first.status_code == 200

    monkeypatch.setenv("IMAGE_UPSCALE_MAX_UPLOAD_BYTES", "64")
    retry = client.post(
        "/internal/upscale/submit",
        headers=WORKER_HEADERS,
        data=data,
        files={"image": ("final.png", png, "image/png")},
    )
    assert retry.status_code == 413
    assert not list((tmp_path / "images").glob("*.upload"))


def test_submit_rejects_non_png_with_expected_pixels(monkeypatch, tmp_path: Path) -> None:
    store = _install_test_store(monkeypatch, tmp_path)
    claimed = _claimed_task(store, tmp_path / "sources", "2k")
    output = io.BytesIO()
    Image.new("RGB", (2048, 2048)).save(output, format="JPEG")
    response = TestClient(service.app).post(
        "/internal/upscale/submit",
        headers=WORKER_HEADERS,
        data={"task_id": claimed["task_id"], "claim_token": claimed["claim_token"]},
        files={"image": ("not-png.jpg", output.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 422


def test_generation_worker_stops_at_awaiting_upscale(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMAGE_API_KEY", "unit-test-key")
    monkeypatch.setenv("IMAGE_API_BASE_URL", "https://image-api.invalid/v1")
    store = service.TaskStore(tmp_path / "tasks.db")

    class FakeGenerator:
        def generate(self, prompt: str, quality: str, output_dir: Path) -> GenerationResult:
            output_dir.mkdir(parents=True)
            path = output_dir / "source.png"
            path.write_bytes(_png_bytes((1536, 1024)))
            metadata = service.inspect_image(path)
            return GenerationResult(
                requested_model="test-model",
                requested_quality=quality,
                requested_size="1536x1024",
                request_body={},
                status_code=200,
                request_id=None,
                api_seconds=1.0,
                download_seconds=0.2,
                total_seconds=1.2,
                response_image_trail="test",
                usage=None,
                image=metadata,
            )

    manager = service.TaskManager(
        store,
        source_dir=tmp_path / "sources",
        generator_factory=lambda _: FakeGenerator(),
    )
    manager.start()
    task = manager.submit("cloud generates only", "4k")
    manager.queue.join()
    manager.stop()
    result = store.get(task["task_id"])
    assert result is not None
    assert result["status"] == "awaiting_upscale"
    assert result["source_width"] == 1536
    assert result["source_height"] == 1024
    assert not result["image_filename"]


def test_public_done_response_never_exposes_server_paths(monkeypatch, tmp_path: Path) -> None:
    store = _install_test_store(monkeypatch, tmp_path)
    task = store.create("p", "2k")
    store.update(
        task["task_id"],
        status="done",
        completed_at=100,
        image_filename="final.png",
        local_path="/secret/server/path.png",
        manifest_path="/secret/manifest.json",
        width=2048,
        height=2048,
        file_bytes=123,
        sha256="abc",
        metrics_json="{}",
    )
    payload = TestClient(service.app).get(f"/v1/result/{task['task_id']}").json()
    assert payload["status"] == "done"
    assert "local_path" not in payload
    assert "manifest_path" not in payload
