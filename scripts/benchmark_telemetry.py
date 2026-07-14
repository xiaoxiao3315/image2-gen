from __future__ import annotations

import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_pipeline.telemetry import TelemetryStore


def run(rounds: int = 250, workers: int = 8) -> dict[str, float | int]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        database = Path(directory) / "tasks.db"
        telemetry = TelemetryStore(database, timeout_seconds=1)
        baseline: list[float] = []
        observed: list[float] = []
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_rows(id INTEGER PRIMARY KEY,value INTEGER)"
            )
        for index in range(rounds):
            started = time.perf_counter()
            with sqlite3.connect(database) as connection:
                connection.execute("INSERT INTO benchmark_rows(value) VALUES(?)", (index,))
            baseline.append(time.perf_counter() - started)

            started = time.perf_counter()
            with sqlite3.connect(database) as connection:
                connection.execute("INSERT INTO benchmark_rows(value) VALUES(?)", (index,))
            telemetry.try_event(f"task-{index}", "accepted")
            observed.append(time.perf_counter() - started)

        success = 0
        lock = threading.Lock()

        def append(index: int) -> None:
            nonlocal success
            if telemetry.try_event(f"concurrent-{index}", "accepted"):
                with lock:
                    success += 1

        started = time.perf_counter()
        threads = [threading.Thread(target=append, args=(index,)) for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        concurrent_seconds = time.perf_counter() - started

        return {
            "rounds": rounds,
            "workers": workers,
            "baseline_median_ms": round(statistics.median(baseline) * 1000, 4),
            "observed_median_ms": round(statistics.median(observed) * 1000, 4),
            "median_overhead_ms": round(
                (statistics.median(observed) - statistics.median(baseline)) * 1000,
                4,
            ),
            "concurrent_success": success,
            "concurrent_seconds": round(concurrent_seconds, 4),
            "telemetry_failures": telemetry.failure_count,
        }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
