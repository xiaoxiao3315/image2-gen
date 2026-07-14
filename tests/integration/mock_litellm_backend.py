from __future__ import annotations

import os
import secrets
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request


app = FastAPI(title="Mock private image backend")
captured: list[dict[str, Any]] = []


def _require_backend_token(request: Request) -> None:
    expected = os.environ["MOCK_BACKEND_TOKEN"]
    authorization = request.headers.get("authorization", "")
    actual = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    if not secrets.compare_digest(actual, expected):
        raise HTTPException(status_code=401, detail="invalid backend token")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/images/generations")
async def generate(request: Request) -> dict[str, Any]:
    _require_backend_token(request)
    payload = await request.json()
    count = int(payload.get("n", 1))
    captured.append(
        {
            "model": payload.get("model"),
            "n": count,
            "user": payload.get("user"),
            "idempotency_key": request.headers.get("idempotency-key"),
        }
    )
    return {
        "created": int(time.time()),
        "model": "image-gen",
        "data": [
            {"url": f"https://images.example.test/mock-{index + 1}.png"}
            for index in range(count)
        ],
    }


@app.get("/captured")
def get_captured() -> list[dict[str, Any]]:
    return list(captured)


@app.post("/reset")
def reset() -> dict[str, bool]:
    captured.clear()
    return {"reset": True}


@app.get("/internal/gateway/stats")
def gateway_stats(request: Request) -> dict[str, int]:
    _require_backend_token(request)
    return {"queue_length": 2, "generation_workers_active": 3}
