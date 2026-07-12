# Elastic service validation

This document records repeatable acceptance evidence for the staged conversion from
a single-machine demo to an elastic service. It intentionally contains no provider
address, public server address, API key, or bearer token.

## Stage 1: elastic GPU worker pool

Stage 1 keeps the existing outbound pull model: every GPU worker initiates its own
HTTP connection to the cloud queue. SQLite claims are serialized with
`BEGIN IMMEDIATE`, and every claim now records its worker identity, claim token,
lease heartbeat, expiry, and attempt count.

Recovery behavior:

- A live worker renews its claim while downloading, upscaling, and preparing upload.
- A processing failure releases the task immediately for another worker.
- A crashed or disconnected worker is recovered after lease expiry.
- A task becomes terminally failed after the configured bounded attempt count.
- A late submit from an expired claim is rejected, while a repeated successful submit
  with the same claim token is idempotent.

### 2026-07-11 isolated real-GPU verification

Command:

```powershell
python scripts/verify_worker_pool.py --port 18123 --timeout-seconds 240
```

The script starts an isolated FastAPI service, seeds two real PNG source tasks, and
starts two independent worker processes with concurrency one. Both workers run the
installed Real-ESRGAN executable with `tile=0`, upload their results through the
normal internal API, and are terminated after verification.

| Measurement | Result |
| --- | ---: |
| Independent worker processes | 2 |
| Tasks | 2 |
| Distinct workers that completed tasks | 2 |
| Wall time | 9.229s |
| Duplicate claims | 0 |
| Lost tasks | 0 |
| Attempts per task | 1, 1 |
| Pillow output verification | both 2048×2048 PNG |

The complete automated suite after this stage passed with `57 passed` before the
stage documentation update. The verification output and generated images are under
the ignored `remote-worker-data/elastic-stage1/` directory.
