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

## Stage 2: LiteLLM gateway and elastic generation workers

Stage 2 separates the existing leader web API from the developer API. The public
developer model is `image-gen`; the private pipeline group is never included in a
customer response or customer model list. LiteLLM owns virtual keys, RPM/budget
enforcement and spend persistence in PostgreSQL. The image task ledger remains in
its own SQLite database.

Generation worker behavior is now backlog-driven: one minimum worker stays warm,
queued batches scale the pool to the configured ceiling, and surplus workers retire
after an idle interval. A batch is still limited to five images, while the global
logical generation hard limit is eight. Upstream idempotency keys are generated at
task creation and stored in SQLite so a process restart cannot silently turn a
retry into a new billable request.

### 2026-07-12 digest-pinned container integration

The integration used the real LiteLLM Proxy 1.91.2 database image and PostgreSQL
16.14 image, both addressed by immutable digest. A local private mock implemented
only the OpenAI Images transport; it did not bypass LiteLLM authentication, routing,
hook loading, rate limiting or spend logging.

| Check | Result |
| --- | --- |
| LiteLLM liveness | HTTP 200 |
| Two independent virtual keys | created without persisting or printing key values |
| Customer `GET /v1/models` | both keys saw only `image-gen` |
| Direct private group request | HTTP 403 for both keys |
| Real custom hook | rewrote alias to private group |
| Tenant scope | opaque and distinct for the two virtual keys |
| `Idempotency-Key` | forwarded unchanged to the private backend |
| RPM limit 1 | first image request 200, immediate second request 429 |
| Batch response | `n=1` returned one URL; `n=5` returned five URLs |
| PostgreSQL SpendLogs | success rows persisted |
| Configured price | `n=1` recorded 0.01 USD; `n=5` recorded 0.05 USD |

Two version-specific findings were fixed during this integration:

1. LiteLLM resolves a callback relative to the config directory, so `gateway/` must
   be mounted at `/app/gateway`; `PYTHONPATH` alone is insufficient.
2. Image price and `mode: image_generation` belong in `model_info`, and
   `model_info.id` must match the private model group. Putting the price only in
   `litellm_params`, or using a different ID, produces successful zero-cost rows.

The local first-start Prisma migration used approximately 700–750 MiB, but the
target cloud host required more headroom: both 900 MiB and 1200 MiB cgroup limits
were killed during migration. The production LiteLLM limit is therefore 1600 MiB
with a 768 MiB reservation. PostgreSQL retains a 256 MiB limit. The Python image
service keeps its independent systemd `MemoryHigh=450M` and `MemoryMax=500M`
limits.

LiteLLM Community Edition authenticates passthrough endpoints for the master key,
but granting a virtual key `allowed_passthrough_routes` is an Enterprise feature.
`GET /v1/stats` therefore uses Nginx `auth_request` against LiteLLM `/v1/models`,
then injects the private backend token from a root-owned Nginx snippet. This keeps
virtual-key validation in LiteLLM without recreating tenant authentication in the
application.

### 2026-07-12 cloud and real-GPU acceptance

The gateway, PostgreSQL and updated backend were deployed behind Nginx. Ports
8012, 4000 and 55432 were verified as loopback-only. Nginx configuration validation
and reload succeeded without printing the private authorization snippet.

Two temporary virtual keys were created through the loopback-only master-key API.
The key values were never printed or persisted in the repository and were revoked
after the test.

| Check | Result |
| --- | --- |
| Both customer model lists | only `image-gen` |
| Both authenticated `GET /v1/stats` calls | HTTP 200 |
| Direct private group request | HTTP 403 |
| Invalid virtual key | HTTP 401 |
| Concurrent developer requests | `n=1` HTTP 200 in 49.071s; `n=2` HTTP 200 in 129.364s |
| Delivered files | three downloadable PNG files |
| Pillow verification | all three exactly 2048×2048 |
| File sizes | 3,213,202; 6,325,427; 6,208,234 bytes |
| Shared idempotency key across two tenants | two ledger rows, two distinct opaque tenant scopes |
| SpendLogs delta | two rows; 0.03 USD total (`0.01 + 0.02`) |
| Acceptance harness wall time | 204.293s including three sequential public downloads and final diagnostics |
| Python backend memory peak | 143.324 MiB under the 500 MiB hard limit |
| LiteLLM memory peak | 1197.477 MiB under the 1600 MiB limit |
| Host minimum available memory | 1075.164 MiB |
| Process health | backend 0 restarts; LiteLLM healthy, no OOM, 0 restarts |

The real provider silently ignored `n=2`: it returned HTTP 200 with one image
instead of explicitly rejecting the parameter. The generator now preserves that
already-paid partial result and requests only the missing image with its own
persisted upstream idempotency key. The process remembers the capability result,
so later batches skip the native probe. The successful request recorded generation
modes `native_n_partial` and `single_fallback`; the concurrent one-image request
recorded `single`.

Final deployment review added the following fail-closed boundaries:

- Public environment templates contain empty secret values, not usable known
  placeholders; production also sets `IMAGE_REQUIRE_SERVICE_AUTH=true`.
- Placeholder-shaped tokens are rejected by the application and never make the
  health readiness flags true.
- Gateway reload forces container recreation so imported hook/config changes cannot
  remain stale; gateway stop grace is 930 seconds.
- Backend generation drain is 990 seconds with a 1020-second systemd window, covering
  the bounded three-attempt provider worst case.
- Backend, LiteLLM and Nginx synchronous deadlines are nested at 900/910/930 seconds.
- Retention deletes complete terminal batches atomically. An already-incomplete
  idempotency ledger is rejected instead of replaying fewer images or regenerating.
- Linux integration maps `host.docker.internal` through Docker `host-gateway`.

The full Python suite after this hardening passed with `76 passed`; `compileall` and
`git diff --check` also passed. This completes the technical stage-2 rollout. The
current IP-over-HTTP endpoint remains for internal acceptance only: a trusted domain,
TLS certificate and forced HTTPS redirect are still required before issuing
production customer keys.
