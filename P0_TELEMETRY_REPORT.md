# P0 Production Telemetry Report

Date: 2026-07-13
Baseline: `master@be9ab7b88236ebbbd86184b42adfad37e25a2647`
Scope: additive, best-effort SQLite facts for reconstructing generation and task-stage failures, plus one explicitly authorized restart-recovery logic correction in `TaskStore.__init__`.

## Result

Implemented the P0 production fact layer without changing the existing `tasks` table, retry counts, retry backoff, generation limits, worker limits, lease rules, queue API, web UI, or deployment configuration. This is no longer described as a pure observation-only P0: it includes one authorized restart-recovery correction in `TaskStore.__init__`, ensuring every task actually reset from `running`/`processing` to `queued` receives exactly one `generation_queued(reason=restart_recovery)` fact. The correction does not change which tasks are recovered or any retry/concurrency/lease/API-response behavior.

No real image API was called. No production service was started. Disposable local FastAPI instances and loopback fake providers were used only for acceptance. No deployment was run. P0/P0.1 were committed only to `feat/p0-production-telemetry`; the default branch was not modified or merged.

## Scope separation: P0, authorized recovery correction, and P0.1

- **P0 telemetry** is the additive fact layer described in this report.
- **Authorized restart-recovery correction** is one deliberate business-implementation change in `TaskStore.__init__`: after the authoritative recovery transaction identifies tasks reset from `running`/`processing` to `queued`, each identified task gets one best-effort `generation_queued(reason=restart_recovery)` fact regardless of whether its old open attempt was reconciled. A second initialization does not duplicate the event because the task is already `queued`; an originally queued task is not mislabeled as restart-recovered.
- **P0.1 error-classification correction** is a separate committed increment on `feat/p0-production-telemetry`. `normalize_error` inspects the current exception first and, only if it is unknown, follows explicit `__cause__` for at most two hops with cycle protection. It never reads implicit `__context__`.
- Current acceptance evidence uses Hermes Python 3.11.15 / pytest 9.1.1 with inherited service-authentication variables removed: restart recovery target `1 passed in 1.12s`; exact four-worker concurrency target `1 passed in 2.73s`; final complete suite `106 passed in 25.80s`; AST `ast ok`.
- Earlier `101`, `102`, and `105`-test totals are retained below only as historical audit evidence, not current acceptance evidence.

## Changed files

- `image_pipeline/telemetry.py` — new additive schema, redaction, append APIs, recovery handling, task timeline, and bounded window statistics.
- `image_pipeline/generator.py` — optional attempt observer around the physical generation POST loop.
- `image_pipeline/service.py` — post-commit lifecycle facts and compatible observer attachment only for concrete `GptImageGenerator` instances.
- `cli.py` — read-only `telemetry-task` and `telemetry-stats` commands.
- `scripts/cleanup_service_data.py` — telemetry retention follows terminal task retention.
- `scripts/benchmark_telemetry.py` — local SQLite write overhead/concurrency measurement.
- `tests/test_telemetry.py` — schema, contention, fault-injection, lifecycle, recovery, query, cleanup, concurrency, and security tests.
- `tests/test_cleanup_service_data.py` — telemetry cleanup coverage.

## Schema and semantics

### `task_events`

Append-only lifecycle facts with:

- stable `event_id`;
- sanitized event type;
- UTC epoch timestamp;
- optional duration and attempt number;
- irreversible worker hash;
- event-specific, value-allowlisted `details_json`; unsupported details are discarded on write and hidden from timeline output.

Observed lifecycle includes:

`accepted → generation_queued → generation_started → generation_completed → upscale_queued → upscale_started → upscale_finished → delivery_completed`

Terminal failures use `terminal_failed`. Restarted generation work receives another `generation_queued` event with `reason=restart_recovery`; open physical attempts attributed to recovered tasks are closed as `interrupted`.

### `generation_attempts`

One row equals one physical upstream generation `POST`.

- Native `n>1` remains one physical attempt with `requested_n` and a list of attributed task IDs.
- A 503 retry or transport retry creates another physical attempt row.
- Fallback `n=1` calls create their own physical attempt rows.
- Download `GET` does not create a generation attempt. A returned-image download failure classifies the associated POST as `image_download_failure`.
- Attempt rows carry no prompt, request/response body, endpoint, headers, credential, idempotency key, image bytes, or returned URL.
- `provider_request_id` stores only an irreversible SHA-256 hash of the provider-controlled request-ID header, never the raw value.

Error taxonomy:

- `connect_timeout`
- `read_timeout`
- `connection_error`
- `remote_disconnect`
- `http_503`
- `http_429`
- `http_4xx`
- `http_5xx`
- `invalid_response`
- `image_download_failure`
- `unknown`

## Mock fault-injection results

All provider interactions were fake in-memory sessions.

1. **HTTP 503 → success**: two physical attempts, stable existing business retry/backoff behavior, first row `http_503`/retry, second row success.
2. **ReadTimeout → success**: first row classified `read_timeout`, second physical attempt succeeds.
3. **Three transport failures**: exactly three physical attempts and the original terminal `ImageGenerationError` behavior.
4. **Malformed HTTP 200**: one physical attempt classified `invalid_response`; a hash of the provider request ID is retained without raw header/body text.
5. **Returned image download failure**: one generation POST attempt classified `image_download_failure`; no fake second generation attempt for the GET.

Additional checks cover native `n=3` cardinality, observer exceptions, absent-observer compatibility, restart recovery, slow synthetic upscale timing, cleanup, and eight concurrent appenders.

## Read-only queries

```bash
python cli.py telemetry-task --database <tasks.db> --task-id <task-id>
python cli.py telemetry-stats --database <tasks.db> --since 2026-07-13T00:00:00Z
python cli.py telemetry-stats --database <tasks.db> --since 2026-07-13T00:00:00Z --until 2026-07-14T00:00:00Z
```

`telemetry-task` excludes `prompt`, task errors, filesystem paths, credentials, headers, URLs, and response data. It returns safe task metadata, ordered events/attempts, and integrity diagnostics.

`telemetry-stats` uses the reproducible half-open interval `[since, until)` and nearest-rank P50/P95/P99. It reports requests accepted in the window, deliveries/failures completed in the window, rates, delivered/hour, oldest queued age, failure-category counts, and latency distributions for generation queue, physical attempts, retry backoff, upscale queue, upscale execution, delivery overhead, and end-to-end.

Synthetic CLI smoke test:

```text
telemetry-task=ok secret_safe=true
telemetry-stats=ok requests=1
```

## SQLite contention and overhead

Command:

```bash
python scripts/benchmark_telemetry.py
```

Measured on the current Windows host, 250 rounds, eight concurrent appenders:

```json
{
  "baseline_median_ms": 14.2194,
  "observed_median_ms": 30.0715,
  "median_overhead_ms": 15.8521,
  "concurrent_success": 8,
  "concurrent_seconds": 0.1803,
  "telemetry_failures": 0
}
```

A separate lock-contention test holds `BEGIN IMMEDIATE`; the telemetry call returns false in under 0.5 seconds and the business row remains valid. This is evidence from this host, not a universal latency guarantee.

## P0.1 cause-only correction evidence

The later error-classification correction is not part of the pure P0 telemetry scope. It is tracked as P0.1 and is committed separately on `feat/p0-production-telemetry`.

- Classification first uses the current exception type.
- If that is unknown, it follows only explicit `__cause__`, never implicit `__context__`.
- Cause traversal is capped at two hops and tracks object identities to stop cycles.
- A regression test constructs an `ImageGenerationError` whose only implicit `__context__` is an unrelated `ReadTimeout`; the expected category remains `unknown`. Reintroducing `__context__` traversal would change it to `read_timeout` and fail the test.
- Additional tests cover one-hop explicit cause, two-hop explicit cause, rejection of a third hop, and a cause cycle.
- Focused P0.1 telemetry result before the authorized recovery correction: `27 passed in 7.13s`.
- Current complete P0 + authorized recovery correction + P0.1 result: `106 passed in 25.80s`.
## Implementation hardening before P0.1

An independent review reproduced six edge cases that the ordinary suite did not initially cover. They were addressed during the uncommitted implementation:

- open attempt rows left by a transient finish-write lock are reconciled after restart even when the task already advanced to `awaiting_upscale`;
- restart business recovery commits independently of telemetry reconciliation, so malformed/busy telemetry cannot prevent task requeue;
- physical attempts spanning a window boundary are attributed exactly once by `finished_at`;
- cleanup tolerates partial/legacy telemetry schemas;
- cleanup commits each successfully unlinked artifact reference immediately; if a later artifact unlink fails, the retained task remains discoverable without pointing at a file already deleted;
- historical backlog output no longer reconstructs pre-restart state from a cleared mutable `started_at`; it reports the current backlog at the query boundary, while event/attempt latency windows remain reproducible from append-only facts.

- **P0.1 terminal lifecycle classification** follows only the explicit wrapped transport cause, so exhausted `ReadTimeout` retries emit `terminal_failed.details.category=read_timeout` instead of `unknown` without consulting implicit context.
- **Authorized restart recovery correction** removes the incorrect coupling between attempt reconciliation and recovery queue facts. The reconciled-task set means “an open attempt was closed,” not “the new queue transition was already recorded”; every task captured by the authoritative restart recovery transaction now receives one recovery queue fact.
- **Concurrency test integrity** was restored by reverting the uncommitted relaxation `1 < peak <= 4` to the original `peak == 4`. `git blame` attributes the original exact assertion to `7425030386b4cca968de7a4aaaa8c56d77a2d14e` (`feat: add LiteLLM gateway and elastic generation`, 2026-07-12); the relaxation had no commit or documented design rationale. The exact test passes without changing concurrency implementation.

Historical pre-correction evidence retained for audit:

- focused implementation tests: `20 passed in 2.62s`;
- service/concurrency/worker regression set: `44 passed in 19.74s`;
- earlier mixed P0 + unsafe cause/context correction suite: `102 passed in 30.44s`.

That earlier `102 passed` run is not current acceptance evidence because its classifier still consulted implicit `__context__`. The later `105 passed` run also ceased to be valid after test-integrity review exposed a missing restart-recovery assertion; adding the real assertion produced `1 failed, 105 passed in 36.21s` and revealed the authorized recovery defect. After the scoped correction and restoration of the exact concurrency assertion, current evidence is `106 passed in 25.80s`.

Additional checks:

- Final complete suite with the authorized recovery correction, exact concurrency assertion, P0.1 cause-only classification, and P0.2 details contract: `147 passed in 39.12s`.
- Complete telemetry suite including final hostile-object, details-contract, and legacy-projection coverage: `69 passed in 13.81s`.
- Focused P0.2 details contract matrix after adversarial review: `42 passed, 27 deselected in 12.32s`.
- Restart recovery target, including exactly-once and no-mislabel assertions: `1 passed in 1.12s`.
- Exact four-worker concurrency target: `1 passed in 2.73s`.
- `tests/test_telemetry.py` AST parse: `ast ok`.
- `git diff --check` — passed; only Git's existing Windows LF/CRLF conversion warnings were printed.
- CLI synthetic database smoke test — passed; prompt text absent.
- Changed-file sensitive-pattern scan — no real credential, channel address, raw provider response, network address, or token value found. Matches were existing production identifiers/code (`Authorization`, proxy handling, URL recognition), redaction patterns, and synthetic `.invalid` test URLs only.

## Explicit non-goals

Not implemented:

- task-level durable generation requeue;
- retry-count/backoff changes;
- higher generation or GPU worker concurrency;
- ETA or web UI changes;
- periodic expired-lease scanning;
- LiteLLM financial reconciliation;
- deployment or production migration execution.

## Acceptance criteria 1–10

1. **Additive SQLite facts without changing `tasks`** — met.
2. **Every physical upstream POST is reconstructable** — met for concrete `GptImageGenerator`; native batches are not multiplied into fake per-task attempts.
3. **Requested error categories and retry/backoff facts** — met and mock-tested.
4. **Task lifecycle and stage timing reconstruction** — met, including generation queue, upscale queue, synthetic 110-second upscale execution, delivery, and end-to-end.
5. **Restart recovery facts** — met; open attempts close as interrupted and recovered tasks get a new queue event.
6. **Telemetry never fails business behavior** — met by non-throwing store/observer boundaries, short SQLite timeout, lock test, and fake-generator compatibility.
7. **Read-only reproducible task/window queries** — met through CLI and module queries with deterministic tie-breakers.
8. **Retention cleanup** — met for direct task rows and batch-attributed native attempts; old databases without telemetry tables remain compatible.
9. **No known sensitive payload persistence/output in supported details contracts** — current producers, event-specific detail key/value projection, provider request-ID hashing, safe summaries, synthetic security tests, CLI smoke tests, and changed-file scans exclude the known sensitive fields tested. Unsupported details are discarded on write and hidden when legacy rows are read through `task_timeline`; existing raw legacy rows are not rewritten.
10. **No unauthorized behavior/deploy/real-call/merge overreach** — implementation evidence is present and the complete suite passes. The scope explicitly includes one authorized restart-recovery correction in `TaskStore.__init__`; no retry/concurrency/worker-limit/lease/API-response behavior was changed. P0/P0.1 were committed and pushed only to `feat/p0-production-telemetry`; no deployment, real provider call, default-branch merge, or direct default-branch push occurred.

## P0.2 details contract closure

The benign-key sensitive-value gap in `task_events.details_json` is closed by an event-specific, closed key/value contract:

- `generation_queued`: `reason=restart_recovery`;
- `generation_completed`: `mode` in `single`, `single_fallback`, `native_n`, `native_n_partial`;
- `upscale_queued`: `reason` in `lease_expired`, `worker_release`;
- `upscale_finished`: `outcome` in `success`, `retry`, `failed`, `lease_expired`;
- `terminal_failed`: `stage=generation` with a declared error category, or `stage=upscale` with `category=worker_failure`.

All other details are discarded. Unknown event types still persist, but without details; event-type vocabulary enforcement remains deferred. The same projection runs before new writes and when `task_timeline` reads legacy rows. Valid legacy fields remain visible, unsupported or sensitive extras are hidden, and the original SQLite rows are not rewritten. No schema migration is required.

Adversarial coverage includes benign keys carrying prompt/token/URL text, known keys with invalid or unhashable values, cross-event keys, punctuation aliases that sanitize to known event names, unknown events, malformed/non-object legacy JSON, and read-time verification that legacy raw rows remain unchanged.

## P0.3 terminal HTTP classification closure

Terminal task classification now carries the final physical request's structured HTTP status through the trusted `ImageGenerationError` family:

- exhausted HTTP 503 classifies as `http_503` at both the physical attempt and task terminal event;
- non-retried HTTP 429, other 4xx, and 5xx classify as `http_429`, `http_4xx`, and `http_5xx` respectively;
- explicit native-batch unsupported errors retain status/phase while preserving existing fallback behavior;
- retry count, backoff, `will_retry`, idempotency, native fallback, concurrency, lease, and API behavior are unchanged;
- exceptions outside the `ImageGenerationError` contract cannot spoof status/phase metadata or trigger arbitrary attribute descriptors; they keep existing fallback classification.

Review found one trusted-boundary issue in the initial implementation: `_fail_task` used generic `getattr` on any exception. The final implementation accepts structured metadata only from `ImageGenerationError`, with regression coverage for misleading and hostile exception attributes.

Final local acceptance after review: focused P0.3 `13 passed in 3.50s`; P0.2 + P0.3 regression `82 passed in 19.17s`; complete suite `160 passed in 37.44s`; changed Python AST parse `ast ok`; `git diff --check` passed with Windows LF/CRLF warnings only; targeted changed-file secret-pattern scan returned no matches. No real provider call, service deployment, retry-policy change, default-branch merge, or direct default-branch push occurred.

## Deferred optimization findings

The following reviewed findings remain separate follow-up work:

- read-only task timelines can fail against a partial telemetry schema;
- attempt-number allocation and some statistics paths scan retained history;
- generation queue latency is anchored to task creation rather than the latest queue episode;
- `ProxyError` / `SSLError` can fall through to `unknown`;
- lifecycle event types are sanitized but not enforced against a vocabulary.

These are documented findings, not accepted behavior or completed fixes.

## Known limitations

- Telemetry is best-effort by design: a busy/locked SQLite database can drop a fact while preserving the business transition. `TelemetryStore.status()` exposes the in-process failure count and last exception type for diagnostics, but this P0 does not add a monitoring endpoint.
- A process may crash after a business commit and before its post-commit telemetry append. Integrity diagnostics flag missing lifecycle facts; the task ledger remains authoritative.
- Physical native-attempt attribution uses `task_ids_json` because one request can serve multiple logical tasks. The existing `tasks` table is unchanged.
- Window statistics use two explicit cohorts: `requests` means accepted in `[since, until)`, while delivered/failed rates and throughput mean terminal completions in `[since, until)`. This avoids retroactively changing a closed historical window.
