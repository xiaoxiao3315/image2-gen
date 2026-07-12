# Elastic load and stability validation

This report records the real stage-3 acceptance of the elastic image service. It
contains no provider address, public server address, API key, bearer token or SSH
credential.

## Scope and method

The cloud generation pool was temporarily configured with minimum 1 and maximum 5
workers; the code hard limit remained 8. The local GPU worker kept three concurrent
upscale slots. All customer traffic entered through LiteLLM with one temporary
virtual key restricted to `image-gen`. The key was revoked after the test.

Two phases used unique idempotency keys and `n=1`, so this test measured concurrent
tasks rather than native provider batching:

1. Burst: eight requests started together.
2. Sustained: six requests ran with three continuously replenished in-flight slots.

Generation POSTs were not retried by the load client. A connection failure or HTTP
error would therefore count as a failure instead of being hidden. Read-only status
and image downloads used bounded GET retries. After generation, all output URLs were
downloaded with concurrency three and decoded with Pillow.

The monitor sampled `/v1/stats`, the Python systemd cgroup, the LiteLLM container
cgroup and host available memory. NVML was unavailable in the Codex process, so GPU
memory was separately measured with Windows `GPU Adapter Memory` performance
counters during three concurrent, real `tile=0` 4K upscales of the same source.

## Results

| Measurement | Burst | Sustained |
| --- | ---: | ---: |
| Requests | 8 | 6 |
| Success | 8 | 6 |
| Failure | 0 | 0 |
| Wall time | 144.311s | 191.073s |
| Window completion rate | 199.569 images/hour | 113.046 images/hour |
| Request latency range | 65.861–144.307s | 76.676–97.814s |

The window completion rate is a short-run observed rate, not a promise of day-long
capacity. The sustained three-in-flight phase is the better conservative estimate
for the restored production default of three generation workers: approximately 113
delivered 2K images/hour in this run.

Across both phases:

- 14/14 tasks reached `done`; SQLite reported no failed task.
- 14/14 public downloads were valid PNG files and Pillow decoded every file as
  exactly 2048×2048.
- Total downloaded bytes were 86,191,799 (about 82.2 MiB); individual files ranged
  from 5,304,548 to 6,735,968 bytes.
- Average task creation-to-completion time was 102.030s; maximum was 141.812s.
- All generation modes were `single`.
- LiteLLM SpendLogs increased by 14 rows and 0.14 USD, matching the configured
  per-image test price of 0.01 USD.

## Elasticity evidence

241 authenticated stats samples recorded:

| Signal | Observed peak/final |
| --- | ---: |
| Generation workers active | peak 5, final 1 |
| Generation workers busy | peak 5 |
| Global generation slots in use | peak 5 of hard limit 8 |
| Queue length | peak 3 |
| Tasks awaiting upscale | peak 2 |
| Tasks upscaling | peak 3 |
| Busy GPU worker slots | peak 3 |

This demonstrates both scale-out on backlog and scale-in after the idle retirement
window. The service was restored to the conservative production configuration
`min=1`, `max=3` after the acceptance run.

## Memory and process stability

| Measurement | Result | Limit/headroom |
| --- | ---: | ---: |
| Python backend peak | 285.883 MiB | 500 MiB hard limit |
| LiteLLM peak | 1160.184 MiB | 1600 MiB container limit |
| Host minimum available memory | 1013.473 MiB | remained positive |
| Backend restarts | 0 | active after test |
| LiteLLM restarts/OOM | 0 / false | healthy after test |

The system did not hit a cloud memory limit and did not crash under either workload.

## GPU `tile=0` concurrency probe

Three independent 4K upscales ran concurrently with `REALESRGAN_TILE_SIZE=0`:

| Measurement | Result |
| --- | ---: |
| Processes / successful exits | 3 / 3 |
| Wall time | 31.767s |
| Verified outputs | 3 × 3840×2160 PNG |
| Idle dedicated GPU memory | 1731.242 MiB |
| Peak dedicated GPU memory | 5691.648 MiB |
| Workload increase | 3960.406 MiB |

Three seamless whole-image upscales are therefore well within the 16GB GPU memory
capacity. This probe establishes memory safety at concurrency three; it does not by
itself prove linear speedup at four or five because GPU compute and home uplink share
the same machine.

## Bottleneck and operating recommendation

At generation max five, all five generation slots and all three GPU slots became
busy, with two images briefly waiting for upscale. The immediate burst bottleneck at
this configuration was therefore the three-slot local GPU/upload stage, not cloud
RAM. GPU memory has ample headroom, so a controlled uplink test at GPU concurrency
four is the next sensible expansion step.

After that expansion, the upstream image channel is expected to become the strategic
ceiling: prior measurements show errors beginning above roughly eight concurrent
generation calls, while the service hard limit is already eight. The current safe
operating recommendation is:

- default: generation `min=1/max=3`, GPU concurrency 3;
- temporary burst mode: generation max 5 only while monitoring stats and host memory;
- next scale test: GPU concurrency 4 with upload-bandwidth monitoring;
- do not raise the generation hard limit above 8 without a new provider failure-rate
  test.

The measured safe default is about 113 2K images/hour in this short sustained run.
Claims near the earlier theoretical 700 images/hour are not supported by this
end-to-end deployment and should not be used for customer commitments.
