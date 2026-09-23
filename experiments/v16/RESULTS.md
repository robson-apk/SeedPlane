# V16 — all-device gains through request-level parallelism

Qwen2.5-0.5B-Instruct F16 · Arc B580 on the Ryzen 5600X · Mac M4 over LAN · 4,096-token span prefill.
Raw successful runs: `results_18_run1.json` and `results_18_run2.json`.

## Result

The scheduler now distributes independent requests instead of forcing the Mac into the critical path of one short
request. It calibrates the complete job size and admits a slower worker only when its predicted completion is earlier
than draining the remaining queue on the other devices.

| Queue | B580 only | B580 + Mac | Assignment | Speed-up |
|---:|---:|---:|---|---:|
| 17 (initial boundary test) | 3.573 s | 3.635 s | 16 + 1 | **0.983×** |
| 18, run 1 | 3.783 s | 3.576 s | 17 + 1 | **1.058×** |
| 18, run 2 | 3.783 s | 3.597 s | 17 + 1 | **1.052×** |

All 18 results were returned in their original order in both successful runs. Aggregate throughput rose from 19,489
tok/s on the B580 alone to 20,616 and 20,498 tok/s with both devices.

## What the failed boundary taught us

The original 1,024-token probe underestimated the Mac's complete 4k service time. At 17 requests the Mac finished one
job 62 ms after the B580 finished the other 16, causing a 1.7% regression. Full-size calibration puts the physical
break-even point at 18 queued requests. Below that point, the gain-preserving policy leaves the Mac idle; at or above
it, both devices contribute.

## Verdict

- **A1 passed at the measured break-even:** B580 completed 17 jobs and Mac completed one.
- **A2 passed twice:** aggregate gains were **5.79%** and **5.18%**.
- **A3 passed twice:** 18/18 ordered results were present.

No scheduler can guarantee both non-idling and a speed gain when there is less independent work than the slower
device's break-even threshold. SeedPlane therefore guarantees the useful property: every admitted device has a
measured opportunity to reduce makespan; devices that would delay completion wait for more work.
