# V15 — network-aware short-prompt scheduling

Qwen2.5-0.5B-Instruct F16 · Arc B580 on the Ryzen 5600X · Mac M4 over LAN · span mode · five timed repetitions.
Raw results: `results_run1.json` and `results_run2.json`.

## Root cause

The V13 planner used worker-reported compute time, excluding serialization, network round trips and coordinator delay.
It also accepted predicted gains smaller than normal calibration error. At 4k this produced a split with only 17–44
core tokens on the Mac. Creating that split adds a 256-token boundary halo to the B580's span plus a LAN request, so
more total work and fixed latency were introduced than the tiny Mac piece removed.

Run 1 changed calibration to concurrent end-to-end time and used a 1% gain threshold. It fixed 4k, but 8k exposed a
2–3 percentage-point optimism error because the B580 becomes more efficient on spans longer than the 1,024-token probe.
The pre-recorded adendum raised the safety margin to 4% for run 2.

## Run 2 result

| Prompt | B580 alone | Former compute-only plan | End-to-end + 4% margin | Mac core tokens | Candidate / B580 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 19,487 tok/s | 18,083 tok/s | **19,461 tok/s** | 0 | **0.999×** |
| 8,192 | 19,384 tok/s | 19,316 tok/s | **19,404 tok/s** | 0 | **1.001×** |
| 16,384 | 19,374 tok/s | **19,777 tok/s** | 19,730 tok/s | 756 | **1.018×** |

The candidate eliminated the short-prompt regressions and retained a measured **+1.84%** gain at 16k. The 16k
compute-only and end-to-end plans differ by only 23 tokens; their 0.24% throughput difference is within run-to-run
variation and does not justify preferring compute-only calibration.

## Verdict

- **S1 passed:** 4k is 0.999× B580 alone and the Mac receives zero tokens.
- **S2 passed:** candidate is 1.076× the former plan at 4k in run 2.
- **S3 passed:** at 16k the predicted gain cleared 4%, the Mac was retained, and measured throughput improved 1.84%.
- **Adendum 8k check passed:** the Mac was excluded and throughput was 1.001× B580 alone.

This optimizes prompt prefill/scoring. It does not make autoregressive token generation faster across the Mac and B580.
