# V15 — network-aware short-prompt scheduling

Protocol written before collecting results, 2026-09-23.

## Question

Can end-to-end calibration and a minimum predicted-gain threshold remove the 4k regression seen when a Mac worker is
added to the Arc B580, without preventing useful contribution on longer prompts?

## Cause under test

V13 planned from worker-reported compute time. That excludes serialization, network round trips and coordinator delay.
At 4k the planner assigned only 43–44 core tokens to the Mac, but the Mac still paid for a 256-token halo and one LAN
request. The hypothesis is that this fixed cost is larger than the GPU work removed.

## Conditions

- Qwen2.5-0.5B-Instruct F16, same native workers and span mode as V13.
- Coordinator and Arc B580 worker on the Ryzen 5600X; Mac M4 worker over the LAN.
- Prompt lengths: 4,096, 8,192 and 16,384 tokens; five timed repetitions after warm-up.
- Compare B580 alone, the former compute-only plan, and the candidate end-to-end plan.
- Candidate calibration runs workers concurrently and includes total send-to-response time.
- Candidate initially uses a 1% minimum predicted improvement; otherwise the fastest worker runs alone.

## Pre-registered criteria

- **S1:** at 4k, candidate throughput is at least 0.99× B580 alone and the Mac receives zero core tokens.
- **S2:** candidate is faster than the former plan at 4k.
- **S3:** at 16k, the Mac is retained only if predicted improvement is at least 1%.
- All allocations, end-to-end rates and raw timings are saved; failures remain recorded.

## Adendum 1 — safety margin after run 1

Run 1 passed S1/S2 at 4k, but exposed the same smaller effect at 8k: the model predicted about +3.1% and measured
−0.57%. At 16k it predicted about +4.6% and measured +1.88%. The 1,024-token probe underestimates the B580's efficiency
on a long contiguous span, leaving a 2–3 percentage-point optimism error. Before run 2, the default minimum gain is
therefore raised from 1% to 4%. This should select GPU-only at 4k/8k and retain the Mac at 16k. Run 1 remains saved as
`results_run1.json`; run 2 will be judged separately against the original S1–S3 plus the new 8k no-regression check.
