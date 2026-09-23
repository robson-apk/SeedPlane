# V16 — all-device gain through request-level parallelism

Protocol written before collecting results, 2026-09-23.

## Problem

A slow device cannot be guaranteed to reduce the latency of one request served by a much faster GPU: synchronization,
halo duplication and network latency can exceed its contribution. The physically achievable interpretation of “all
devices work and the system gains” is saturated service throughput: independent requests are routed concurrently, so a
slow device adds capacity without blocking the B580's request.

## Test

- Qwen2.5-0.5B-Instruct F16, 4,096-token span prefill.
- Coordinator and B580 on the Ryzen 5600X; Mac M4 worker over LAN.
- Start at 17 independent requests after warm-up, near the previously measured
  B580:Mac throughput ratio. If full-job measurement shows this is below the
  physical break-even point, record the failure and retest at the first integer
  batch size above that measured threshold.
- Baseline: all requests sequentially on the B580.
- Candidate: shared request queue with a tail guard; each worker pulls a request only if it should finish before the
  remaining queue on the other workers. Report total requests/s, aggregate tokens/s, wall time and jobs per worker.

## Criteria

- **A1:** both B580 and Mac complete at least one request.
- **A2:** aggregate candidate throughput is greater than B580-only throughput.
- **A3:** every result is present in original request order.

This experiment targets service throughput, not the latency of a single response and not autoregressive generation.
