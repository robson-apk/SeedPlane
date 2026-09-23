# Changelog

All notable changes are documented here. SeedPlane is currently alpha research software.

## 0.12.3 — 2026-09-23

- Added gain-aware request-level batching across heterogeneous workers.
- Added a full-job tail guard so slow devices contribute only above their measured break-even queue depth.
- Validated B580 + Mac on 18 independent 4k requests: +5.79% and +5.18% aggregate throughput in two runs.
- Added the pre-registered V16 protocol, raw timelines and results.

## 0.12.2 — 2026-09-23

- Added concurrent end-to-end worker calibration including network and contention.
- Added a configurable 4% scheduling safety margin to prevent short-prompt regressions.
- Validated the new policy on Arc B580 + Mac M4: no regression at 4k/8k and +1.84% at 16k.
- Added the pre-registered V15 protocol, raw timings and results.

## 0.12.1 — 2026-09-23

- Changed Python and native workers to loopback-only defaults; external binds now require an explicit authentication key.
- Added strict native-protocol bounds validation and client response validation.
- Added package, planner, security and protocol tests plus continuous integration.
- Replaced the obsolete Hadamard quickstart demo with the current shard-window layout.
- Fixed dependency metadata, legacy-pip wheel builds and machine-specific experiment paths.
- Clarified the README's causal prefill/scoring scope, quality trade-offs and historical diffusion track.

## 0.12.0 — 2026-09-23

- Added native llama.cpp workers, device probing, dynamic scheduling and heterogeneous pipeline planning.
- Published V13/V14 speed, quality, KV-cache and device-planning experiments.
