# Changelog

All notable changes are documented here. SeedPlane is currently alpha research software.

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
