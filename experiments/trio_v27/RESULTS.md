# V27-T3 — B580 + RX 570 + M4 results

**Verdict: exploratory result; the pre-registered throughput and latency gates were not met.**

The complete three-device matrix ran on 2026-09-24. The exact protocol is in
[`PROTOCOL.md`](PROTOCOL.md), the runner in [`benchmark_trio.py`](benchmark_trio.py), and all per-request measurements
in [`results_20260924_exploratory.json`](results_20260924_exploratory.json).
The exact worker commands, identities, network notes and invocation are preserved in
[`run_20260924_exploratory.sh`](run_20260924_exploratory.sh); pass a new output filename to replay it after restaging
the three runtime/model paths.

## What was tested

- Seven configurations: each device alone, all three pairs, and the trio.
- Two arrival patterns per configuration: 24 requests at once, and 24 requests spaced one second apart.
- Three rounds, with rotated condition order: **42 measured workloads**, **1,008 requests**, and **129,024 generated tokens**.
- 128 greedy tokens/request, identical prompt, same complete model bundle and plan on each host. The runtime reported
  `shadow-batch`, shard 512, halo 256, and four sinks on all three devices.
- B580: Intel Arc B580 / Vulkan on Windows. RX 570: Radeon RX 570 / RADV on Linux. M4: Apple M4 / MoltenVK on macOS.
- Runtime SHA-256 values: B580 `c0fcb93563c715947f54eb6f7b851106918187a3897985e0fe4429d8318c1e3a`; RX 570
  `6b89b2e14e355186e04d3064da1909b5996b75bfef93c4d98c0212a04772be0b`; M4
  `f0bccaff7ea2e79ad11d17a56db4f178db67505dea8595b3fc3992bd35738aac`. The bundle, weights, tokenizer and plan
  hashes matched across all three hosts; see the raw JSON for the complete identities and ready reports.
- Harness/source commit: `d69a328395e250c93f86a09bf8d74bac8368b7e9`. The native executables came from the main CI build
  artifacts; their hashes are recorded above.

## Throughput and latency

Values below are medians over the three rounds. Arrival-to-completion latency includes queue wait; p99 is the maximum
of 24 observations under the pre-registered nearest-rank method.

| Arrival pattern | Devices | Aggregate tok/s | vs B580 | Completion p50 / p95 / p99 (s) |
|---|---|---:|---:|---:|
| Burst | B580 | 271.3 | 1.00× | 5.68 / 10.84 / 11.32 |
| Burst | RX 570 | 120.4 | 0.44× | 13.19 / 24.48 / 25.51 |
| Burst | M4 | 68.5 | 0.25× | 21.49 / 42.87 / 44.82 |
| Burst | B580 + RX 570 | 378.6 | 1.40× | 4.28 / 8.02 / 8.11 |
| Burst | B580 + M4 | 319.8 | 1.18× | 4.71 / 8.96 / 9.61 |
| Burst | RX 570 + M4 | 174.5 | 0.64× | 9.11 / 16.25 / 17.60 |
| Burst | **B580 + RX 570 + M4** | **389.4** | **1.44×** | **3.74 / 6.57 / 7.89** |
| 1 request/s | B580 | 130.8 | 1.00× | 0.49 / 0.51 / 0.52 |
| 1 request/s | RX 570 | 125.0 | 0.96× | 1.31 / 1.54 / 1.57 |
| 1 request/s | M4 | 66.3 | 0.51× | 12.27 / 22.30 / 23.35 |
| 1 request/s | B580 + RX 570 | 130.6 | 1.00× | 0.57 / 1.04 / 1.04 |
| 1 request/s | B580 + M4 | 128.2 | 0.98× | 0.55 / 1.97 / 2.00 |
| 1 request/s | RX 570 + M4 | 123.3 | 0.94× | 1.06 / 2.02 / 2.04 |
| 1 request/s | **B580 + RX 570 + M4** | **130.7** | **1.00×** | **1.04 / 1.98 / 1.99** |

Round-by-round checks for the trio:

| Pattern | Round | Throughput vs B580 | Arrival p99 vs B580 | Beats every pair? |
|---|---:|---:|---:|---|
| Burst | 1 | 1.50× | 0.67× | Yes |
| Burst | 2 | 1.44× | 0.70× | Yes |
| Burst | 3 | 1.37× | 0.73× | **No** |
| 1 request/s | 1 | 0.98× | 4.15× | No |
| 1 request/s | 2 | 1.00× | 3.83× | No |
| 1 request/s | 3 | 1.00× | 3.78× | No |

The trio improves the median burst throughput over B580 alone, but misses the 1.50× gate on two of three burst rounds
and does not beat the best pair in the third. At one request per second, it adds no aggregate throughput and increases
tail latency substantially. The simple next-available-worker queue sent a median of 4 of 24 burst jobs to the M4 and
7 of 24 steady jobs to it; a gain-aware admission policy should avoid putting a slower worker on the critical path when
the B580 can already keep up.

## Integrity, network and limits

- **Output integrity passed:** all 1,008 requests generated exactly 128 tokens, and all token-sequence hashes matched
  across workers, conditions and rounds (one unique token hash in the raw data).
- **Network qualification failed:** the X79's RX 570 host negotiated 100 Mb/s full duplex, below the protocol's
  ≥900 Mb/s wired requirement. The Mac controller used Wi-Fi; prior five-ping averages were 9.295 ms to the B580 and
  9.896 ms to the X79. `iperf3` was unavailable on the X79. Treat every result involving the RX 570 as
  network-conditioned; this is not a formal network-qualified pass.
- No GPU clock, power, or temperature telemetry was captured, so thermal/clock drift was not independently excluded.
- The runner records per-request relative arrival/dispatch/TTFT/completion offsets, not absolute per-request wall-clock
  start/end timestamps; the file-level UTC timestamp is preserved in the JSON. This is a harness metadata limitation.
- This experiment distributes **independent requests** over persistent worker streams. It does not split one response's
  decode across GPUs, and it does not benchmark the production `seedplane` scheduler or establish general speedup.
- The reported `service round-trip − runtime decode time` is a non-decode remainder, not pure network overhead; it also
  includes prefill, tokenization, scheduling, transport, and serialization.

## Decision and next test

The experiment proves that this runtime can execute the same deterministic workload on a heterogeneous three-device
pool without output divergence. It does **not** prove the desired scaling gate. Next, fix or replace the X79 100 Mb/s
link, wire the Mac if possible, capture pairwise link throughput and device telemetry, and add gain-aware worker
admission so slower devices are used only when their added capacity outweighs their queue/tail cost. Then repeat this
unchanged protocol as a separately identified run; do not relabel these network-conditioned numbers as a pass.
