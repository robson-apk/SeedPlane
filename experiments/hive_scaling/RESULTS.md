# Hive / CPU–GPU scaling: exploratory results

**Run date:** 2026-09-24. **Verdict:** the reference CPU path does not scale well
with more threads, and a naive all-device request queue is much slower than the
GPU-only pool for short bursts. This is evidence about independent-request
scheduling, **not** a performance verdict on speculative decoding.

## What this can and cannot answer

The Hive proposal targets the latency of **one generation** by having drafters
propose branches and asking a target model to verify many positions in a batch.
SeedPlane currently has only a correctness prototype for the forest/commit
state machine; it has no real drafters, batched tree verifier, cross-device KV
fabric, or distributed cancellation/rebase. Therefore there is no implemented
Hive path to benchmark against ordinary generation yet.

The measurements below answer a narrower question first: what do the three
machines contribute when CPUs and GPUs execute **separate, independent
requests**? This is a useful baseline for the existing work-pool design, but it
must not be presented as cooperative single-answer decode.

## CPU thread scaling

Same Qwen2.5-0.5B-Instruct HF snapshot/config, raw text prompt, greedy 96-token
generation, PyTorch 2.14.0 SeedPlane `Qwen2Engine`, two warm-up/three measured
passes as recorded in the raw JSON. All thread counts on each host produced the
same token hash. CPU weights/activations are FP32. The worker is the portable
reference implementation, not an optimized AVX2/llama.cpp CPU backend.

| Host | Threads tested | Best measured | All logical threads | Result |
|---|---:|---:|---:|---|
| M4 CPU (10 total: 4 performance + 6 efficiency) | 1, 2, 4, 6, 8, 10 | 1 thread: **29.29 tok/s** | 10 threads: 27.39 tok/s | No benefit from more threads; full count was 6.5% slower. |
| Ryzen 5 5600X (6C/12T) | 1, 2, 3, 4, 5, 6, 8, 12 | 4 threads: **8.60 tok/s** | 12 threads: 7.92 tok/s | Flat curve; full SMT was 7.9% below best. |
| Xeon E5-2650 v2 (8C/16T) | 1, 2, 4, 6, 8, 12, 16 | 6 threads: **8.52 tok/s** | 16 threads: 1.93 tok/s | 16-thread result repeated at 2.04 tok/s (32 tokens); severe regression, not a one-off. |

The same PyTorch reference graph on M4 MPS reached **51.74 tok/s** for 96
tokens, with the same output hash as M4 CPU. This is an auxiliary backend result;
it is not the native Vulkan M4 runtime. Raw curves: [M4 CPU](mac_m4_cpu_thread_sweep.json),
[M4 MPS](mac_m4_mps_reference.json), [5600X CPU](5600x_cpu_thread_sweep.json),
[X79 CPU](x79_cpu_thread_sweep.json), [X79 16-thread repeat](x79_16thread_validation.json).

## CPU + GPU on one machine

Workers load the same model and serve through the same JSON-lines shape. CPU
reference workers use PyTorch FP32; GPU workers use SeedPlane `qwen_vk` and the
native FP16 bundle. Greedy output hashes matched in every reported pool. These
are deliberately small exploratory burst tests, not statistically strong
performance claims.

| Machine/pool | GPU only | CPU + GPU together | Completion p95: GPU → pair |
|---|---:|---:|---:|
| M4 + MPS/Vulkan, local, 6 × 32 tokens | 70.94 tok/s | 55.46 tok/s | 2.71 → 3.46 s |
| 5600X + B580, controller over SSH, 6 × 16 tokens | 199 tok/s | 50 tok/s | 0.48 → 1.90 s |
| X79 + RX 570, controller over SSH, 6 × 16 tokens | 54 tok/s | 11 tok/s | 1.77 → 8.67 s |

The same-host micro-pair checks also completed: B580+5600X CPU (3 × 16 tokens)
reported 21.3 tok/s burst, p95 2.25 s; RX 570+X79 CPU reported 5.8 tok/s,
p95 8.27 s. Both preserved the greedy output hash. The M4 local pair is recorded
in [mac_cpu_gpu_pool_exploratory.json](mac_cpu_gpu_pool_exploratory.json).

**Interpretation:** assigning a whole short request to a much slower CPU worker
puts that request on the critical path. The GPU can be idle while the pool
waits for the CPU tail. The remedy for independent requests is admission based
on queue depth / measured break-even, not “use every device on every request.”
This does not rule out CPUs as speculative drafters: a small drafter can help a
single generation if enough proposals are accepted and the verifier batches
them cheaply. That needs its own implementation and test.

## Fleet pool: all CPUs, all GPUs, all six workers

The Mac coordinated six persistent workers (CPU and GPU on each host). Every
worker agreed on the 16-token warm-up hash; no within-pool output mismatch was
recorded. One burst round, six requests × 16 tokens; “steady” arrivals were
spaced 0.3 seconds apart. At six samples p95/p99 are the maximum, so treat tail
figures as directional only.

| Pool | Burst throughput | Burst completion p95 | 0.3 s arrivals throughput | Completion p95 |
|---|---:|---:|---:|---:|
| B580 + RX 570 + M4 GPUs | **334 tok/s** | 0.287 s | 57 tok/s | 0.279 s |
| All three CPUs (PyTorch reference) | 11 tok/s | 8.02 s | 10 tok/s | 8.25 s |
| All three CPUs + all three GPUs | 11 tok/s | 8.46 s | 9 tok/s | 8.72 s |

The all-six pool was about **30× slower** than the GPU-only pool on this short
burst, with p95 roughly **29× higher**. The X79 16-thread CPU worker dominated
the tail. The slower workers were assigned work because this diagnostic
harness intentionally exercised every member; it is not the production
gain-aware scheduler. Full per-request raw data is in
[fleet_cpu_gpu_pools_exploratory.json](fleet_cpu_gpu_pools_exploratory.json).

Same-controller individual/pair GPU conditions, including output checks, are
in [fleet_gpu_pairs_exploratory.json](fleet_gpu_pairs_exploratory.json). The
longer previous GPU-only benchmark remains [V27-T3](../trio_v27/RESULTS.md):
its burst trio reached 389.4 tok/s (1.44× B580), but missed the 1.50× gate and
was network-conditioned (X79 link 100 Mb/s; Mac on Wi-Fi).

## First single-generation speculative test (M4 MPS)

To test the central latency hypothesis before changing the native runtime, an
experimental Python path copied proposals from repeated suffix n-grams in the
committed token history and verified each proposal block with one multi-token
Qwen2Engine forward. It used the same Qwen2.5-0.5B snapshot, prompt and greedy
token semantics as the baseline: 128 output tokens, one full baseline warm-up,
one 16-token warm-up per speculative width, and five paired rounds in randomized
configuration order. All output hashes matched the ordinary greedy baseline.

| M4 MPS mode | Median decode rate | Median time | Paired round speed change vs greedy |
|---|---:|---:|---:|
| Greedy, one token per forward | 55.89 tok/s | 2.290 s | baseline |
| N-gram speculative, width 2 | 58.74 tok/s | 2.179 s | +4.1% median paired |
| N-gram speculative, width 4 | 58.58 tok/s | 2.185 s | +4.5% median paired |
| N-gram speculative, width 8 | 58.62 tok/s | 2.183 s | +4.9% median paired |

For width 8, only 21 of 340 proposed tokens were accepted (6.2%); it issued 12
batched target calls and 107 single-token calls. This suggests that even a weak
proposal source can benefit from MPS block execution in this particular small
model/prompt case, but the apparent gain is modest and must be retested with
better draft sources, more prompts/lengths, confidence intervals, inter-token
latency, and an optimized/native verifier. It does not test branching trees,
other GPUs, remote drafters, or the B580+RX 570+M4 fleet.

Reproduce from the repository root (with the existing `.venv-hive` and cached
HF snapshot):

```bash
PYTHONPATH=. .venv-hive/bin/python experiments/hive_scaling/benchmark_speculative.py \
  --bundle /path/to/Qwen2.5-0.5B-Instruct-snapshot --device mps \
  --tokens 128 --repeats 5 --widths 2 4 8 \
  --output experiments/hive_scaling/mac_m4_mps_speculative_exploratory.json
```

Raw paired timings and per-wave counters are in
[mac_m4_mps_speculative_exploratory.json](mac_m4_mps_speculative_exploratory.json).

## Three-island HIVE pull transport and pool matrix (3 rounds)

This repeated experiment expanded the first three-island check to all seven
GPU pools: B580, RX 570, M4, each pair, and all three together. Three rotated
rounds used the same native Qwen runtime, prompt, six requests × 16 greedy
tokens, per-agent warm-up, and exact token-hash comparison. Direct stream and
HIVE pull order alternated; pool order was rotated. The burst lane submits all
six requests together. The `steady_0.3s` lane intentionally spaces arrivals
and therefore measures this arrival workload, not saturated pool capacity.

| Pool | Burst direct | Burst HIVE pull | Pull/direct | Spaced direct | Spaced HIVE pull |
|---|---:|---:|---:|---:|---:|
| B580 | 211.1 | 188.5 | 0.886× | 60.9 | 60.7 |
| RX 570 | 106.7 | 100.0 | 0.938× | 57.8 | 57.8 |
| M4 | 65.3 | 60.4 | 0.939× | 54.3 | 54.4 |
| B580 + RX 570 | 310.4 | 299.0 | 0.941× | 57.8 | 57.8 |
| B580 + M4 | 182.7 | 177.2 | 0.990× | 53.8 | 54.1 |
| RX 570 + M4 | 158.2 | 150.0 | 0.947× | 54.0 | 53.9 |
| B580 + RX 570 + M4 | **311.0** | **299.6** | **0.961×** | 54.0 | 54.6 |

Rates are median tok/s over only three rounds. All output hashes matched in
every pool and path. For the trio, median burst completion p95 was 0.309 s
(direct) and 0.320 s (pull); with spaced arrivals it was 0.279 and 0.277 s.
With six requests, p95 is effectively the maximum and is directional only.
The HIVE pull path is within 3.9% of direct for the trio burst, but this test
does not show a throughput improvement. On spaced arrivals it is effectively
equal; the offered load is too low to establish saturated capacity.

The direct isolated-device burst medians sum to about 383.1 tok/s. Relative
to that common denominator, trio burst fleet efficiency is 81.1% for direct
and 78.2% for pull. (The legacy `fleet_efficiency_pull_median` field instead
normalizes pull-trio against pull-isolated rates; do not compare that field
directly to direct efficiency.) The trio's 311 tok/s is essentially the same as the B580+RX 570 pair's
310 tok/s in this small workload, so the M4 did not raise the observed burst
ceiling. This is not proof the M4 is useless: only six requests are available
to distribute, there is no online cost model, and some device capacity can be
left unused. In the spaced lane, the observed ~54 tok/s primarily reflects
arrival cadence and cannot be used as a fleet capacity/efficiency result.

For each isolated GPU, HIVE pull trails direct by 6–11% on bursts, while the
two-GPU pools trail by 1–5.9%; likely fixed broker/protocol/dispatch overhead
is more visible for short work. These measurements motivate the next
implementation stage: measured cost model, adaptive leases, and batching; do
not justify a claim that distributed scheduling is faster yet. This is still
independent-request throughput testing, not single-response latency or
speculative decoding.

The complete per-request timings, assignments, environment, and round order
are in [hive_fleet_matrix_v1_3rounds_20260924.json](hive_fleet_matrix_v1_3rounds_20260924.json).
The earlier single-round run below is retained as historical context.

## Expanded fleet matrix: 16 requests × 32 tokens (3 rounds, pre-fix)

To reduce the small-queue limitation, the seven-pool matrix was repeated with
16 requests × 32 tokens, still three rotated rounds and both arrival patterns.
All per-request token hashes matched. Burst median direct/HIVE-pull tok/s:

| Pool | Direct | HIVE pull | Pull/direct |
|---|---:|---:|---:|
| B580 | 247.4 | 230.8 | 0.936× |
| RX 570 | 118.5 | 109.3 | 0.922× |
| M4 | 66.1 | 65.1 | 0.972× |
| B580 + RX 570 | **362.7** | 339.0 | 0.941× |
| B580 + M4 | 261.3 | 248.4 | 0.936× |
| RX 570 + M4 | 168.0 | 168.9 | 1.000× |
| B580 + RX 570 + M4 | 330.6 | **340.1** | **1.049×** |

For the trio, direct/pull completion p95 was 1.549/1.505 s. Assignments were
identical and stable in each burst round: B580 9, RX 570 4, M4 3 of 16
requests. The M4 was used, but the trio did not beat the B580+RX570 pair's
direct rate; pull trio only narrowly exceeded the pair's pull rate. Three
rounds are still a small sample. Across the three rounds the median trio
efficiency against the **same direct isolated-device denominator** was 76.7%
direct and 78.6% pull. The JSON's legacy path-specific `fleet_efficiency_*`
fields use a different denominator for each transport and should not be
compared across lanes; the runner has been corrected and will additionally emit
`fleet_efficiency_pull_vs_direct_isolated_median` for fair cross-path reading.

The 0.3 s arrival lane reached about 102 tok/s for the trio pull path (direct
102.2), with the M4 and pair combinations close behind. That is an offered
load test, not a saturated capacity estimate. The main transport result is
mixed: pull was ~6–8% behind direct for individual accelerators and the B580
pairs, essentially tied on RX570+M4, and 4.9% ahead for the trio in this
medium workload. This suggests that removing remote process/SSH-stream costs
can pay off for the combined pool once work is larger, but the per-round
variation and unchanged assignment pattern mean this is not proof of
cost-aware scheduling or M4-derived speedup. Raw data:
[hive_fleet_matrix_v1_16x32_3rounds_20260924.json](hive_fleet_matrix_v1_16x32_3rounds_20260924.json).

## M4 admission / no-regression fix (3 rounds)

The pre-fix run above exposed the failure mode: the direct completion-order
queue gave the trio 9/4/3 requests (B580/RX570/M4), although per-request service
times were about 0.12/0.27/0.52 seconds. The third M4 request extended the
batch tail, so adding a worker could make the measured batch finish later.
This was a scheduling policy problem, not evidence that M4 inference itself
was slower than its isolated measurements.

The direct benchmark and HIVE batch admission now share a measured-cost
minimum-predicted-makespan planner. It uses three warmed end-to-end service
samples per worker, including dispatch/transport/result overhead, then assigns
whole independent requests proportionally to predicted service time. A worker
receives no work when it cannot improve predicted makespan. For paced arrivals,
the scheduler keeps work on the fastest profiled worker when it can finish
before the next arrival. This planner applies to homogeneous independent
request batches; it is not yet a general asynchronous queue optimizer.

The controlled rerun used the same native model/runtime, prompt, greedy output,
16 requests × 32 tokens, three rounds, both direct and persistent HIVE-pull
paths, and exact output-hash checks:

| Pool | Direct burst | HIVE pull burst | Trio vs pair, direct / pull |
|---|---:|---:|---:|
| B580 + RX 570 | 351.6 tok/s | 339.1 tok/s | baseline |
| B580 + RX 570 + M4 | **396.6 tok/s** | **371.1 tok/s** | **+14.1% / +9.2%** |

Each burst round planned and executed 10/4/2 tasks on B580/RX570/M4. Median
completion p95 changed from 1.456 to 1.291 s direct and 1.510 to 1.380 s with
pull. In the 0.3-second-arrival lane the scheduler assigned all 16 tasks to the
B580; trio throughput was effectively tied with the pair (110.59 vs 110.61
tok/s direct, 110.10 vs 110.33 pull), rather than burdening the batch with
slower workers. All output hashes matched.

This is a three-round early result, not a universal wall-clock guarantee:
service estimates and the optimal plan are conditional on the measured workload,
while OS/network/thermal noise remains. The safety property is at the planner
level—adding a worker cannot increase its predicted optimal makespan because
it can always be assigned zero work. Raw per-request data, profiles,
assignments, and summaries:
[hive_no_regression_test_v2_16x32_3rounds_20260924.json](hive_no_regression_test_v2_16x32_3rounds_20260924.json).

The broker also now receives agent-measured service duration and records
bounded samples by task/model/worker/size/batch/load. Three samples are
required before exposing p50/p95 estimates. This data currently stays in
memory and does **not** steer assignment. A tested lease-sizing helper
computes the minimum divisible-work size needed to amortize fixed overhead;
generation requests are not split. Actual model-aware assignment and runtime
microbatching remain future work, since this native executor still serves one
request at a time.

## First three-island HIVE pull transport comparison

The first real B580 + RX 570 + M4 HIVE agent run used the native `qwen_vk`
runtime and the same six independent requests × 16 greedy tokens for both
paths. Each direct worker remained a persistent JSON-lines process over its
existing local/SSH stdio stream. In the HIVE path, SSH only launched the agents;
request/result traffic used persistent authenticated framed TCP to the Mac
broker. Every agent was individually warmed; direct and HIVE per-request token
hashes were identical in both arrival patterns.

| Pattern | Direct stream tok/s / p95 | HIVE pull TCP tok/s / p95 | Pull/direct throughput |
|---|---:|---:|---:|
| Synchronized burst | 313.66 / 0.306 s | 303.46 / 0.316 s | 0.967× |
| Arrivals every 0.3 s | 54.07 / 0.275 s | 53.57 / 0.292 s | 0.991× |

Burst assignment was B580 3, RX 570 2, M4 1 on both paths. In the spaced run,
direct assigned 2/2/2 and pull assigned B580 1, RX 570 1, M4 4. Each pool
reached all three islands. With only six requests, nearest-rank p95/p99 is the
maximum and assignment variation is expected. This is one exploratory round,
not evidence of a throughput improvement; the new transport was ~3.3% slower
in the burst and ~0.9% slower in the spaced run.

This same-workload direct baseline measured 313.7 tok/s burst and 54.1 tok/s
spaced, versus the earlier exploratory pool's 334/57 tok/s. Those runs used
different harnesses/order and are not statistically reconciled. Re-run paired,
rotated rounds before setting the fleet-efficiency denominator or calling the
sum of isolated device rates a measured ceiling. Raw per-request timings,
assignments, warm-up identity and environment are in
[hive_fleet_transport_v1_20260924.json](hive_fleet_transport_v1_20260924.json).
The runner is [benchmark_hive_fleet.py](benchmark_hive_fleet.py); it refuses to
overwrite an existing output. HMAC authenticates this trusted-LAN test but does
not encrypt it.

## Reproducibility and caveats

- Model config SHA-256 on all CPU snapshots:
  `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`.
- Native bundle weights SHA-256:
  `ef9f3f59f925e46a303193d7b88a979a899c53b5fec2ed0ff679b61fdf3cbe49`.
  The Windows bundle was converted from its cached HF snapshot and matched the
  existing M4/X79 bundle. Six workers had identical 16-token warm-up output.
- CPU code/runtime differs from the optimized Vulkan GPU runtime and uses
  FP32 vs FP16 weights. The tested CPU backend is a correctness reference and
  may substantially understate what optimized AVX2 CPU kernels can do.
- Fleet control used SSH from the Mac; Mac Wi-Fi and the X79 100 Mb/s wired
  link remain in the path. The microburst includes scheduling, transport,
  tokenization and serialization. No power/temperature/clock telemetry was
  captured. There were no extra model downloads; isolated Python environments
  were installed on each host (Windows CPU-only PyTorch kept the broken XPU
  installation untouched).
- Aggregate throughput and request completion latency answer a different
  question from single-generation token latency. Do not compare these numbers
  as if they measured speculative decoding.

## Decision / next falsifiable experiment

Keep the current independent-request scheduler and GPU-only fallback. The
single-device MPS result justifies continuing, not declaring victory. Next
integrate a batched tree verifier into the native runtime and compare greedy
against speculative over multiple prompts, lengths, draft sources and devices.
Measure acceptance per target pass, TTFT, inter-token p50/p95, tokens/s, KV/peak
memory, bytes/RTT and energy. After that add remote drafters and compare the
B580, RX 570, M4, and all-device configurations. Until those tests pass, Hive
remains a promising but only partially evidenced upgrade.
