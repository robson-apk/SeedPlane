# HIVE implementation baseline

**Status:** agreed architecture baseline; implementation is staged, not complete.

HIVE is a heterogeneous, hierarchical scheduler with separate execution lanes.
It selects a parallelism strategy only when the measured cost model predicts a
net gain. The goal is not to keep every device busy at any cost; useful idle is
preferable to work that delays the critical path or cannot be committed.

## Execution lanes

| Lane | Work | Primary outcome | Initial scheduling unit |
|---|---|---|---|
| `THROUGHPUT` | Independent requests, compatible prefill chunks, microbatches | Aggregate completed tokens/s and requests/s | Ready request / prefill tile |
| `LATENCY` | Draft proposals, speculative branches, target tree verification, commit | TTFT, inter-token p50/p95, accepted tokens per target pass | Dependency-ready branch / verify wave |
| `MODEL_PARALLEL` | Sequence/tree × hidden/tensor operations inside an island | Lower latency for a model operation | Executor-supported tensor tile |

Throughput results do not establish single-generation speedup. A single
autoregressive response cannot be partitioned into arbitrary token ranges;
latency parallelism comes from speculative branches and batched verification.
Hidden/tensor sharding is disabled across hosts by default. Enable it only
inside an island when measured saved compute exceeds collective and sync cost.

## Topology

The current home lab is three device islands, not a uniform GPU mesh:

```text
Island A: Ryzen 5600X + Arc B580 + local RAM/KV
Island B: Xeon E5-2650 v2 + RX 570 + local RAM/KV
Island C: Apple M4 CPU/GPU + unified memory/local KV
```

Between islands, prefer coarse work and compact data (requests, useful prefill
chunks, draft branches, logits, or KV pages only when transfer pays). Do not
move hidden activations layer-by-layer across the LAN without a measured
crossover showing a net benefit.

## Ready work and compatibility

Each lane feeds a global reservoir of **dependency-ready** work. Agents keep a
model resident and pull one compatible lease at a time. Work is eligible only
when the agent advertises the requested model and all required capabilities.
The coordinator accounts for deadlines, priorities, backpressure, and device
locality; work stealing must not violate KV/session ownership.

`WorkTile` records kind, model, dependency lineage, capability requirements,
optional sequence/hidden ranges, token budget, branch, predicted cost, deadline,
movability, and stealability. Ranges do not imply that every executor can run
them. The scheduler may enqueue only ready work; a future dependency manager
must resolve prerequisites before release to this queue.

Leases adapt to measured fixed cost and compute cost:

```text
T_total = queue + dispatch + serialization + RTT + transfer + compute + sync
```

There is no universal 50 ms quantum. Choose work size so fixed overhead is an
acceptable fraction for that device/link/task, then update observations online.
Start with measured p50/p95 and a simple EWMA or regression keyed by task,
model, device/island, size, batch, load, RTT, bandwidth, and KV locality. Do not
train a scheduler policy until there is a representative trace dataset.

## Transport and security

HIVE protocol v1 has a fixed-width, versioned binary frame header and persistent
TCP connection. Work/control metadata is compact JSON inside the frame for
rapid iteration; large tensor payloads and a typed binary data plane remain
future work. A challenge-response HMAC authenticates agents when a key is set.
The default broker binds loopback; non-loopback binds require an explicit key.
HMAC does not encrypt traffic, so remote use must remain on a trusted network
or gain TLS before exposure to untrusted networks. SSH may deploy/start agents
and collect logs, but is not the request transport.

## Initial implementation and measured smoke test

The repository already had a native TCP/binary persistent worker for
llama.cpp-based prefill windows and a dynamic coordinator-side scheduler. The
new `seedplane.hive` path adds persistent pull agents and a broker ready queue
for generation tasks. It matches work by advertised model and capabilities,
uses request IDs plus monotonically unique lease sequences, and exposes a
native `qwen_vk` agent adapter. It currently has no retries after agent loss,
no cross-host TLS, no async streaming of output tokens, and no dependency
resolver or adaptive cost model.

On the M4, six independent 16-token native Qwen requests produced identical
text hashes in direct and pull-agent paths. Direct measured 63.54 tok/s and the
loopback pull path 63.19 tok/s (0.994×) in this single short run. This confirms
the vertical slice works on the real runtime and gives only a preliminary
local transport-overhead check; it is not a multi-host scaling result or a
formal throughput benchmark. Reproduction script and raw JSON:

- [`experiments/hive_scaling/benchmark_hive_pull.py`](../experiments/hive_scaling/benchmark_hive_pull.py)
- [`experiments/hive_scaling/mac_m4_hive_pull_smoke.json`](../experiments/hive_scaling/mac_m4_hive_pull_smoke.json)

The first three-island comparison also completed: six requests × 16 tokens,
same prompt and native runtime, individually warmed workers, and exact token
hashes matching between paths. Direct JSON-lines/SSH versus HIVE persistent TCP
pull measured 313.66 vs 303.46 tok/s on burst and 54.07 vs 53.57 tok/s with
0.3 s arrivals (pull/direct 0.967× and 0.991×); p95 was 0.306 vs 0.316 s and
0.275 vs 0.292 s, respectively. All three islands participated. This single
round shows the HIVE path is functional but does not yet outperform the
existing transport. Six samples make p95 the maximum; the earlier exploratory
334/57 tok/s pool result is not reconciled. See
[`experiments/hive_scaling/RESULTS.md`](../experiments/hive_scaling/RESULTS.md)
and [`hive_fleet_transport_v1_20260924.json`](../experiments/hive_scaling/hive_fleet_transport_v1_20260924.json).

Unit tests cover fragmented frame reads, compatibility-based pull assignment,
persistent sessions, and HMAC authentication. Formal fleet results still need
rotated repeated rounds, equal prompts/output lengths, warm-up, output
integrity, per-request queue/service latency, and reconciliation against the
earlier exploratory pool. Previously observed aggregate figures used different
harnesses/workload order and must not be summed as if they were one calibrated
ceiling.

The follow-up ran three rotated rounds across all seven GPU pool combinations
(three individual devices, three pairs, and the trio), keeping six requests ×
16 tokens per condition. All greedy output hashes matched. Median burst rates
for direct vs HIVE pull were: B580 211.1/188.5, RX 570 106.7/100.0, M4
65.3/60.4, B580+RX 310.4/299.0, B580+M4 182.7/177.2, RX+M4 158.2/150.0,
and trio 311.0/299.6 tok/s. For the trio, pull/direct was 0.961× and p95 was
0.309/0.320 s. The isolated-device burst sum was 383.1 tok/s, yielding 81.1%
direct and 85.7% pull fleet efficiency. Since B580+RX reached 310.4 tok/s,
adding M4 did not increase the measured burst ceiling in this six-request
case. This is an early signal about queue/work assignment, not a verdict on
M4 capacity. The 0.3 s arrival lane is underloaded and not a capacity test.
Three rounds are still exploratory; p95 is based on six completions and
therefore effectively the maximum. Pull is close but not faster than direct on
burst work, motivating measured cost estimates, adaptive lease sizes, and
microbatching before performance claims. Raw evidence:
[`hive_fleet_matrix_v1_3rounds_20260924.json`](../experiments/hive_scaling/hive_fleet_matrix_v1_3rounds_20260924.json).

A larger pre-fix follow-up used 16 requests × 32 tokens in three rounds across
the same seven pools. Hashes again matched. Its completion-order scheduler
over-allocated work to the M4 tail; the later controlled admission test below
supersedes its trio-vs-pair conclusion. Median burst direct/pull tok/s were:
B580 247.4/230.8; RX570 118.5/109.3; M4 66.1/65.1; B580+RX570 362.7/339.0;
B580+M4 261.3/248.4; RX570+M4 168.0/168.9; trio 330.6/340.1. Trio pull was
1.049× direct with p95 1.549/1.505 s, and each trio round assigned 9/4/3
requests to B580/RX570/M4. But the trio did not exceed the B580+RX570 direct
pair (362.7 tok/s); it only narrowly exceeded the pair on pull. Against one
common denominator (sum of direct isolated rates in each round), median trio
efficiency was 76.7% direct and 78.6% pull. The 0.3 s-arrival lane is not
saturated-capacity evidence. So this larger run hints that persistent pull
transport may help the trio versus its SSH/stdout control path at medium work
size; it does not show that the M4 raises the scaling ceiling or prove
adaptive scheduling. Three rounds remain exploratory. See
[`hive_fleet_matrix_v1_16x32_3rounds_20260924.json`](../experiments/hive_scaling/hive_fleet_matrix_v1_16x32_3rounds_20260924.json).

## Frozen implementation order

1. Persistent agents, framed protocol, pull queues; run the same fleet workload
   with no SSH in request transport and reconcile isolated/pool measurements.
2. Measured cost model, adaptive leases, compatible work stealing, and
   microbatching; establish break-even thresholds. **In progress:** the broker
   now uses three warmed end-to-end service samples (including lease overhead)
   to plan homogeneous independent-request batches for minimum predicted
   makespan. The direct control uses the same planner. A 16×32, three-round
   retest assigned 10/4/2 requests to B580/RX570/M4 and measured trio gains
   over the pair of 14.1% direct / 9.2% HIVE pull; with 0.3 s arrivals, it sent
   all work to B580. The prediction allows a slow worker to receive zero work,
   so adding a worker cannot worsen *predicted* makespan; actual wall time still
   varies. Estimates remain in-memory; asynchronous queue-aware replanning,
   splitting/retries, heterogeneous tiles, and true runtime batching remain.
3. Separate throughput and latency benchmark suites and report their metrics
   independently.
4. Real speculative runtime: concrete drafters, candidate trie, batched/tree
   verifier, backpressure, cancellation/rebase, and accepted-token metrics.
5. CPU execution plane, admitting each task only when its measured marginal
   value wins (including transfer and contention).
6. Sequence × hidden within islands; measure layer latency and collective
   crossover before allowing planner selection.
7. KV/prefix fabric, guided by measured transfer frequency and locality.
8. Compare heuristic scheduling with a learned policy only after accumulating
   real traces across representative hardware/load regimes.

## First-class evaluation metrics

```text
Fleet efficiency = measured useful aggregate throughput
                   / sum of isolated device throughput

Speculation efficiency = committed target-equivalent tokens
                         / speculative compute cost
```

Report the denominator definition (time, accelerator-seconds, energy, or a
normalized target-equivalent cost) so the efficiency ratio is reproducible.
Also record output identity, queue depth/wait, worker assignment, transfer
bytes/time, device utilization where available, memory/KV, and error/retry
counts. A high utilization number without committed useful work is not success.
