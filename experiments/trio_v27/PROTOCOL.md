# V27-T3 — native-runtime heterogeneous pool pre-registration

**Status:** pre-registered; no measurements included yet.  
**Question:** does pooling independent requests across the B580, RX 570 and M4 increase useful local-AI throughput, and what does it cost in request latency and coordination?  
**This is not a test of splitting one generation across three devices.**

## Fixed workload

- Runtime: latest native `qwen_vk --serve` build from the SeedPlane `main` CI artifacts, with each platform's matching shader directory. Record artifact SHA-256 and source commit for every executable.
- Model: one complete native Qwen bundle copied unchanged to all three hosts. Record SHA-256 for `weights.spw`, `tokenizer.json`, `seedplane.json`, and the complete bundle manifest on each host. Any mismatch invalidates a formal comparison.
- Request text: `Explain in one concise paragraph how a computer memory cache improves performance.`
- Requests per condition: 24; output length: exactly 128 greedy tokens; temperature 0, top-k 0, top-p 1, reset session for every request, no stop strings.
- All machines remain plugged in, idle except for the experiment, with the same runtime/model resident for the full matrix. One untimed warm-up per worker before measurement; discard warm-up.
- Preserve per-request token hashes and require identical output across devices and conditions. Any mismatch invalidates the affected comparison and is reported, never filtered out.

## Conditions and arrival patterns

Run the full seven-condition matrix: B580; RX570; M4; B580+RX570; B580+M4; RX570+M4; B580+RX570+M4. This supplies isolated baselines, every pair, and the complete pool.

For each condition, run both workloads:

1. **Synchronized burst:** all 24 requests arrive at t=0 and enter one shared FIFO queue.
2. **Steady arrivals:** 24 requests arrive at 1.0-second intervals; each is eligible for dispatch only at its scheduled arrival time. Requests are still queued centrally and dispatched to the next available worker.

Repeat the complete condition matrix three times. Rotate the condition ordering by round to reduce ordering/thermal bias; execute each condition's burst and steady run consecutively. Leave workers resident between runs. Record room/host temperature if available, GPU clocks/power, and any unrelated load; rerun rather than silently excluding a contaminated round.

## Measurements

For each condition and arrival pattern, report:

- aggregate generated tokens/s and completed requests/s, using elapsed time from first scheduled arrival through final completion;
- time-to-first-token and completion latency from each request's scheduled arrival (therefore including queue wait): p50, p95, p99;
- queue wait and worker service/SSH round-trip latency: p50, p95, p99 per worker and overall;
- runtime-reported decode tokens/s per worker; request assignments and completed work per worker;
- bytes sent and received by the JSON-lines control stream, and `service round-trip − runtime-reported decode time` as **non-decode remainder**. Do not label this remainder as pure network or coordinator overhead: it also includes prefill, tokenization, scheduling, transport, and serialization.
- per-request token hash, request count, generated-token count, errors, worker identity/device string, and start/end timestamps.

Use the nearest-rank percentile definition `sorted[ceil(p*n)-1]`. With 24 requests, p99 is the maximum observation; state this explicitly. For the steady workload, latency starts at scheduled arrival, not at dispatch. Report both arrival-to-completion and service latency so queueing cannot be mistaken for worker speed.

Before the timed matrix, capture link negotiation and RTT for every pair. The X79 was previously observed at 100 Mb/s full duplex; until corrected and verified, any result involving it is **exploratory and network-conditioned**, not a formal pass of the ≥900 Mb/s network gate. Do not combine the cost of staging the model with inference timing; verify identical hashes after staging.

## Prospective gates (not results)

- Integrity: complete identical bundle hashes, same request/settings, exactly 24 × 128 tokens, and identical greedy token hashes.
- Throughput: trio aggregate throughput at least 1.50× B580-only on both workloads, and higher than every pair on at least the burst workload. Show every round; do not decide on median alone if a round regresses.
- Latency: trio arrival-to-completion p99 no more than 1.20× B580-only in each workload. Also publish service p99; it is diagnostic and cannot replace the arrival-based gate.
- Network qualification: each wired path ≥900 Mb/s before calling this a formal network-qualified V27 result.

If a gate fails, retain the raw data and diagnose scheduler policy, slower-worker tail, or network limits. Never retune request count, arrival schedule, token length, or gate after seeing the measurements; any new workload is a separately versioned protocol.

## Reproducibility and artifact handling

The runner must write into a new, explicit run directory and must not overwrite prior V22/V22b/V23 experiment directories. Keep the runner, exact command, environment/build IDs, JSON output, and a human-readable `RESULTS.md` beside this protocol. Do not publish model weights or credentials. If any host is already under another benchmark, postpone GPU measurements until it is idle.
