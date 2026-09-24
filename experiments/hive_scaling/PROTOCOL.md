# Hive scaling matrix — protocol draft v1

## Claims kept separate

This matrix measures (A) single-request generation and CPU thread scaling, and
(B) aggregate throughput from independent requests on CPU/GPU workers. It does
**not** measure the proposed speculative forest: SeedPlane has no batched tree
verifier or multi-model drafter runtime yet. A result for (B) cannot establish
that one answer decodes faster under Hive.

## Common CPU reference workload

- Model: cached `Qwen/Qwen2.5-0.5B-Instruct`, exact HF revision/config and weight
  hashes recorded per host; greedy sampling, identical plain-text prompt (no
  chat-template wrapper, to match the native JSON `generate` path).
- Runtime: SeedPlane `Qwen2Engine` reference implementation, PyTorch 2.14.0,
  safetensors/tokenizers. CPU uses FP32; record that MPS uses FP16. CPU results
  are the primary cross-machine thread-scaling comparison; MPS is auxiliary.
- Warm up 16 generated tokens; measure 96-token decode, two repeats; report
  per-run duration, median generated tok/s, output hash, peak RAM, utilization
  and power/temperature where available. The short exploratory Mac run already
  used this workload; replace it with the pinned-version run before final.
- Thread sweeps: Ryzen 5600X 1/2/3/4/5/6 physical-core threads plus 8/12 SMT;
  Xeon E5-2650 v2 1/2/4/6/8 physical cores plus 12/16 SMT; M4 1/2/4/6/8/10
  total physical cores (4 performance + 6 efficiency). Verify actual topology.
- Interleave thread counts or rotate their order in a new formal run; idle hosts
  except for the benchmark. Three repeats are required for formal reporting.

## Device/pool conditions

1. CPU alone per host; full thread sweep above.
2. GPU alone per host: same SeedPlane native Vulkan/MoltenVK runtime and bundle
   as V27 where available (B580, RX 570, M4); rerun a common prompt/length.
3. CPU + GPU on each host; CPU and GPU execute independent queued requests in
   parallel. This is aggregate local throughput, not collaborative decode.
4. All CPUs, all GPUs, and all six workers together. Report burst and steady
   arrivals separately, with request completion p50/p95/p99, aggregate tok/s,
   per-worker work, output hashes, errors, memory, link/serialization remainder.
- Use one model/bundle/tokenizer/prompt/output length wherever runtimes allow;
  record backend and precision. Any precision/runtime mismatch is a comparison
  caveat, not silently treated as equivalent.
- Local 100 Mb/s X79 link and Mac Wi-Fi make multi-host results exploratory until
  link speed/RTT are measured and qualified. Don't claim single-request speedup.

## Hive gate needed before testing the diagram itself

Implement drafters, a batched target-tree verifier, KV rollback/rebase and a
distributed commit ledger. Then compare target-only greedy decode with the
identical target plus local/remote drafters, same prompt and fixed output
quality. Record accepted draft tokens per target pass, tokens/s, TTFT,
inter-token p50/p95, network bytes/RTT, peak memory and energy. Reject any
configuration that changes the target's greedy output or worsens latency at the
single-request workload. Preserve baseline fallback.
