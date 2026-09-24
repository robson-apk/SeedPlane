# Hive / speculative forest — prototype status

## What this test establishes

`seedplane.speculative` is a backend-neutral correctness reference for one decode
wave. It merges duplicate draft prefixes into a trie, checks each candidate path
against a target-logits callback, emits the target token at the first mismatch,
selects the branch with the longest accepted draft prefix, and advances a commit
ledger only after verification. `tests/test_speculative.py` covers these
invariants with deterministic logits.

Run it with:

```bash
python -m unittest tests.test_speculative -v
```

## What it does not establish

The forest module is not yet an integrated decoder. A separate exploratory
M4 MPS experiment now batches a *linear n-gram proposal* through the Python
Qwen2Engine and produced a modest same-output speed signal; it does not batch a
branching tree or use native Vulkan. The drafter is local history-copying, not
another model/device. There is no cross-device live KV transfer, KV
paging/prefix fabric, cancellation transport, capability/energy telemetry, or
multi-host Hive service. See
[`experiments/hive_scaling/RESULTS.md`](../experiments/hive_scaling/RESULTS.md)
and the raw paired timings in
[`experiments/hive_scaling/mac_m4_mps_speculative_exploratory.json`](../experiments/hive_scaling/mac_m4_mps_speculative_exploratory.json).

Environment smoke test on 2026-09-24: the existing macOS ARM64 Vulkan runtime
loaded `/Users/robson/qwen05_v1.sp` on Apple M4 and generated 4 tokens from a
text prompt (`prefill_s=0.032903`, `decode_tok_s=86.005` for this short single
run). This proves the local target runtime/model are usable for a next-stage
integration test; it does not measure speculative decoding or establish a
performance baseline (four tokens are far too few for that).

Current SeedPlane components that can be reused:

| Diagram block | Current status |
| --- | --- |
| Hive device selection | Partial: `seedplane.planner` models device rates, memory, bandwidth and link latency; it is a planner, not live telemetry. |
| Context plane / prefill | Partial: existing shard-window prefill and worker scheduling distribute independent windows, not shared generation context. |
| Global KV / prefix fabric | Not implemented: Qwen engines keep their KV local to one runtime/device. |
| Draft branches / forest | Minimal correctness prototype in `seedplane.speculative`; no real drafters or transport. |
| Verifier island | Target-logits callback contract plus experimental linear batching in Python MPS; no integrated tree verifier in native Vulkan. |
| Commit ledger | Local token-frontier prototype; no distributed cancellation/rebase. |

## Next falsifiable experiment

Next add a true batched tree-verification path to the native Qwen runtime and
repeat across prompts, lengths, and draft sources; compare it with ordinary
greedy decode using the same bundle, prompt, device, and token semantics.
Record accepted draft tokens per target pass, end-to-end tokens/s,
time-to-first-token, p50/p95 inter-token latency, peak memory, and energy if
measurable. Only then add remote drafters and include network RTT and
serialization in the measurements. The single-device baseline must remain the
fallback whenever collaboration is slower.

Initial CPU thread-scaling and heterogeneous CPU/GPU pool measurements are
recorded separately in [`experiments/hive_scaling/RESULTS.md`](../experiments/hive_scaling/RESULTS.md).
Those exploratory results measure independent-request scheduling, not this
speculative-decoding design; they must not be interpreted as a direct Hive
versus legacy-generation comparison.
