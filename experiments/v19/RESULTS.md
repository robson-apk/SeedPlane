# V19 — native SeedPlane decode + native bundle converter: results

Protocol: [PROTOCOL.md](PROTOCOL.md) (written before implementing or measuring). Arc B580, Qwen2.5-0.5B-Instruct,
prompt `9707,11,1879,0`, plan S=64 / H=32 / sinks=4, 296 generated tokens (positions 0..299), 3 runs.
Raw data: `native_plan.json`, `native_full.json`, `oracle.json`, `bundle_manifest.json`, `gates.json`.

## Verdict: all five gates pass

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| G1 | same `weights.spw` from HF snapshot and v1 bundle | both `ef9f3f59…cbe49` | **pass** |
| G2 | greedy = window oracle through position 130 | identical for all 296 tokens, 3/3 runs | **pass** |
| G3 | max \|logit error\| ≤ 0.10 at dump positions | 1.4e-5 … 2.3e-5; 1.48e-4 at position 299 | **pass** |
| G4 | plan actually applied; full mode = V18 | plan vs full at pos 65: 1.857; full = V18 (128/128) | **pass** |
| G5 | steady-state decode ≥ 245 tok/s | median **257.6** (252.3 / 259.4 / 257.6) | **pass** |

G4 confirms that the manipulation was exercised. With the plan, the generated text departs from full attention at
generated token index 68 (the token at position 72, predicted from position 71, which is 7 positions after the first boundary at 64). With `--full`, the native engine reproduces V18
token for token.

## Reported, not gated

| | SeedPlane plan (S64/H32/K4) | full attention (same engine) |
|---|---:|---:|
| overall decode tok/s (incl. rebuilds) | 171.8 / 177.5 / 172.6 | 228.8 / 233.3 / 229.2 |
| steady-state tok/s (no rebuild) | 252.3 / 259.4 / 257.6 | — |
| boundary rebuilds / extra submissions | 4 / 144 | 0 |
| KV cache | 2.46 MB (100 slots) | 7.37 MB (300 slots) |

- Bundle load: 1.10 s (V18, converting safetensors at load time: 2.29 s).
- FP16 conversion: 0 overflows, 716 values flushed to zero. That matches the independent C++ count in V18.

## What this does and does not show

- **Measured:** the native engine decodes the SeedPlane model, meaning the exact `ShardPlan.windows` semantics, and
  matches an independent from-scratch window oracle to FP32 rounding.
- **Measured, unfavourable:** with this small test plan, SeedPlane decoding is **slower overall** than full attention
  (~174 vs ~230 tok/s). Each boundary re-runs sinks + halo (36 tokens every 64) one submission at a time.
  With the default plan (S=512/H=256) the re-run is 260 tokens every 512.
- **Inferred, not measured:** at this length, full attention is already slower per token than a 100-slot window
  (229 vs 258 tok/s). The single-workgroup-per-head attention kernel scales with context, so the window's advantage
  should grow with length. Long-context decode (≥ 4k) has not been measured. The kernel caps windows at 4096 slots.
- **Next lever:** batch the boundary rebuild (multi-token prefill in one submission) instead of 36 or 260 one-token
  submissions.
