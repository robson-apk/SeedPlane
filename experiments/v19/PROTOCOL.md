# V19 — native SeedPlane decode (shard-window attention) + native bundle converter

Protocol written before implementing or measuring, 2026-09-23.

## What changes vs V18

V18 decoded with full causal attention, ignoring the bundle's shard plan. V19 makes the native Vulkan engine
decode *the SeedPlane model*: the position-`t` token of shard `k = t // S` attends only to
`[sinks] + [halo of shard k-1] + [shard k up to t]` (exactly `ShardPlan.windows`), with original position ids.

Faithful semantics: in the reference definition (`engine.window_logits`), halo and sink K/V are computed *inside
window k*, not reused from window k-1. The engine therefore rebuilds its window-local KV cache at every shard boundary by
re-running the sink + halo tokens in the new window. KV memory is bounded by `sinks + halo + shard` slots.

Converter: `seedplane-bundle/2` = `seedplane.json` (architecture, plan, tensor table, hashes) + `weights.spw`
(engine-ready FP16 weights, fused QKV, FP32 norms/biases, 256-byte aligned) + tokenizer. It needs only
`safetensors` + `numpy`, and accepts an HF safetensors directory or a v1 bundle.

## Oracle

The V17 PyTorch `Qwen2Engine` on CPU FP32, fed each window from scratch (`positions` = original ids), which mirrors
`engine.window_logits`. Greedy generation in the oracle is recomputed per position from that definition.

## Fixed test

Qwen2.5-0.5B-Instruct, prompt `9707,11,1879,0`, plan S=64 / H=32 / sinks=4 (small so boundaries at 64, 128, 192, 256
are crossed), 296 generated tokens (positions 0..299). Logit dumps at positions 3, 63, 64, 65, 128, 200, 299.

## Gates (fixed now)

- G1 converter determinism: `weights.spw` built from the HF snapshot and from the existing v1 bundle
  (`seedplane_v12/qwen05.sp`, FP32) have identical SHA-256.
- G2 exactness under the plan: greedy tokens identical to the oracle through position 130 (first 127 generated
  tokens, two boundaries). First divergence over all 296 is reported.
- G3 numerics: max |logit error| vs the oracle at every dump position ≤ 0.10.
- G4 manipulation exercised: at position 65, max |logits(plan) − logits(full attention)| from the native engine
  > 0.01; and with the plan disabled (S ≥ length) the native tokens equal the V18 tokens (first 128).
  If G4 fails, G2/G3 are uninformative and the run is invalid.
- G5 speed: steady-state decode (submissions that are not boundary rebuilds) ≥ 245 tok/s (0.95 × V18 median 258.2).
  Overall tok/s including rebuilds, rebuild count and KV bytes are reported, not gated.

Failures are recorded as failures with the measured numbers.
