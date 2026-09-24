# Changelog

All notable changes are documented here. SeedPlane is currently alpha research software.

## Unreleased

- V22: sampling moved to the GPU. Temperature-only uses Gumbel-max (1% cost, was 43%); top-k/top-p use GPU statistics,
  a histogram threshold and a small candidate list (top-p 0.9: 0.96× greedy, was 0.67×). The pre-registered ≥0.95×
  gate fails for k=40/p=0.9 (0.9486×). Found afterwards: top-p 0.99 falls back to the host (116.8 tok/s). Fix planned
  as V22b.
- V21: the native runtime reads and writes text. It adds a C++ Qwen2 byte-level BPE tokenizer (NFC + split regex,
  identical to HF `tokenizers` on all of WikiText-2), temperature/top-k/top-p sampling, persistent chat sessions
  (continuation equals fresh prefill), `qwen_vk --chat` / `--serve`, `seedplane chat|generate --native`,
  `seedplane.native.NativeEngine`, and a CMake build. Sampling runs at 0.92× greedy (246 vs 267 tok/s), which fails the
  pre-registered 0.95× gate; the first run was 0.04× because of uncached readback memory, fixed post hoc.
- Added `seedplane-bundle/2` and `seedplane convert --native` / `python -m seedplane.native_bundle`: engine-ready FP16
  weights with fused QKV, FP32 norms/biases, architecture, shard plan and SHA-256 hashes; converts from a Hugging Face
  directory, a v1 bundle or a hub id using only numpy (V19 G1: identical weights from both sources).
- Added the native Vulkan decoder `native/vulkan_decode` (C++ + GLSL, one command-buffer submission per token). With full
  attention: 258.2 tok/s on the Arc B580 vs 18.11 tok/s for the PyTorch decoder, but below the pre-registered 296 tok/s
  (V18 G3 failed).
- The native decoder now runs the SeedPlane model: shard-window attention with a window-local KV cache, rebuilt at
  every shard boundary so decoding matches `ShardPlan.windows` exactly (V19: 296/296 tokens equal to a from-scratch
  window oracle). Boundary rebuilds currently make overall decode slower than full attention at short lengths.
- `Qwen2Engine.forward` accepts original `positions` for window-local caches.
- Native decoder V20: boundary modes `shadow-batch` (new default, exact), `shadow`, `rebuild-batch`, `rebuild` and
  approximate `reuse`; split (chunked) attention; K/V-only batched prompt prefill; teacher-forced NLL scoring
  (`--score-file`). At 3–4k context on the B580, exact SeedPlane decode is 266.9 tok/s vs 250.0 for full attention
  (V19 rebuild: 187.0, with 0.92 s boundary stalls). Split attention is 4.78× the old kernel at that context, and
  prompt prefill is 6.2× faster. Reuse was rejected (worse NLL than exact on all 3 chunks). The pre-registered
  max-latency gate failed in every mode, including full attention, because of ~30 ms spikes of unidentified origin.
- Added a SeedPlane-owned Qwen2/Qwen2.5 autoregressive graph loading safetensors directly, with persistent preallocated
  KV cache, fused QKV and gate/up projections, sampling and streaming chat.
- Added CPU and Arc B580 oracle validation: eight greedy tokens match Transformers; CPU FP32 maximum logit error is
  2.29e-5 and B580 FP16 maximum error is 8.81e-2.
- Raised the new B580 decoder from 1.40 to 17.57 output tok/s; this remains below the ~296 tok/s release target and is
  explicitly recorded as an incomplete performance milestone.

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
