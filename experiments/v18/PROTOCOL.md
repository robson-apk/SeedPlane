# V18 — native Vulkan decode for the SeedPlane Qwen2 graph

Protocol written before building or measuring the native engine, 2026-09-23.

## Motivation (measured before this protocol)

Profiling the V17 PyTorch-eager decoder on the Arc B580 (128-token greedy decode, prompt `9707,11,1879,0`):
18.11 tok/s (55.2 ms/token), ~1,189 non-view ATen ops per token, GPU busy time at most 4.7 ms/token (≤ 8.5%).
The 97 matrix-vector products take ~2.56 ms/token, close to the ~2.2 ms bandwidth floor of ~1 GB FP16 weights.
Conclusion: V17 is bound by per-op Python/launch overhead, not by arithmetic. PyTorch 2.8 XPU exposes no graph capture.

## Implementation under test

`native/vulkan_decode/`: C++ host + GLSL compute shaders (SPIR-V via glslc), MSVC build, Vulkan SDK 1.4.357.
Loads the same safetensors directly, FP16 weights, FP32 activations and FP32 KV cache. The whole per-token graph
(embedding, 24 layers, final norm, LM head, argmax, position increment) is recorded once into a single command buffer
and replayed with one queue submission per token; the host only reads back the chosen token.
Prompt tokens are fed through the same one-token graph (prefill speed is out of scope).

## Oracle

The V17 PyTorch engine on CPU FP32 (itself validated against Transformers at 2.29e-5 max logit error), same weights,
same prompt.

## Gates (fixed now)

- G1 correctness: the first 8 greedy tokens are identical to the oracle.
- G2 numerics: maximum absolute error of the last prompt-token logits vs the oracle is ≤ 0.10 (V17 B580 FP16: 8.81e-2).
- G3 speed: 128-token greedy decode on the B580 is **> 296 output tok/s** (the V17 target), median of 3 runs.

Also reported, not gated: first divergence index vs the oracle over 128 tokens, and ms/token.

If G3 fails, it is recorded as failed with the measured number; no gate is relaxed after the fact.
