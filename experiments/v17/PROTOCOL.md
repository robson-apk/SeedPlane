# V17 — independent SeedPlane autoregressive runtime

Protocol written before collecting results, 2026-09-23.

## Goal

Replace the llama.cpp model/runtime dependency with a SeedPlane-owned Qwen2 graph that loads safetensors directly,
maintains a persistent KV cache and generates tokens. Transformers is an offline correctness oracle only.

## Gates

1. Eight greedy decode tokens are identical to the oracle from the same prompt and weights.
2. Maximum last-token logit error is reported, not hidden behind token agreement.
3. Interactive generation uses the persistent cache and reports output tok/s separately from TTFT.
4. Performance claims require B580 measurements and must beat the existing ~296 tok/s decode baseline.

The portable PyTorch backend establishes correctness. It is not a performance victory; the optimized Vulkan backend
must pass the same token/logit checks before it can replace it.
