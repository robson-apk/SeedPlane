# V18 — native Vulkan decode: results

Protocol: [PROTOCOL.md](PROTOCOL.md) (written before building or measuring). Arc B580, Qwen2.5-0.5B-Instruct safetensors,
prompt `9707,11,1879,0`, 128-token greedy decode, 3 runs. Raw data: `b580_vk_decode.json`, `cpu_fp32_oracle.json`.

## Verdict

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| G1 | first 8 greedy tokens = CPU FP32 oracle | identical in 3/3 runs | **pass** |
| G2 | max \|logit error\| ≤ 0.10 on the last prompt token | 2.29e-5 (mean 3.84e-6) | **pass** |
| G3 | median decode > 296 tok/s | **258.2 tok/s** (258.2 / 260.7 / 247.3) | **fail** |

G3 failed: the native engine is 14.3× faster than the V17 PyTorch decoder (18.11 tok/s measured the same day) but
still 13% below the pre-registered 296 tok/s target.

## Also reported

- All 128 generated tokens are identical to the CPU FP32 oracle in all 3 runs (no divergence).
- 3.84–4.04 ms/token; 147 compute dispatches per token in one command buffer, one submission per token.
- FP16 weight conversion: 0 overflows, 716 nonzero BF16 weights flushed to zero (below FP16 subnormal range).
- The G2 maximum error (2.288818e-5) equals, to the printed digits, V17's CPU-vs-Transformers error. Both are small
  multiples of the FP32 ulp at logit magnitude ~16–32, so identical maxima are plausible; this was not investigated further.
- The FP32 KV cache is 2× the memory of an FP16 cache; its speed cost at 132 positions is expected to be negligible (not measured).

## Why it matters

The V17 profile predicted that removing per-op overhead would land the same work at ~210–300 tok/s. The measurement
(258 tok/s) falls inside that prediction. The remaining gap to the ~2.2 ms/token weight-bandwidth floor
(3.87 vs ~2.2 ms) has not been decomposed yet; per-dispatch GPU timestamps are the next measurement.
