# V17 — independent SeedPlane autoregressive runtime (foundation)

This is a correctness milestone, **not yet a decode-speed victory**.

## Correctness

SeedPlane now loads Qwen2.5 safetensors directly and implements RMSNorm, RoPE, GQA attention, SwiGLU, LM head,
persistent KV cache, sampling and streaming without `AutoModelForCausalLM`, `llama_model` or `llama_decode`.

Against Transformers as an offline oracle:

- CPU FP32 maximum last-token logit error: `2.29e-5`.
- Arc B580/XPU FP16 maximum last-token logit error: `8.81e-2`.
- Eight greedy decode tokens: identical on CPU and B580.
- Cache length after a four-token prompt plus eight decode steps: 12.

## Arc B580 decode

| Implementation stage | 128-token decode | Change |
|---|---:|---:|
| Growing KV with concatenation | 1.40 tok/s | baseline of the new portable graph |
| Preallocated KV cache | 15.31 tok/s | **10.96×** |
| Preallocated KV + fused QKV and gate/up projections | **17.57 tok/s** | **12.58×** total |
| Existing optimized reference runtime | ~296 tok/s | target still not reached |

`torch.compile` was also tested, but PyTorch 2.8 XPU on Windows failed in Inductor with an `UntypedStorage` weak-reference
compiler error. The portable backend launches too many small XPU operations per token; further Python-level tuning will
not close a 16.8× gap. The next implementation layer is a native SeedPlane graph using attributed MIT-licensed
SYCL/Vulkan tensor kernels, while keeping model loading, graph construction, KV policy, sampling and APIs in SeedPlane.

Raw timings are preserved in this directory. Performance claims remain blocked until the native backend beats 296
output tok/s with the same model and greedy token sequence.
