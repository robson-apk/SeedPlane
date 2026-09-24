# V23 — the same runtime and bundle on AMD RX 570 / RADV / Linux

Protocol written before building or running anything on the X79, 2026-09-24.

## Question

Does the unchanged native runtime (commit under test = the V22 code) run the unchanged seedplane-bundle/2 correctly on
a different vendor, driver, OS and subgroup size? The node is Xeon E5-2650 v2 + Radeon RX 570 8 GB (Polaris, RADV,
subgroup 64, no `shaderFloat16`), Ubuntu 26.04. The reference is the Intel Arc B580 on Windows (subgroup 32).

## Setup

- Clean checkout of the branch on the X79; CMake + Ninja; GCC and Clang builds.
- Bundle rebuilt on the X79 from the Hugging Face snapshot with `seedplane convert --native`. It must have the same
  `weights.spw` SHA-256 as on the 5600X (`ef9f3f59…cbe49`).
- The engine must report the RX 570, never `llvmpipe`.

## Gates (fixed now)

1. Build: GCC and Clang builds both succeed (CMake, Release).
2. Shaders: every shader compiles with the system `glslc`.
3. Validation layers: an 8-token run with `SP_VK_VALIDATE=1` reports zero validation errors, if the Khronos layer is
   installed; if it is absent, this gate is reported as *not executed*, not passed.
4. Device: every run reports `AMD Radeon RX 570` (RADV), and no run uses `llvmpipe`.
5. Tokens: with full attention, the first 8 and all 128 greedy tokens equal the V18 CPU FP32 oracle
   (`experiments/v18/cpu_fp32_oracle.json`). With the V19 plan (S64/H32/K4), all 296 tokens equal the V19 window oracle.
6. Logits: max |logit error| ≤ 0.10 at the last prompt token vs a CPU FP32 oracle computed on the X79 (PyTorch
   `Qwen2Engine`, same bundle weights).
7. NLL: on WikiText chunk 1 (V20 data, 4,096 tokens, default plan, shadow-batch), |mean NLL(RX570) − mean NLL(B580, V20)|
   ≤ 1e-3 nats and max per-position |Δ| ≤ 0.05.
8. Stability: 3 speed runs (V20 speed prompt, 1,024 tokens) with no crash or hang and identical greedy tokens across
   runs; these tokens also equal the B580 V20 shadow-batch tokens (reported; the gate is run-to-run identity).
9. Memory: VRAM in use (amdgpu sysfs) after run 3 is within 5% of after run 1.

Reported without a gate: tok/s (full vs SeedPlane), prefill time, sampling tok/s (3 configurations), p50/p99/max
latency, VRAM, load time, and whether the ~30 ms spikes seen on the B580 occur here.
