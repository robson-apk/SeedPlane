# V23 — the same runtime and bundle on AMD RX 570 / RADV / Linux: results

Protocol: [PROTOCOL.md](PROTOCOL.md), committed (39c8e43) before anything ran on the X79. Clean checkout of
`native-runtime` on the X79 (Xeon E5-2650 v2, Radeon RX 570 8 GB Polaris, Mesa RADV, Ubuntu 26.04, kernel 7.0).
The engine code is identical to V22 (39c8e43); the runner was added in c0f27c2. Raw data: `gates.json`.

## Verdict: 9/9 gates pass

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| G1 build | GCC and Clang (CMake + Ninja, Release) | both build | **pass** |
| G2 shaders | every shader compiles with system `glslc` | 21/21 SPIR-V | **pass** |
| G3 validation | 0 errors with `VK_LAYER_KHRONOS_validation` | 0 errors; the layer is confirmed loaded (see below) | **pass** |
| G4 device | RX 570 (RADV), never llvmpipe | `AMD Radeon RX 570 Series (RADV POLARIS10)` in every run | **pass** |
| G5 tokens | = V18 oracle (8 and 128, full attention); = V19 oracle (296, S64/H32/K4) | all identical | **pass** |
| G6 logits | max \|error\| ≤ 0.10 vs a CPU FP32 oracle computed on the X79 | 3.3e-5, same argmax | **pass** |
| G7 NLL | \|Δ mean\| ≤ 1e-3 and max per position ≤ 0.05 vs the B580 (V20) | Δ mean 2.4e-7; max 7.4e-5 (4,095 positions) | **pass** |
| G8 stability | 3 runs, identical tokens | identical; also identical to the B580 V20 tokens (1,024) | **pass** |
| G9 memory | VRAM after run 3 ≤ 1.05 × after run 1 (see method below) | 991.1 MiB in both windows (988 samples) | **pass** |

The bundle converted on Linux has the same `weights.spw` SHA-256 as the one converted on Windows (`ef9f3f59…cbe49`), and the
tokenized WikiText data files are byte-identical across the two machines.

## How G3 and G9 were measured (clarified after review by Codex)

- **G3:** the Khronos layer came from the LunarG 1.4.357.1 SDK tarball, extracted without root, via `VK_LAYER_PATH`.
  A missing VUID alone would only be indirect evidence, so the 8-token run was repeated with `VK_LOADER_DEBUG=layer`.
  The loader log shows `Insert instance layer "VK_LAYER_KHRONOS_validation"` and `Inserted device layer
  "VK_LAYER_KHRONOS_validation"`, with 0 `VUID` / `Validation Error` lines (`g3_layer_proof.txt`).
- **G9:** the runner samples amdgpu `mem_info_vram_used` every 50 ms while one process executes the 3 speed runs. It
  compares the peak in the 20–40% window of the process duration (early, after load) with the peak in the last 20%. This
  is a proxy for "after run 1" vs "after run 3", not two literal snapshots. Both peaks are 991.1 MiB. The engine
  allocates everything at load time, so no growth is expected; the check guards against leaks across runs.

## Reported (no gate)

| RX 570, 3,000-token prompt + 1,024 tokens | value |
|---|---:|
| SeedPlane plan (shadow-batch), median of 3 | **116.8 tok/s** |
| full attention (split kernel), median of 3 | 101.1 tok/s (SeedPlane 1.16× faster) |
| sampling T=0.7 k=40 p=0.9 / T=1.0 / T=0.7 p=0.9 | 116.1 / 114.7 / 116.1 tok/s (0.98–0.99×) |
| Clang build | 116.6 tok/s, tokens identical to GCC |
| prefill 3,000 tokens (batched) | 6.45 s |
| per-token latency p50 / p99 / max | 8.12 / 19.39 / 19.6 ms |
| load time | 3.56 s |
| subgroup size / shared memory | 64 / 64 KiB |
| KV per window cache | 19.0 MB |

For comparison, the B580 does 264 tok/s on the same workload, so the RX 570 runs at ~0.44× of it.

## Observations

- **The ~30 ms spikes seen on the B580 do not appear on the RX 570** (max/median 2.4×, against 8.5–9.9× on the B580).
  Same code, different OS/driver/GPU. This points the spikes away from the decode loop and toward the Windows/Intel side
  (driver, compositor, or another process using the B580). Inferred, not proven.
- **Sampling costs almost nothing on the RX 570** (≥ 0.98×), because the fixed sampling overhead is small next to its
  ~8 ms step. On the B580 the same overhead is ~4–5% of a 3.4 ms step.
- Numerics across vendors: subgroup 64 vs 32 changes reduction order, yet greedy tokens are identical for 1,024 tokens
  and NLL differs by < 1e-4 per position.

## Limits

- One AMD GPU generation (Polaris). No RDNA, no NVIDIA, no Apple Metal tested.
- The X79 link is 100 Mb/s. That does not matter for this single-node experiment, but it blocks the cluster
  experiments (V25+) until the cable/port is fixed.
