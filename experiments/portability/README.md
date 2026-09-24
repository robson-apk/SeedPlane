# Portability smoke checks (not pre-registered experiments)

Short correctness checks outside the V-series gates. Pre-registered cross-vendor evidence is in `experiments/v23`
(AMD RX 570 / RADV / Linux, 9/9 gates).

## Apple M4 / MoltenVK / macOS 27 (2026-09-24)

- Build: CMake + Ninja with Homebrew `molten-vk`, `vulkan-loader`, `vulkan-headers`, `shaderc` (21 SPIR-V shaders).
- The engine needed `VK_KHR_portability_enumeration` (instance) and `VK_KHR_portability_subset` (device). Both are now
  enabled automatically when available. After the change, Windows/B580 and Linux/RX 570 reproduce the V19 oracle.
- Bundle converted on macOS: `weights.spw` SHA-256 `ef9f3f59…cbe49`, identical to Windows and Linux.
- Device: `Apple M4`, subgroup 32, 32 KiB shared memory.
- Full attention, 128 greedy tokens: **identical to the V18 CPU FP32 oracle** (73.0 tok/s, short prompt).
- SeedPlane plan S64/H32/K4, 296 tokens: **identical to the V19 window oracle** (70.7 tok/s).
- Only these two short runs were done on the Mac, following the lab rule of no heavy inference on the Mac.
  Raw: `macos_m4_full128.json`, `macos_m4_plan296.json`.
