# Night log — 2026-09-24

Autonomous session following `~/Downloads/plano para o seedplane` (critical order: V22 → X79 network → V23 RX570 → CI →
protocol → worker CLI → pool → M4 → auto scheduler → ...). Wake-up every 20 min (cron 7,27,47).

## Status
- [ ] V22 finished (G4 pending at start)
- [x] V22 finished and pushed (39c8e43): G1/G3/G4 pass, G2 FAIL (k40 0.9486x); V22b planned (GPU-exact threshold + restricted Gumbel)
- Infra: Mac M4 16 GB (Ethernet en0 10.0.0.92 active, 37 GB free disk). X79: Ubuntu 26.04, RX 570 RADV + llvmpipe, 60 GB RAM,
  gcc/clang/cmake/ninja/glslc present, **link 100 Mb/s** (needs cable/port fix — physical, not done). No sudo on X79;
  Khronos validation layer obtained from the LunarG SDK tarball in ~/sp_v23/vksdk.
- 00:40 user note: Codex is also working on this project. Policy: fetch before every push, never force, only touch my
  own dirs (5600X sp_v19/sp_vk/sp_prof, X79 sp_v23, branch native-runtime).
- [x] V23 RX 570: 9/9 gates pass (tokens identical, NLL Δ 2.4e-7, 0 validation errors, 116.8 tok/s); no 30 ms spikes on RX 570
- Mailbox with Codex: ~/.agent-mailbox/seedplane (README, CLAIMS, both outboxes, DECISIONS). Codex clone at
  ~/Downloads/LABORATORIO…/PREPARANDO PARA LANCAMENTO/SeedPlane (main, clean) — not touched.
- [x] macOS build via MoltenVK; Apple M4 smoke 128/128 + 296/296 identical to oracles (experiments/portability)
- [x] CI: native builds on Linux/macOS/Windows green (8178902); release.yml attaches zips on tags v*
- [ ] V22b pre-registered (2-level GPU threshold + restricted Gumbel); implementing

## 12:20 — Codex takeover validation checkpoint

- V22b G1: 47/48 cases pass. All 48 have 0 draws outside the reference support and 0 fallbacks. The lone failure is
  `synthetic_s2`, T=0.7, top-p=0.99: TV 0.0467298 at support 45,999. The expected empirical TV from multinomial noise
  is approximately 0.0467831; record the fixed pre-registered gate as failed, with the statistical-power limitation noted.
- V22b G2/G3/G4 pass on B580. Final fixed-mode graph G2 ratios: T1 1.0040x; k40+p0.9 0.9743x; p0.9 0.9832x;
  p0.99 0.9760x; k200 0.9727x. Greedy tokens match V20 in all three rounds. Session continuation (2,000 tokens across position 512)
  and native CLI convert plus two chat turns pass. Zero fallback on all runs.
- Final fixed-mode G2 matrices pass the 0.95x gate on M4/MoltenVK and RX570/RADV. M4 greedy median 64.877 tok/s; RX570
  greedy median 116.616 tok/s. RX570 device string confirms `AMD Radeon RX 570 Series (RADV POLARIS10)`, not llvmpipe.
- The same 200,000-draw k40+p0.9 sample on all three GPUs yields support 10, zero outside draws, and TV 0.00179688.
- Fixed-mode command graphs omit unused argmax/Gumbel/candidate dispatches; dynamic `--serve` keeps the full graph. An
  all-equal logits case produced 151,936 candidates and exercised the explicit fallback path with zero outside-support draws.
- Windows, macOS, and Linux CMake/MSVC builds include the new candidate sampler. Detailed data: `experiments/v22b/`.
- Safe isolated directories used: Windows `sp_v22b_codex`; X79 `~/sp_v22b_codex`. The pre-existing `sp_v19` and
  `sp_v23` directories were used read-only. B580 claim released after its tests.
- Next: communicate G1's statistical limitation, decide a prospective replacement protocol without changing V22b; verify
  V22b on more RX570 distributions; then continue cluster inference/data-plane integration. Distributed end-to-end LLM
  throughput gain has not yet been measured or established.
