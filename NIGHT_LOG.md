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
