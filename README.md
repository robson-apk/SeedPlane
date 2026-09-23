# SeedPlane: Spatial Text Diffusion Across Independent CPU Cores

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Hardware Agnostic](https://img.shields.io/badge/hardware-CPU%20%7C%20Intel%20Arc%20%7C%20CUDA-orange.svg)]()
[![Hadamard hypothesis](https://img.shields.io/badge/Hadamard%20routing%20hypothesis-rejected%20(V5%20self--audit)-red.svg)](experiments/v5/RESULTS.md)

> **"What if text generation didn't have to be sequential token-by-token? What if we could render text like a procedural video game world across independent CPU cores?"**

SeedPlane is an open-source research architecture exploring **spatial (sharded) text denoising across CPU worker processes**. It is a small toy prototype (~85k parameters, 1024-word vocabulary, single denoising step), not a production LLM.

---

> [!NOTE]
> ### 🔬 Scientific Status (V5 self-audit — not external peer review)
> This repository documents an open and transparent research trajectory:
> - **In V4:** We hypothesized that deterministic Hadamard orthogonal keys ($K_b \in \{-1, +1\}^D$) could act as a parameter-free coordination field to route boundary proposals without shared-memory locks.
> - **In V5 Audit (270,000 decisions, 180 paired runs):** We tested this hypothesis against the simplest possible engineering baseline: a lightweight message envelope with explicit identifiers (`request_id, generation_step, boundary_id`).
> - **The Finding:** The Hadamard hypothesis was **rejected**. Hadamard matching is equivalent to `boundary_src % 32 == boundary_dst % 32`: it admits modulo-32 collisions and accepts stale messages from the same boundary (120/120 in a live stale test). An exact envelope produced **identical outputs** (max difference 0) and no configuration met the pre-registered ≥10% end-to-end time gain for V5.
> - **Sharding speed (V5b, no injected jitter, this Mac, toy model):** at L=1024, 4 sharded workers took ~3.8 ms vs ~5.4 ms for a single unsharded forward with 4 threads (~29% less time, ~1.4x). At L=512 sharding was **slower** (~2.1 ms vs ~1.6 ms). Sharded inference uses block-local attention, so it does less work than the full forward; output quality vs a global model was not measured.
> 
> - **Checkpoint quality (found 2026-09-23):** the included toy checkpoint does **not use context**. Its masked-token loss is 5.06–5.13 nats at 15%, 50% and 90% masking, vs 5.10 for a unigram (word-frequency) model; accuracy ~7.5%. The boundary-NLL tables (V4, V5) were therefore measured on a context-blind model and are not evidence about seam quality. Routing-correctness and timing results do not depend on model quality. The same training setup (grad clipping at 1.0, 1/√d logit scale with N(0,1) embeddings) reproducibly stalls on the unigram plateau in V6 (`experiments/v6/PROTOCOL.md`, addenda 2–3). Check: `experiments/v6/check_original_checkpoint.py`.
>
> - **V6 (iterative sharded diffusion, pre-registered): INCONCLUSIVE.** A newly trained 5M-parameter denoiser did not use long-range context on TinyStories (global vs isolated shards on long-range tokens: +0.6 / +1.7 / −0.0 pp; required ≥2 pp), so whether neighbor-only halo exchange recovers distant information could not be tested. See `experiments/v6/RESULTS.md`.
>
> - **V7 (in progress):** synthetic long-range key–value task where distant information is *required* by construction, testing (a) the current token-halo design and (b) a latent-message variant where neighboring shards exchange one vector per round. Criteria are pre-registered in `experiments/v7/PROTOCOL.md`; results pending.
>
> Full paired benchmarks, unit tests, and replication scripts are archived in [**`experiments/v5/RESULTS.md`**](experiments/v5/RESULTS.md).

---

## ⚡ The Core Problem & The SeedPlane Approach

Modern LLMs are bound to an **$O(N)$ sequential memory wall**: generating token $t+1$ strictly requires waiting for token $t$. Distributing autoregressive generation across hardware nodes requires expensive, ultra-low-latency interconnects (InfiniBand/NVLink) to keep GPUs synchronized at every single token.

**SeedPlane investigates spatial text sharding:**
1. **Spatial Decomposition:** Text sequences are partitioned into autonomous spatial chunks (e.g., 128 tokens per core).
2. **Decoupling Semantics from Topology:** The language model focuses strictly on local denoising, while an external coordination protocol handles seam stitching ("halos").
3. **Adaptive Seam Deferral (`disagreement-defer`, idea):** When adjacent shards disagree on the overlapping seam, defer commitment and schedule a localized denoising pass. *Not implemented in the V5 runtime; no reproducible measurement of its effect is included in this repository.*
4. **Multi-Worker Execution:** Shards execute concurrently across persistent workers (see V5b timings above for where this helps and where it does not).

```
                         SeedPlane Coordination Plane
           K_1                         K_2                         K_3
            │                           │                           │
      ┌─────┴─────┐               ┌─────┴─────┐               ┌─────┴─────┐
      │   CORE 0  │ ── -K_1 ────► │   CORE 1  │ ── -K_2 ────► │   CORE 2  │
      │  [0..128] │ ◄── +K_1 ──── │ [128..256]│ ◄── +K_2 ──── │ [256..384]│
      └───────────┘               └───────────┘               └───────────┘
            │                           │                           │
            └─────────────►  Deterministic Blending  ◄──────────────┘
                                        │
                            Canonical Reconstructed Text
```

---

## 📊 Empirical Benchmarks

### 1. Sharded vs unsharded inference (V5b, pre-registered in `experiments/v5/PROTOCOL_V5b.md`)

Local CPU (Mac), toy checkpoint, no injected jitter, 20 paired runs × 3 seeds, randomized order. Median ms:

| L | Global forward, 1 thread | Global forward, 4 threads | Sharded, 4 workers (incl. IPC + fusion) | Sharded vs best global |
|:---:|:---:|:---:|:---:|:---:|
| 512 | ~3.26 | ~1.61 | ~2.07 | **slower** (−19% to −30%) |
| 1024 | ~11.85 | ~5.44 | ~3.78 | **~29% less time** (CI95 lower bound > 20% in all 3 seeds) |

Earlier versions of this README reported a "3.5x speedup" (41.9 → 11.95 ms, 1 → 4 workers). That comparison was **wrong as a speedup claim**: its baseline was the sharded pipeline on 1 worker, and each shard included an injected `sleep(uniform(0, 5 ms))` that was serialized on 1 worker and parallelized on 4. Actual compute per L=1024 batch is ~5.5 ms. Numbers kept in `experiments/v5/paired_runtime_results.json` for the record.

### 2. Boundary Noise Rejection (Context 1024)

V4 test (historical). Injected packets came only from *other, non-colliding* boundaries. It does **not** cover modulo-32 collisions or stale messages from the same boundary, which V4 accepts 100% of the time (see V5):

| Foreign Cross-Talk Noise | Unfiltered Baseline (Boundary NLL Drift) | **V4 Hadamard router (Boundary NLL Drift)** |
|---|:---:|:---:|
| **10% Corruption** | +0.0002964 | **+0.0000000 (100% Filtered)** |
| **25% Corruption** | +0.0009342 | **+0.0000000 (100% Filtered)** |
| **50% Corruption** | +0.0024969 | **+0.0000000 (100% Filtered)** |

---

## 🚀 1-Minute Quickstart

### Clone & Run the Interactive Demo
The standalone demo runs in pure Python with zero external dependencies:
```bash
git clone https://github.com/robson-apk/SeedPlane.git
cd SeedPlane
python3 demo.py
```

### Full PyTorch & Audit Suite
```bash
pip install -r requirements.txt

# V4 Hadamard routing test (loads checkpoints/clmp_parity_ctx1024.pt)
python3 experiments/v4/seedplane_hadamard_test_v4.py

# V5 audit: unit tests, paired runtime, report
python3 -m unittest discover -s experiments/v5 -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 python3 experiments/v5/paired_runtime.py
python3 experiments/v5/summarize.py
```

The experiments need the TinyStories validation text at `data/TinyStories-valid.txt` (not included; the tokenized cache is built in `data/` on first run).

---

## 📦 Checkpoints

| File | What it is | Uses context? |
|---|---|---|
| `checkpoints/clmp_parity_ctx1024.pt` | Original ~85k-param toy (V3/V4/V5) | **No** — masked loss equals the unigram baseline at every mask rate (`experiments/v6/check_original_checkpoint.py`) |
| `checkpoints/v6_mdlm_d256_l6_seed1.pt` | V6 denoiser, 5.3M params, same tokenizer/corpus | **Yes** — loss 1.70 at 15% masking vs 5.10 unigram |

The V6 model is not a size-matched replacement: it is ~60× larger and trained longer, so "better" here means "it learned", not "the architecture is better".

## 🔬 Lessons Learned & Open Questions

1. **Why Hadamard Keys Were Replaced by Envelopes:**
   * Computing $\max(0, -\cos(K_{\text{src}}, K_{\text{owner}}))$ on Sylvester Hadamard vectors is isomorphic to testing `boundary_src == boundary_target`.
   * Furthermore, fixed $H_{32}$ keys suffer from modulo collisions ($bid \pmod{32}$) and cannot detect temporal staleness (an outdated step from the same boundary).
   * Explicit message envelopes containing `(request_id, generation_step, boundary_id)` eliminate both issues at lower computational overhead.

2. **What Remains (with limits):**
   * **Sharding is faster only at longer contexts, on this toy model:** ~1.4x vs a 4-thread global forward at L=1024; slower at L=512. This is the known cost profile of block-local attention, not a new mechanism, and quality relative to a global model was not measured.
   * **Adaptive deferral:** earlier drafts cited "+16.97% seam coherence". Its source was later found in the predecessor prototype CLMP-dLM v3, where it was measured on 10 sequences (~340 boundary bytes). A paired re-test on 200 sequences with the same evaluation code gave **−0.2 to −0.4 pp** (95% CI includes 0) on both training seeds. The claim is withdrawn as not replicated.

---

## 📂 Repository Structure

```text
SeedPlane/
├── demo.py                  # Interactive zero-dependency visual demo
├── seedplane/               # Model and router code
│   ├── clmp_parity_seed_v3.py   # Parity denoiser + corpus/tokenizer loader
│   └── clmp_seed_router_v4.py   # V4 trainer and Hadamard router
├── checkpoints/
│   ├── clmp_parity_ctx1024.pt   # Original toy checkpoint (368 KB) — context-blind, kept for the record
│   └── v6_mdlm_d256_l6_seed1.pt # V6 masked-diffusion denoiser (21 MB, 5.3M params) — uses context
├── experiments/
│   ├── v4/                  # Original Hadamard tests (historical)
│   │   └── results/
│   ├── v5/                  # Self-audit: PROTOCOL*.md, RESULTS.md, scripts
│   │   └── results/         # Raw JSON results
│   ├── v6/                  # Iterative sharded diffusion (pre-registered; inconclusive)
│   └── v7/                  # Synthetic long-range test: token halo vs latent messages (in progress)
├── docs/                    # Historical V4 write-up
├── data/                    # Local corpus/cache (git-ignored)
├── requirements.txt
└── LICENSE
```

---

## 🤝 Citation & Community

```bibtex
@software{seedplane2026,
  author = {Robson},
  title = {SeedPlane: Spatial Text Diffusion Across Independent CPU Cores},
  url = {https://github.com/robson-apk/SeedPlane},
  year = {2026}
}
```

**License:** MIT License. Free for academic, personal, and commercial research.

---

## ☕ Support

If you find this research or code useful, you can support independent development here:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/robson.apk)
