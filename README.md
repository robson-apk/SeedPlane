# SeedPlane: Spatial Text Diffusion Across Independent CPU Cores

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Hardware Agnostic](https://img.shields.io/badge/hardware-CPU%20%7C%20Intel%20Arc%20%7C%20CUDA-orange.svg)]()
[![Peer-Audited](https://img.shields.io/badge/audit-V5%20Verified-success.svg)](research_v5/RESULTS.md)

> **"What if text generation didn't have to be sequential token-by-token? What if we could render text like a procedural video game world across independent CPU cores?"**

SeedPlane is an open-source research architecture exploring **asynchronous, non-blocking spatial text diffusion across commodity hardware cores** without quadratic attention costs, centralized KV-caches, or sequential synchronization locks.

---

> [!NOTE]
> ### 🔬 Scientific Status & V5 Peer-Audit Update
> This repository documents an open and transparent research trajectory:
> - **In V4:** We hypothesized that deterministic Hadamard orthogonal keys ($K_b \in \{-1, +1\}^D$) could act as a parameter-free coordination field to route boundary proposals without shared-memory locks.
> - **In V5 Audit (270,000 decisions, 180 paired runs):** We tested this hypothesis against the simplest possible engineering baseline: a lightweight message envelope with explicit identifiers (`request_id, generation_step, boundary_id`).
> - **The Finding:** While multi-core spatial sharding **successfully delivers a 3.5x wall-clock speedup** on commodity CPUs (scaling from ~41 ms down to ~12 ms across 4 workers), the Hadamard tensor matching was found to be mathematically isomorphic to integer boundary matching, while admitting modulo-32 collisions and missing temporal staleness. The exact envelope baseline matched V5's output with **0.0000000 difference** at lower routing overhead.
> 
> Full paired benchmarks, unit tests, and replication scripts are archived in [**`research_v5/RESULTS.md`**](research_v5/RESULTS.md).

---

## ⚡ The Core Problem & The SeedPlane Approach

Modern LLMs are bound to an **$O(N)$ sequential memory wall**: generating token $t+1$ strictly requires waiting for token $t$. Distributing autoregressive generation across hardware nodes requires expensive, ultra-low-latency interconnects (InfiniBand/NVLink) to keep GPUs synchronized at every single token.

**SeedPlane investigates spatial text sharding:**
1. **Spatial Decomposition:** Text sequences are partitioned into autonomous spatial chunks (e.g., 128 tokens per core).
2. **Decoupling Semantics from Topology:** The language model focuses strictly on local denoising, while an external coordination protocol handles seam stitching ("halos").
3. **Adaptive Seam Deferral (`disagreement-defer`):** When adjacent shards disagree on the overlapping seam, rather than trusting raw confidence, the system defers commitment and schedules a quick localized denoising pass (+16.97% boundary coherence).
4. **Multi-Worker Scaling:** Shards execute concurrently across persistent workers, achieving linear scaling on standard multi-core CPUs.

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

### 1. Multi-Core CPU Scaling (Sequence Length L=1024)

Empirical runtime measured across persistent processes on commodity CPU threads:

| Workers | Median Latency (Envelope Baseline) | Median Latency (SeedPlane V5) | Measured Speedup |
|:---:|:---:|:---:|:---:|
| **1 Worker** | ~41.89 ms | ~40.75 ms | 1.00x (Baseline) |
| **2 Workers** | ~20.60 ms | ~20.53 ms | **2.03x** |
| **4 Workers** | ~11.95 ms | ~12.56 ms | **3.50x** |

*Both methods share the identical underlying speedup from spatial parallelism. The difference in final generated probabilities between exact envelope routing and SeedPlane V5 was **0.0000000**.*

### 2. Boundary Noise Rejection (Context 1024)

Under simulated multi-worker latency where foreign tokens and corrupt boundary proposals were injected:

| Foreign Cross-Talk Noise | Unfiltered Baseline (Boundary NLL Drift) | **SeedPlane / Exact Envelope (Boundary NLL Drift)** |
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

# Run the core Hadamard tests
python3 seedplane_hadamard_test_v4.py

# Run the V5 paired audit suite (reproduces 180 comparisons)
OPENBLAS_NUM_THREADS=1 python3 research_v5/paired_runtime.py
python3 research_v5/summarize.py
```

---

## 🔬 Lessons Learned & Open Questions

1. **Why Hadamard Keys Were Replaced by Envelopes:**
   * Computing $\max(0, -\cos(K_{\text{src}}, K_{\text{owner}}))$ on Sylvester Hadamard vectors is isomorphic to testing `boundary_src == boundary_target`.
   * Furthermore, fixed $H_{32}$ keys suffer from modulo collisions ($bid \pmod{32}$) and cannot detect temporal staleness (an outdated step from the same boundary).
   * Explicit message envelopes containing `(request_id, generation_step, boundary_id)` eliminate both issues at lower computational overhead.

2. **What Remains Strong and Valid:**
   * **Spatial text sharding works:** Decoupling sequences into autonomous fragments and denoising them concurrently achieves near-linear speedups on commodity multi-core CPUs.
   * **Adaptive Deferral works:** Deferring boundary resolution when workers disagree produces a measurable +16.97% boost in seam coherence.

---

## 📂 Repository Structure

```text
SeedPlane/
├── demo.py                          # Interactive zero-dependency visual demo
├── seedplane_hadamard_test_v4.py     # Main Hadamard routing & stress test script
├── seed_router_robust_v4.py          # Cross-talk resilience evaluator
├── seed_fusion_ablation_v4.py        # Boundary fusion & Product-of-Experts ablations
├── clmp_seed_router_v4.py            # Trainer and router module
├── clmp_parity_seed_v3.py            # Parity denoiser baseline
├── clmp_parity_ctx1024.pt            # Pre-trained checkpoint (368 KB)
├── research_v5/                      # The V5 Peer-Audit Suite
│   ├── RESULTS.md                    # Detailed audit write-up & falsification data
│   ├── PROTOCOL.md                   # Strict evaluation criteria
│   ├── paired_runtime.py             # 180-run paired benchmark
│   ├── live_stale_test.py            # Real concurrent stale-inference test
│   └── summarize.py                  # Report generator & bootstrap CI calculator
├── requirements.txt                  # Minimal dependencies
├── LAUNCH_KIT.md                     # Ready-to-publish posts (HN, Reddit, X)
└── LICENSE                           # MIT License
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
