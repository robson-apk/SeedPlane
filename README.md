# SeedPlane: Asynchronous Text Diffusion Across CPU Cores via Hadamard Coordination Codes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Hardware Agnostic](https://img.shields.io/badge/hardware-CPU%20%7C%20Intel%20Arc%20%7C%20CUDA-orange.svg)]()
[![Zero-CrossTalk](https://img.shields.io/badge/crosstalk%20degradation-0.00000%20NLL-success.svg)]()

> **"What if text generation didn't have to be sequential token-by-token? What if we could render text like a procedural video game world across independent CPU cores?"**

SeedPlane is a research architecture and distributed execution protocol that enables **asynchronous, non-blocking text diffusion across independent hardware cores** without quadratic attention costs, centralized KV-caches, or communication bottlenecks.

---

## ⚡ The Breakthrough in 30 Seconds

Modern LLMs are trapped in an **$O(N)$ sequential memory wall**: generating token $t+1$ strictly requires waiting for token $t$. Distributing this across GPUs requires ultra-low-latency InfiniBand/NVLink networks to synchronize every single step.

**SeedPlane takes a completely different path:**
1. **Spatial Text Sharding:** The sequence is partitioned into independent spatial chunks (e.g. 128 tokens per core).
2. **The Seed is NOT an Embedding:** Rather than cluttering the latent space with coordinates, the seed acts as an **algebraic coordination plane**.
3. **Zero-Cost Hadamard Orthogonal Codes:** Each boundary $b$ between cores receives a deterministic Hadamard key $K_b \in \{-1, +1\}^D$:
   - Owner core holds $+K_b$
   - Borrowed halo proposal carries $-K_b$
   - Affinity score: $\max\left(0, -\cos(K_{\text{source}}, K_{\text{owner}})\right)$
4. **Zero Cross-Talk Degradation:** In asynchronous stress tests with **50% corrupted / stale packets**, SeedPlane achieved **+0.0000000 boundary NLL delta** (100% rejection of foreign boundary noise) at a matching latency of **~0.006 ms**.

```
                         SeedPlane (Hadamard Routing)
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

## 📊 Empirical Results

### 1. Robustness Against Asynchronous Cross-Talk (Context 1024)

Under simulated multi-worker latency where boundary proposals were corrupted, delayed, and foreign tokens were injected:

| Foreign Cross-Talk Noise | Unseeded Baseline (Boundary NLL Drift) | **SeedPlane Router (Boundary NLL Drift)** |
|---|:---:|:---:|
| **10% Corruption** | +0.0002964 | **+0.0000000 (100% Filtered)** |
| **25% Corruption** | +0.0009342 | **+0.0000000 (100% Filtered)** |
| **50% Corruption** | +0.0024969 | **+0.0000000 (100% Filtered)** |

*Because Hadamard keys for distinct boundaries are strictly orthogonal ($\cos(K_i, K_j) \approx 0$), foreign packets receive zero compatibility and are annihilated before touching the denoiser.*

### 2. Multi-Core CPU Scaling vs Global Forward Pass

Benchmark running on commodity CPU threads (no GPU required):

| Sequence Length | Global Sequential Forward | 4 Shards Parallel | Measured Speedup |
|---|:---:|:---:|:---:|
| **1,024 tokens** | 5.12 ms | 9.37 ms | 0.55x (overhead bound) |
| **4,096 tokens** | 17.71 ms | 12.01 ms | **1.47x** |
| **8,192 tokens** | 48.54 ms | 18.08 ms | **2.68x** |
| **16,384 tokens** | 94.25 ms | 25.40 ms | **3.71x** |

---

## 🚀 1-Minute Quickstart

### Prerequisites
Clone this repository:
```bash
git clone https://github.com/robson-apk/SeedPlane.git
cd SeedPlane
```

### Run the Standalone Interactive Demo (Zero Dependencies!)
You don't even need PyTorch installed to see the core algebra and live multi-core simulation:
```bash
python3 demo.py
```

### Full PyTorch Verification
```bash
pip install -r requirements.txt
python3 seedplane_hadamard_test_v4.py
```

---

## 🔬 Architecture Details

### The Deferral Scheduler (`disagreement-defer`)
In diffusion models, when two neighboring shards disagree about what belongs in the overlapping boundary halo, naive systems lock tokens based on raw confidence. SeedPlane uses an adaptive agreement protocol:
* If both cores agree: token is immediately committed.
* If cores disagree: the token is **deferred**, scheduling an additional localized denoising pass only on the seam.
* **Empirical gain:** **+16.97%** relative improvement in boundary coherence.

### The Hadamard Matrix Generator
SeedPlane constructs deterministic Sylverster Hadamard matrices recursively:
```python
def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
```
At `KEY_DIM = 32`, 32 orthogonal keys cover up to 4,096 tokens across shards of 128 before key hierarchy is needed. Matching 32 boundaries takes **0.006 milliseconds**.

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
├── requirements.txt                  # Minimal dependencies
├── LAUNCH_KIT.md                     # Ready-to-publish posts (HN, Reddit, X)
└── LICENSE                           # MIT License
```

---

## 🗺️ Roadmap

- [x] **Phase 1:** Mathematical proof of concept & zero-cost Hadamard key routing.
- [x] **Phase 2:** Multi-core CPU shard benchmarks & cross-talk stress tests.
- [ ] **Phase 3:** True asynchronous worker runtime (Ray / multiprocessing) with zero shared memory locks.
- [ ] **Phase 4:** Subword BPE scaling to 50M+ parameter denoisers.
- [ ] **Phase 5:** Native Intel Arc (SYCL/IPEX) & Metal (Apple Silicon) multi-device distributed mesh.

---

## 🤝 Citation & Community

If you find this concept interesting or build upon it, please cite or star the repo:

```bibtex
@software{seedplane2026,
  author = {Robson},
  title = {SeedPlane: Asynchronous Text Diffusion Across CPU Cores via Hadamard Coordination Codes},
  url = {https://github.com/robson-apk/SeedPlane},
  year = {2026}
}
```

**License:** MIT License. Free for academic, personal, and commercial research.
