# SeedPlane: Spatial Text Diffusion Across Independent CPU Cores

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/hardware-CPU%20%7C%20Intel%20Arc%20%7C%20Apple%20Silicon-orange.svg)]()
[![Pre-registered](https://img.shields.io/badge/science-pre--registered%20%7C%20falsifications%20kept-8A2BE2.svg)](experiments/)
[![Zero invalid messages](https://img.shields.io/badge/router-0%20invalid%20accepted%20of%2054k%20adversarial%20msgs-success.svg)](experiments/v5/RESULTS.md)

> **"What if text generation didn't have to be sequential token-by-token? What if we could render text like a procedural video game world — each CPU core painting its own region of the page?"**

SeedPlane splits a text sequence into **spatial shards** (128 tokens each), denoises them **in parallel on independent workers**, and stitches the seams with a lightweight coordination protocol. It is an open research lab built and measured entirely on **commodity hardware** (Ryzen 5600X · Intel Arc B580 · Apple M4) — and every claim in this repository was **pre-registered before the numbers were seen**.

---

## ✨ Highlights

| | Result | Evidence |
|---|---|---|
| ⚡ | **~29% faster than the best single-process forward pass** at L=1024 — 4 sharded workers (incl. IPC + fusion) vs a 4-thread global forward. 95% CI lower bound > 20% in all 3 seeds. | [`experiments/v5/PROTOCOL_V5b.md`](experiments/v5/PROTOCOL_V5b.md) |
| 🛡️ | **Zero invalid messages accepted** by the versioned envelope router across 54,000 adversarial test messages (270,000 decisions over 5 compared routers): collisions, stale generations, wrong request/version/target/source, duplicates. | [`experiments/v5/RESULTS.md`](experiments/v5/RESULTS.md) |
| ⏱️ | **0 / 120 stale inferences accepted** in a live concurrent test with *real* out-of-date model outputs racing current ones — and 120 / 120 current ones kept. | `experiments/v5/results/live_stale_results.json` |
| 🎯 | **Bit-identical outputs** between the exact-envelope router and the full SeedPlane V5 router over 180 paired end-to-end runs (max difference 0.0). | `experiments/v5/results/paired_runtime_results.json` |
| 📈 | **More accurate than full attention on TinyStories**: SeedPlane-style shard decoding scored +0.6 pp over the same model with global attention, in all 3 seeds (95% CI excludes 0). | [Comparison](#-seedplane-vs-a-traditional-transformer) |
| 🧠 | **A denoiser that actually learns**: the V6 model (5.3M params) reaches 1.70 nats at 15% masking vs 5.10 for a word-frequency baseline — shipped in `checkpoints/`. | [`experiments/v6/`](experiments/v6/) |
| 🔬 | **A reusable "is my model really using context?" check** that caught a context-blind checkpoint other metrics had missed — one script, three numbers. | [`experiments/v6/check_original_checkpoint.py`](experiments/v6/check_original_checkpoint.py) |
| 🧪 | **Science in the open**: protocols with fixed pass/fail criteria before every run, failed trainings and retracted claims kept in history instead of deleted. | `experiments/*/PROTOCOL.md` |
| 💻 | **Runs anywhere**: zero-dependency demo in pure Python; full suite on PyTorch CPU, Intel XPU (SYCL) or Apple Silicon. | [Quickstart](#-1-minute-quickstart) |

---

## 🥊 SeedPlane vs a Traditional Transformer

Same model weights, same data, same decoding rule — only the attention layout changes: a **traditional Transformer** attends over the whole sequence at once; **SeedPlane** splits it into 128-token shards that each see only their own shard plus a 16-token halo from each neighbor.

| | Traditional (global attention) | **SeedPlane (shards + halo 16)** | |
|---|:---:|:---:|:---:|
| ⚡ Wall-clock, L=1024 (4 CPU threads vs 4 workers, incl. IPC + fusion) | 5.44 ms | **3.78 ms** | 🏆 **SeedPlane, ~29% faster** |
| 🎯 Masked-infill accuracy, L=1024 (3 seeds, ~33k tokens each) | 45.02 / 44.62 / 45.61 % | **45.67 / 45.29 / 46.20 %** | 🏆 **SeedPlane, +0.6 pp** (95% CI excludes 0 in every seed) |
| 🧮 Attention pairs computed, L=1024 | 1,048,576 | **163,840** | 🏆 **SeedPlane, 6.4× less work** |
| 📈 Attention pairs, L=8192 | 67,108,864 | **1,310,720** | 🏆 **SeedPlane, 51× less** — the gap grows linearly with length |
| 🌐 Scale-out | one device must hold the whole sequence | each worker holds one shard; neighbors exchange a thin halo | 🏆 **SeedPlane** |
| ⏱️ Wall-clock, L=512 | **1.61 ms** | 2.07 ms | Traditional (coordination overhead dominates short inputs) |
| 🔭 Information from far-away shards | sees everything | only what reaches it through neighbors | ⏳ Under test in **V7** |

<sub>Accuracy: V6 denoiser (5.3M params), TinyStories, 50% masked, 16 iterative steps (`experiments/v6/results/analysis.json`, criterion H1a). The model was trained half on full sequences and half on shard windows; TinyStories stories are short, so far-away context carries little information there — V7 is the stress test for that. Timing: V5b pre-registered benchmark on the original toy model (`experiments/v5/PROTOCOL_V5b.md`). Attention pairs are exact arithmetic (L² vs shards × 128 × 160).</sub>

---

## ⚡ The Idea

Autoregressive LLMs generate token *t+1* only after token *t*. Distributing that across machines means synchronizing every token over fast interconnects. SeedPlane explores a different shape of computation:

1. **Spatial decomposition** — the sequence is partitioned into shards of 128 tokens.
2. **Independent local denoising** — each worker only sees its own shard plus a thin halo from its neighbors, so attention cost stays local.
3. **Coordination protocol, not a bigger model** — boundary proposals travel in versioned envelopes `(request, generation, boundary, model_version, target, source)` with deduplication, so late, duplicated or foreign messages never corrupt the canonical state.
4. **Parallel workers** — shards run concurrently on persistent processes; the longer the sequence, the more the local-attention savings pay for the coordination overhead.

```
            request r · generation g  (canonical global state)
      ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
      │   SHARD 0    │ ◄────► │   SHARD 1    │ ◄────► │   SHARD 2    │ ◄──► …
      │   [0..128)   │  halo  │  [128..256)  │  halo  │  [256..384)  │
      └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
             │    envelope: (r, g, boundary, version, target, source) + dedup
             └──────────────► validated fusion into the canonical state ◄──────┘
```

---

## 📊 Benchmarks

### Sharded vs unsharded inference (V5b, pre-registered)

Local CPU, 20 paired runs × 3 seeds, randomized order, no injected delays. Median ms:

| L | Global forward, 1 thread | Global forward, 4 threads | **Sharded, 4 workers** (incl. IPC + fusion) | Sharded vs best global |
|:---:|:---:|:---:|:---:|:---:|
| 512 | 3.26 | 1.61 | 2.07 | slower (overhead-bound) |
| **1024** | 11.85 | 5.44 | **3.78** | **~29% less time** |

Sharding pays off as sequences grow: the break-even sits between 512 and 1024 tokens on this setup.

### Routing integrity (V5, 3 seeds × 6,000 messages per fault type)

| Fault injected | Hadamard keys (V4) | Boundary ID only | **Versioned envelope (V5)** |
|---|:---:|:---:|:---:|
| Foreign boundary | rejected | rejected | **rejected** |
| Modulo-32 collision | accepted | rejected | **rejected** |
| Stale generation | accepted | accepted | **rejected** |
| Wrong request / version / target / source | accepted | accepted | **rejected** |
| Duplicate | accepted | accepted | **rejected** |

---

## 🧭 Research Scoreboard

Every experiment has a protocol written before the run. Outcomes are recorded as they came out — including the ones that did not go our way.

| Version | Question | Outcome |
|---|---|---|
| V4 | Do Hadamard orthogonal keys route boundary messages? | ❌ Equivalent to `boundary_id % 32` matching; misses collisions and staleness → replaced by envelopes |
| V5 | Does the envelope router beat plain IDs + version on time? | ✅ Correctness: zero invalid accepted · ❌ No ≥10% end-to-end time gain (identical outputs) |
| V5b | Is sharded inference faster than a global forward? | ✅ ~29% at L=1024 · ❌ slower at L=512 |
| — | Earlier README claims ("3.5x", "+16.97% seam coherence") | ❌ Retracted: the first measured injected sleeps; the second did not replicate on 200 sequences |
| — | Does the original toy checkpoint use context? | ❌ No (loss = unigram); V4/V5 quality tables are not evidence about seams. Fixed by the V6 model |
| V6 | Do neighbor-only halos recover distant info on TinyStories? | ⚪ Inconclusive — TinyStories has too little long-range dependency to test it |
| **V7** | Same question on a task where distant info is **required**: token halos vs latent messages between neighbors | ⏳ **Running.** The reference model already solves the task at 100% at every distance; the verdict will land in `experiments/v7/RESULTS.md` |

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
