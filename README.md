<h1 align="center">SeedPlane</h1>

<p align="center">
  <b>What if AI stopped thinking in single file?</b><br>
  Split the page into shards. Let every CPU core write its own part — at the same time.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT"></a>
  <a href="experiments/v8/RESULTS.md"><img src="https://img.shields.io/badge/V8-2%C3%97%20faster%20%7C%20same--or--better%20quality-success.svg" alt="2x faster, same-or-better quality"></a>
  <a href="experiments/"><img src="https://img.shields.io/badge/science-pre--registered-8A2BE2.svg" alt="pre-registered"></a>
  <img src="https://img.shields.io/badge/runs%20on-CPU%20%7C%20Intel%20Arc%20%7C%20Apple%20Silicon-orange.svg" alt="hardware">
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/hero-race-dark.gif">
    <img src="docs/img/hero-race-light.gif" alt="Real decoding of the same 1,024-token page: SeedPlane finishes in about half the time of a traditional Transformer" width="760">
  </picture>
</p>

<div align="center">

| 🚀 **2.0× faster** | ⚡ **682 vs 341 tok/s** | 🎯 **+0.4 to +1.4 pp accuracy** | 🧮 **6.4× less attention work** |
|:---:|:---:|:---:|:---:|
| 760 ms vs 1,555 ms per 1,024-token page | 6 CPU cores; 1.6× faster even on 1 core | same model, same text, 3/3 seeds | 51× less at 8,192 tokens |

<sub>Same checkpoint · same inputs · same CPU cores (Ryzen 5600X) · criteria committed to git <i>before</i> each run — <a href="experiments/v8/RESULTS.md">V8</a> · <a href="experiments/v9/RESULTS.md">V9</a></sub>

</div>

---

## The blank page isn't a queue. It's a territory.

Language models write the way we have since Gutenberg: one word, then the next, then the next. **SeedPlane borrows a trick from open-world game engines instead** — split the world into chunks and let every core render its own chunk at once.

- **Spatial shards** — the page is cut into 128-token shards, spread across the cores.
- **Parallel refinement** — every shard fills in its hidden words over 16 steps, all shards at the same time.
- **A thin seam** — each shard reads a 16-token halo from its neighbours; a versioned message envelope makes sure no late, duplicated or foreign update ever lands.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/how-it-works-dark.gif">
    <img src="docs/img/how-it-works-light.gif" alt="Four shards of the same page filling their hidden words in parallel, real model output" width="760">
  </picture>
</p>

A traditional Transformer makes every word attend to every other word — cost grows with the square of the length. SeedPlane only looks inside the shard and its halo, so the work grows linearly and splits cleanly across cores.

---

## Results

### Faster, without getting worse

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/quality-dark.svg">
    <img src="docs/img/quality-light.svg" alt="Accuracy per seed: SeedPlane is 0.4 to 1.4 points above the traditional Transformer" width="720">
  </picture>
</p>

### More tokens per second on every core count

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/tokens_per_second-dark.svg">
    <img src="docs/img/tokens_per_second-light.svg" alt="Tokens per second from 1 to 6 CPU cores: SeedPlane 247 to 682, traditional 154 to 341" width="720">
  </picture>
</p>

**1.6× faster on a single core, 2.0× on six.** The traditional Transformer stops scaling around 5 cores; SeedPlane's shards are independent, so each core gets its own work.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/scaling-dark.svg">
    <img src="docs/img/scaling-light.svg" alt="Speed-up over one core: SeedPlane reaches 2.7x, traditional peaks at 2.4x and drops at 6 cores" width="720">
  </picture>
</p>

### The price: memory

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/memory-dark.svg">
    <img src="docs/img/memory-light.svg" alt="Peak RAM: traditional stays near 400 MB, SeedPlane grows from 700 MB to 2.3 GB with 6 cores" width="720">
  </picture>
</p>

Today every SeedPlane core is a separate process with its own copy of the runtime and model (~330 MB each). The compute is cheap; the duplication is not — sharing one copy across cores is the next optimization.


### Where it loses — on purpose, in public

When the answer sits several shards away, a shard simply cannot see it. We built a task that forces exactly that:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/long_range-dark.svg">
    <img src="docs/img/long_range-light.svg" alt="Recall of information two or more shards away: traditional 100%, SeedPlane at chance" width="720">
  </picture>
</p>

**SeedPlane shines when context is local** (like the short stories we measured on) and is **not** a drop-in replacement when a text must connect ideas far apart. Closing that gap is the next experiment.

---

## Science in the open

No hidden footnotes: every experiment has its pass/fail criteria written *before* the numbers are seen, and every miss stays in the history.

| | Question | Verdict |
|---|---|---|
| **V8** | Same model — faster *and* at least as good? | ✅ 2.0× faster · +0.4 to +1.4 pp · lower loss |
| **V9** | Faster than the traditional Transformer on every core count, 1 → 6? | ✅ 1.6× to 2.0× faster, identical output to V8 · ❌ optimizations gained only 4–6% (10% needed) · ⚠️ uses 1.7–5.8× more RAM |
| V7 | Can shards recall far-away information? | ❌ not with halos or latent messages (yet) |
| V6 | Same question on TinyStories | ⚪ inconclusive — too little long-range signal |
| V5 | Is the message router safe? | ✅ 0 invalid updates accepted out of 54,000 adversarial ones |
| V4 | Do Hadamard keys route messages? | ❌ equivalent to plain IDs → replaced |
| — | Early claims ("3.5×", "+16.97%") | ❌ retracted after re-testing |

<details>
<summary><b>More detail: routing safety, attention cost, checkpoints, lessons</b></summary>

**Routing integrity (V5, 3 seeds × 6,000 messages per fault type)**

| Fault injected | Hadamard keys (V4) | Boundary ID only | **Versioned envelope (V5)** |
|---|:---:|:---:|:---:|
| Foreign boundary | rejected | rejected | **rejected** |
| Modulo-32 collision | accepted | rejected | **rejected** |
| Stale generation | accepted | accepted | **rejected** |
| Wrong request / version / target / source | accepted | accepted | **rejected** |
| Duplicate | accepted | accepted | **rejected** |

Live concurrency test: **0 / 120** stale model outputs accepted, **120 / 120** current ones kept.

**Attention pairs computed** (exact arithmetic): L=1024 → 1,048,576 global vs 163,840 SeedPlane (6.4×) · L=8192 → 67,108,864 vs 1,310,720 (51×).

**Checkpoints**

| File | What it is |
|---|---|
| `checkpoints/v6_mdlm_d256_l6_seed1.pt` | 5.3M-param masked-diffusion denoiser used in V6/V8 — learns real context (loss 1.70 vs 5.10 unigram) |
| `checkpoints/clmp_parity_ctx1024.pt` | Original 85k-param toy — context-blind (kept for the record; see `experiments/v6/check_original_checkpoint.py`) |

**Lessons:** Hadamard orthogonal keys reduce to `boundary_id % 32`, so explicit envelopes replaced them. A 1,024-token sequence has 8 shards, so at most 8 cores do useful work per page. Model quality must always be checked against a word-frequency baseline — it caught a context-blind checkpoint.

</details>

---

## Quickstart

```bash
git clone https://github.com/robson-apk/SeedPlane.git
cd SeedPlane
python3 demo.py                     # zero-dependency visual demo
```

<details>
<summary>Reproduce the experiments</summary>

```bash
pip install -r requirements.txt
# needs the TinyStories validation text at data/TinyStories-valid.txt (cache is built on first run)
python3 experiments/v8/v8.py --cache data/tinystories_word1024_cache.pt --ckpt checkpoints/v6_mdlm_d256_l6_seed1.pt
python3 experiments/v7/v7.py train_a && python3 experiments/v7/v7.py train_b && python3 experiments/v7/v7.py eval
python3 docs/record_trajectory.py && python3 docs/make_gifs.py && python3 docs/make_charts.py
```

Every experiment folder has `PROTOCOL.md` (criteria, written first), the code, raw `results/`, and `RESULTS.md`.

</details>

<details>
<summary>Repository layout</summary>

```text
SeedPlane/
├── demo.py            # zero-dependency visual demo
├── seedplane/         # original model + router code
├── checkpoints/       # original toy + V6 denoiser
├── experiments/       # v4 … v9 — protocol, code, raw results, verdict
├── docs/              # charts, GIFs and the scripts that render them
└── data/              # local corpus/cache (git-ignored)
```

</details>

---

## Cite & support

```bibtex
@software{seedplane2026,
  author = {Robson},
  title  = {SeedPlane: Spatial Text Diffusion Across Independent CPU Cores},
  url    = {https://github.com/robson-apk/SeedPlane},
  year   = {2026}
}
```

MIT licensed. If this research is useful to you:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/robson.apk)
