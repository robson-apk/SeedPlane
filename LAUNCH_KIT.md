# 🚀 SeedPlane Launch Kit (Social & Community Copy)

Use these pre-formatted templates to publish SeedPlane to the AI / open-source community.

---

## 1. Hacker News (news.ycombinator.com)

**Title:**
> Show HN: SeedPlane – Spatial text diffusion across CPU cores (Hadamard keys vs. message envelopes)

**Body / First Comment:**
```text
Hey HN,

Modern LLMs are stuck in an O(N) sequential bottleneck: generating token t+1 requires waiting for token t. Distributing autoregressive generation across machines requires expensive InfiniBand/NVLink networks to keep GPUs in lockstep every single token.

I wanted to explore an alternative inspired by procedural world generation in video games:
What if text could be partitioned into independent spatial chunks (shards) rendered concurrently by independent CPU cores, without needing a global attention lock or centralized KV cache?

In this project (SeedPlane), I explored two things:
1. Spatial Sharding: Denoising 128-token text chunks across independent CPU worker processes. This yielded a 3.5x wall-clock speedup (scaling from ~41ms down to ~12ms on 4 CPU workers).
2. Boundary Coordination: How to route and filter overlapping "halo" proposals without shared-memory locks.

We started with an elegant mathematical hypothesis (V4): using deterministic Sylvester Hadamard orthogonal keys (+K_b / -K_b) so foreign boundary proposals cancel out by orthogonality.

Then, we conducted an exhaustive peer-audit (V5, 270,000 decisions, 180 paired runs) comparing Hadamard vectors against a classic engineering baseline: lightweight message envelopes (request_id, generation_step, boundary_id).

What we learned:
- Hadamard matching is mathematically isomorphic to integer boundary matching, but introduces modulo-32 collisions and cannot detect temporal staleness (delayed steps from the same boundary).
- An exact message envelope matches the output with 0.0000000 difference at lower CPU overhead.
- However, the underlying spatial diffusion sharding and adaptive disagreement-defer scheduler (+16.97% seam coherence) are genuinely effective on commodity CPUs.

The repository includes a standalone zero-dependency terminal demo:
`python3 demo.py`

Full code, weights, and replication scripts for the audit: https://github.com/robson-apk/SeedPlane

I built this on commodity hardware (Ryzen 5600X + Intel Arc B580) and would love feedback from distributed systems and ML researchers!
```

---

## 2. Reddit — r/LocalLLaMA

**Post Title:**
> [Project] SeedPlane: Exploring asynchronous spatial text diffusion on CPU cores (and what we learned auditing Hadamard routing)

**Post Body:**
```markdown
Hey r/LocalLLaMA,

We all know the biggest pain point of running local LLMs: autoregressive generation is strictly sequential. You wait token by token, and scaling across multiple consumer CPU cores usually hits memory-bandwidth walls.

I've been working on an open-source research prototype called **SeedPlane**.

Instead of generating text left-to-right, it treats text as an open spatial field (like chunks in Minecraft):

### How it works:
1. **Spatial Sharding:** Text is divided into 128-token shards handled concurrently by independent CPU cores. On 4 workers, median generation latency dropped from ~41.8ms to ~11.9ms (**3.5x speedup** on standard CPU threads).
2. **Boundary Coordination & The V5 Audit:** We initially tested parameter-free Hadamard orthogonal keys to coordinate shard boundaries. In our audit, we stress-tested this against classic message envelopes (`request_id, generation_step, boundary_id`). We found that exact envelopes match the output with 0.0000000 difference while completely eliminating staleness and modulo collisions.
3. **Adaptive Seam Deferral:** When neighboring cores disagree on the overlapping seam, the system defers commitment and schedules a quick localized denoising pass (`disagreement-defer`), boosting boundary coherence by +16.97%.

It has a 10-second interactive CLI demo that requires ZERO dependencies (pure Python):
```bash
git clone https://github.com/robson-apk/SeedPlane.git
cd SeedPlane
python3 demo.py
```

Full audit write-up & code: https://github.com/robson-apk/SeedPlane

Would love to hear thoughts on how we can push non-autoregressive local inference further!
```

---

## 3. Twitter / X Thread

**Tweet 1 (Hook):**
> Modern LLMs are trapped in an O(N) sequential loop: wait for token t to compute token t+1. 
>
> What if we rendered text like a procedural video game world across independent CPU cores?
>
> Introducing SeedPlane: Spatial Text Diffusion across CPU cores. 🧵👇

**Tweet 2 (The Multi-Core Speedup):**
> In autoregressive models, distributing across cores hits a memory wall.
>
> With spatial diffusion sharding, 4 persistent CPU worker processes dropped latency from 41.8ms to 11.9ms (3.5x speedup) on context L=1024.

**Tweet 3 (The Hadamard Hypothesis & The Audit):**
> We first tested Sylvester Hadamard orthogonal keys for lock-free boundary routing.
>
> Then we audited it against simple message envelopes (request_id, step, boundary_id).
>
> Result: Envelopes produce identical outputs with zero modulo collisions and detect staleness. Real science > hype!

**Tweet 4 (Adaptive Seam Deferral):**
> When cores disagree on overlapping boundaries, naive systems guess.
>
> SeedPlane uses `disagreement-defer`: defer commitment and allocate a local localized diffusion pass.
> 
> Result: +16.97% boost in boundary coherence.

**Tweet 5 (Open Source & Demo):**
> Built on commodity hardware (Ryzen 5600X). 
>
> Try the zero-dependency interactive demo right now in your terminal:
> `python3 demo.py`
>
> ⭐️ Open-source on GitHub: https://github.com/robson-apk/SeedPlane
```

---

## 4. LinkedIn Post

```text
Excited to share SeedPlane — an open-source research project exploring spatial text diffusion across independent CPU cores.

In traditional Large Language Models (LLMs), generation is inherently sequential: token t+1 depends strictly on token t. 

SeedPlane explores spatial rendering:
- Text is partitioned into autonomous shards processed concurrently by independent CPU workers, achieving a 3.5x wall-clock speedup across 4 cores.
- An adaptive disagreement-defer scheduler boosts boundary coherence by +16.97%.
- In our V5 peer-audit (270,000 decisions), we compared our initial Hadamard routing against exact message envelopes, establishing clear trade-offs between algebraic keys and classic networking envelopes.

Try the interactive demo (pure Python, zero dependencies required):
https://github.com/robson-apk/SeedPlane

#MachineLearning #ArtificialIntelligence #OpenSource #DeepLearning #DistributedSystems #ComputerScience
```
