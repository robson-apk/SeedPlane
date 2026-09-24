# V20 — fixing where native SeedPlane decode loses: results

Protocol: [PROTOCOL.md](PROTOCOL.md). The protocol and Addendum 1 were written before any V20 data; the addendum
followed smoke tests on the V19 scenario only. Arc B580, Qwen2.5-0.5B-Instruct, plan S=512 / H=256 / sinks=4.
Data: new WikiText-2 test ranges (see `raw/data.json`). Raw outputs: `raw/*.json` and `raw/nll_per_position.npz`.
Gates: `gates.json` (reproducible with `python gates.py raw`).

Coverage: 3 chunks × 4,095 scored positions, 7 shard boundaries per chunk. Speed: 3,000-token prompt + 1,024 decoded
tokens (context 3,000–4,023) crossing boundaries 3,072 and 3,584. 3 runs per mode.

## Verdict

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| G1 shadow | exact vs rebuild | max \|ΔNLL\| 0.0; tokens identical 3/3 | **pass** |
| G1 shadow-batch | exact vs rebuild | max \|ΔNLL\| 3.1e-5; tokens identical 3/3 | **pass** |
| G1 rebuild-batch | exact vs rebuild | max \|ΔNLL\| 3.2e-5; tokens identical 3/3 | **pass** |
| G2 speed shadow | ≥ 0.90 × reuse | 0.892 | **fail** |
| G2 speed shadow-batch | ≥ 0.90 × reuse | 0.951 | **pass** |
| G2 speed rebuild-batch | ≥ 0.90 × reuse | 0.931 | **pass** |
| G2 latency (all exact modes) | max ≤ 3 × median | shadow 9.5×, shadow-batch 9.9×, rebuild-batch 48.6× | **fail** |
| G3 reuse quality | NLL ≤ 1.005 × exact on each chunk | 1.0066 / 1.0006 / 1.0041 | **fail** (reuse rejected) |
| G4 split attention | ≥ 1.2× old, \|ΔNLL\| ≤ 1e-3 | **4.78×**, 2.7e-5 | **pass** |
| G5 SeedPlane vs full | best exact > best full | shadow-batch 266.9 vs full 250.0 tok/s (**1.07×**) | **pass** |

Decision rule outcome: the default is **shadow-batch**, the fastest exact mode. Reuse failed G3, so it does not become
the default. The engine now defaults to `--mode shadow-batch`.

## Decode speed (median of 3, tok/s, context 3.0–4.0k)

| mode | exact? | tok/s | prefill 3,000 tok | max latency | where |
|---|---|---:|---:|---:|---|
| full attention, old kernel (V19) | — | 52.3 | 2.0 s* | 49–58 ms | — |
| full attention, split kernel | — | 250.0 | 2.0 s* | 31–32 ms | random |
| rebuild (V19 SeedPlane) | yes | 187.0 | 15.2 s | **917–926 ms** | boundaries |
| shadow (dual column) | yes | 250.2 | 12.6 s | 31–32 ms | random |
| rebuild-batch | yes | 261.1 | 2.4 s | **136–165 ms** | boundary 3,072 |
| **shadow-batch** | **yes** | **266.9** | **2.5 s** | 30–34 ms | random |
| reuse (approximate) | no | 280.5 | 1.7 s | 29–30 ms | random |

\*Batch prefill (K/V-only batches of 8) is used in full, batch and reuse modes. Rebuild and shadow prefill one token per
submission. Batch prefill always uses the split kernel, so `--attn old` changes only the decode steps. Median latency is 3.39–3.40 ms in every SeedPlane mode and 3.81 ms with full attention at this context.

KV cache: 19.0 MB per window cache for SeedPlane (2 caches in shadow/reuse modes) vs 98.9 MB for full attention at 4,024
tokens.

## Quality (mean NLL per token; perplexity ratio vs full attention)

| chunk | full | exact SeedPlane | reuse |
|---|---:|---:|---:|
| 1 | 2.6873 | 2.7642 (ppl ×1.080) | 2.7826 (×1.100) |
| 2 | 2.5838 | 2.6605 (×1.080) | 2.6621 (×1.081) |
| 3 | 2.7363 | 2.8341 (×1.103) | 2.8456 (×1.116) |

The reuse shortcut is not free. Reusing halo K/V from the previous window is worse than recomputing them on every chunk
(+0.06% to +0.66% NLL), so the "richer context" intuition was wrong here. The exact plan's own quality cost vs full
attention (+8–10% perplexity at 4k, S512/H256) is unchanged by V20. It is the price of the SeedPlane attention pattern
itself (consistent with V12's "+13% on Wikipedia").

## Failures, stated plainly

- **G2 latency failed for every exact mode.** For rebuild-batch the cause is real: a 136–165 ms stall at the boundary.
  For shadow and shadow-batch, the maxima (30–34 ms) are at non-boundary positions. They match the spikes of full
  attention (31–32 ms, 8.5× median) and reuse (29–30 ms, 8.8×), which do no boundary work. An exploratory
  `--spin` fence wait did not remove them. The source is outside the decode loop (GPU/driver or another GPU client;
  ~8 GB of B580 memory was held by an external process in earlier sessions). It was not identified. The gate still counts as failed.
- **G2 speed failed for dual-column shadow** (0.89): a 2-column token costs ~+46% over a single token (measured in the V19-scenario smoke test).
- **G3 failed:** reuse is rejected as a default; it stays available as `--mode reuse`.

## What was fixed (engineering, all exact)

1. **Boundary stall:** rebuild at 187 tok/s with 0.92 s stalls → shadow-batch at 267 tok/s with no boundary stall
   (1.43× faster overall).
2. **Attention scaling:** split attention made full-attention decode at 3–4k context 4.78× faster and removed the
   4,096-slot cap. Without it, SeedPlane's advantage over full attention would have been an artefact of a bad baseline.
3. **Prompt prefill:** K/V-only batches of 8 cut prefill of 3,000 tokens from 15.2 s to 2.4–2.5 s (6.2×).
4. **Kernel regression caught during the smoke tests:** 2-column shared memory had slowed single-token decode by ~35%.
   It was fixed with 1/2-column variants and vec4 loads (single token 3.59 → 3.27 ms on the V19 scenario).

## Not measured / out of scope

- Contexts beyond 4k (the advantage over full attention should grow with length; not measured).
- Other models, other GPUs, sampling (all runs are greedy).
