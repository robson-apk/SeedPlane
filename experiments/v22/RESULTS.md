# V22 — GPU-assisted sampling: results

Protocol: [PROTOCOL.md](PROTOCOL.md) with Addenda 1–2. All were written before any gate measurement; the addenda
followed exploratory speed probes, which are reported there. Raw data: `gates.json`.
Arc B580, Qwen2.5-0.5B, speed prompt (3,000 tokens) + 1,024 decoded tokens, shadow-batch.

## Verdict

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| G1 distribution | TV ≤ 0.01, 0 draws outside support, 24 cases | max TV 0.0083, 0 outside, 0 fallbacks | **pass** |
| G2 speed | each config ≥ 0.95 × greedy (median of 3, interleaved) | T=1: **0.990**; top-p 0.9: **0.959**; k=40/p=0.9: **0.9486** | **fail** |
| G3 greedy untouched | tokens = V20 | identical in all 3 rounds | **pass** |
| G4 runtime | V21 G5 + G6 on the new binary | both pass | **pass** |

G2 fails by 0.15 percentage points on k=40/p=0.9. Medians: greedy 264.1, T=1 261.4, top-p 253.2, k=40/p=0.9
250.5 tok/s.

## Against V21

| config | V21 (host sampler) | V22 | change |
|---|---:|---:|---:|
| T=0.7 k=40 p=0.9 | 245.6 | 250.5 | +2% |
| T=1.0 (temperature only) | 153.3 | **261.4** | **+71%** |
| T=0.7 top-p 0.9 | 179.8 | **253.2** | **+41%** |

(V21 figures come from the exploratory analysis in PROTOCOL.md, measured with the same prompt.)

## Why k=40 still misses (exploratory probe after the gates, 2 runs each)

| config | median latency | GPU step | host sampler |
|---|---:|---:|---:|
| greedy | 3.406 ms | 3.783 ms | — |
| top-k 1 | 3.483 | 3.910 | 0.7 µs |
| top-k 40 | 3.605 | 4.069 | 3.3 µs |
| top-k 200 | 3.939 | 4.256 | 14 µs |
| top-p 0.9 | 3.484 | 3.923 | 0.9 µs |
| **top-p 0.99** | **8.485** | **7.315** | **1.25 ms → 116.8 tok/s** |

1. The GPU step grows with the number of candidates. The candidate list and its atomic counter live in host-visible
   memory, so every append crosses PCIe.
2. **High top-p is pathological.** At p=0.99 the kept set genuinely holds thousands of tokens. It overflows the 4,096
   candidates and falls back to the full host path. The gate did not test p=0.99, but it is a common default, so this is
   a real defect.

Next (V22b, to be pre-registered): select the top-k/top-p threshold exactly on the GPU, with a second-level histogram
inside the crossing bin, and draw with Gumbel-max restricted to `l ≥ τ`. That removes candidates and the host from
the path for every configuration and every kept-set size.

## What stays true

- Temperature-only sampling costs ~1% (Gumbel-max, exact up to the 32-bit uniform; tokens more than ~23 nats below the
  winner can never be drawn, total mass < V·e⁻²³).
- The sampling distributions are correct (G1), including 15 non-trivial cases.
