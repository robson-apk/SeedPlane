# V22 — GPU-assisted sampling

Protocol written before implementing, 2026-09-24.

## Analysis that motivated it (exploratory, V21 final binary, speed prompt, 1,024 tokens, median of 3)

| config | tok/s | step (GPU + logits copy) | host sampler |
|---|---:|---:|---:|
| greedy | 267.6 | 3.737 ms | — |
| greedy + logits copy only | 262.3 | 3.812 ms | — |
| T=0.7 k=40 p=0.9 | 245.6 | 3.810 ms | 0.263 ms |
| T=1.0 k=0 p=1 | 153.3 | 4.315 ms | 2.213 ms |
| T=0.7 k=0 p=0.9 | 179.8 | 4.028 ms | 1.536 ms |

The V21 G4 gate covered only the cheapest configuration. The cost is host work over the full vocabulary
(exp / partial sort / heap over 151,936 logits). The 600 KB copy is ~2%.

## Design under test

- **Pure temperature (k=0, p=1):** Gumbel-max on the GPU, `argmax_i(l_i/T + G_i)` with `G_i = -log(-log(u_i))` and
  `u_i` from a counter-based hash of (seed, draw counter, i). This is an exact categorical draw from softmax(l/T), up to
  the 32-bit uniform resolution. No host work and no logits copy.
- **top-k and/or top-p:** a GPU kernel returns the 256 largest logits (radix select, ties at the boundary included up
  to 256), the maximum, and Z = Σ exp((l - max)/T) over the whole vocabulary. The host applies the reference top-k /
  top-p definition to those candidates (Z supplies the full-vocabulary softmax denominator for top-p without top-k) and
  draws. If the kept set might extend beyond the 256 candidates (top_k > 256, a tie at the 256th value, or top-p mass
  not reached within 256), it falls back to the V21 host path over the full logits. Fallbacks are counted.

## Gates (fixed now)

- **G1 distribution:** for the 3 configurations × the 8 logit vectors of the V21 post-hoc G3 (3 decode, 3 WikiText,
  2 synthetic), the GPU-assisted sampler's empirical distribution over 10^6 draws has TV ≤ 0.01 vs the float64
  reference, and never draws outside the reference support. The long tail is pooled for k=0, p=1 as in V21.
  Fallbacks are reported.
- **G2 speed:** for each of the 3 configurations, median decode tok/s ≥ 0.95 × greedy median (speed prompt,
  1,024 tokens, 3 runs, same session of measurements). All three must pass.
- **G3 greedy untouched:** greedy tokens identical to the V20 shadow-batch speed tokens (1,024).
- **G4 runtime still works:** V21 G5 (session continuation) and G6 (CLI convert + chat) pass on the new binary.

Failures are recorded as failures with the measured numbers.

## Addendum 1 (before any gate measurement; after one exploratory speed probe)

Probe on the first implementation (1 run each, not gate data): Gumbel path 265.3 tok/s vs greedy 266.4 (0.996×). The
single-workgroup top-256 radix kernel made the candidate path slower (234.4 and 238.9 tok/s, step +0.45 ms), because
it does 6 serial passes over the vocabulary in one workgroup.

Replacement for the top-k / top-p path (Gumbel path unchanged):
1. `sample_stats` (64 workgroups): partial max and Σexp per workgroup;
2. `sample_reduce` (1 workgroup): M and Z over the whole vocabulary;
3. `sample_collect` (many workgroups): appends every token with logit ≥ M − 30·T (up to 4,096 candidates).

The host applies the reference top-k / top-p to the candidates, with Z as the full softmax denominator.
Exactness: top-p is exact whenever its kept prefix is reached inside the candidates (all excluded tokens are smaller
than every candidate); otherwise it falls back. Top-k is exact when ≥ k candidates exist. With fewer than k candidates,
the missing tokens each have probability < e^-30 of the most likely token and are dropped, a documented
approximation. More than 4,096 candidates falls back to the full host path. The gates are unchanged.

## Addendum 2 (before any gate measurement; after a second exploratory probe)

Probe of Addendum 1 (1 run each, not gate data): the fixed M − 30·T window overflowed the 4,096-candidate buffer on
647/1,024 tokens (k=40) and 561/1,024 (top-p), because Qwen's logits are wide. The fallbacks gave 250.0 and 203.0 tok/s.

Replacement threshold: `sample_hist` bins every logit within 30·T of M into 1,024 bins (parallel, global atomics).
`sample_select` then walks the bins from the top and stops at the first bin edge where
(top-k) the count reaches k, or (top-p) a *lower bound* of the mass above the edge, count × e^{(edge−M)/T} / Z,
exceeds p. `sample_collect` appends the tokens above that edge. Exactness is unchanged from Addendum 1: top-p is exact
because the lower bound guarantees the kept prefix lies inside the candidates, and top-k is exact whenever k tokens lie
within 30·T. The gates are unchanged.
