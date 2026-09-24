# V22b — sampling without the candidate bottleneck

Protocol written before implementing, 2026-09-24. Follows the V22 failure (k=40/p=0.9 at 0.9486×) and the post-gate
finding that top-p 0.99 falls back to the host (116.8 tok/s).

## Design under test

- **top-k only** and **top-p only**: the threshold τ is found on the GPU with two histogram levels. Level 1 has 1,024
  bins over [M − 30T, M]; level 2 has 1,024 sub-bins inside the crossing bin. Counts and **fixed-point probability mass**
  (uint32 atomics, 2^31 scale) are accumulated per bin. The draw is Gumbel-max restricted to `l ≥ τ`: exact categorical
  sampling over the kept set, with nothing copied to the host.
- **top-k with top-p**: the same threshold search gives the top-k set (≤ 256 tokens). They are collected in device
  memory and copied (2 KB) to the host, which applies the reference top-p inside the top-k set and draws. More than
  256 tokens falls back to the V21 host path; fallbacks are counted.
- Temperature only: Gumbel-max as in V22.

Known approximations, stated now: tokens within one level-2 sub-bin (30·T/2^20 nats) of the cut are either all kept
or all dropped; with fewer than k tokens within 30·T of the maximum, the kept set is those tokens. The fixed-point
mass has an absolute error ≤ V·2^-31 ≈ 7e-5.

## Gates (fixed now)

- **G1 distribution:** on the 8 logit vectors of V22 G1, for 6 configurations (T=0.7 k=40 p=0.9; T=1.0; T=0.7 p=0.5;
  T=0.7 p=0.99; T=0.7 k=40; T=0.7 k=200), 10^6 draws each: TV ≤ 0.01 vs the float64 reference and 0 draws outside the
  reference support. For T=1.0, the tail is pooled below 1e-4 as before.
- **G2 speed:** for 5 configurations (T=1.0; T=0.7 k=40 p=0.9; T=0.7 p=0.9; T=0.7 p=0.99; T=0.7 k=200), median decode
  tok/s ≥ 0.95 × greedy (speed prompt, 1,024 tokens, 3 interleaved rounds). All must pass.
- **G3:** greedy tokens identical to V20 (1,024).
- **G4:** V21 G5 (session continuation) and G6 (CLI convert + chat) pass.

Failures are recorded as failures.
