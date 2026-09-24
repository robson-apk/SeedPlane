# V20 — fixing where native SeedPlane decode loses: boundary cost and attention scaling

Protocol written before implementing or measuring, 2026-09-23.

## Problems (from V19, measured)

- **P1 boundary cost:** the exact rebuild re-runs sinks + halo one token per submission at each shard boundary.
  Overall decode was ~174 tok/s vs ~258 tok/s between boundaries (S64/H32/K4).
- **P2 attention scaling:** one workgroup per head with a 4096-slot shared array. Full attention got slower with
  context (229 tok/s at ~300 tokens vs 258 at ~130), and windows are capped at 4096.

## Candidates

For P1, three boundary modes on the same engine:

- **rebuild** (V19 baseline, exact): sequential re-run of sinks + halo at the boundary.
- **shadow** (exact): during the last H tokens of shard k, each token is also run inside window k+1 (second batch
  column, its own KV cache) and seeded with the sink K/V. Sink K/V are position-identical across windows, because
  positions `< sinks` only ever see earlier sinks. At the boundary the shadow cache becomes the current one, so there
  is no stall. Matrix-vector products read each weight once for both columns.
- **reuse** (approximate, zero cost): at the boundary, keep the halo K/V computed in window k−1 (sliding reuse)
  instead of recomputing them inside window k. This is a different model from `ShardPlan.windows`, so its quality
  has to be measured.

For P2: **split** attention, partitioned over context chunks of 128 slots plus a combine pass, with no 4096 cap. It is
compared with the V19 kernel (**old**).

## Data (new, never used before)

WikiText-2 test (`seedplane_v12/data/wikitext2_test.txt`), Qwen tokenizer. Earlier experiments used tokens `< 163,840`.
- Quality: 3 chunks of 4,096 tokens at token offsets 200,000 / 204,096 / 208,192, teacher-forced NLL per position.
- Speed: prompt = tokens 220,000..222,999 (3,000 tokens), then 1,024 greedy tokens (positions up to 4,023), 3 runs.
  With the default plan S=512 / H=256 / sinks=4, generation crosses the boundaries at 3,072 and 3,584.

## Gates (fixed now)

- **G1 shadow is exact:** per-position NLL of shadow vs rebuild, max |Δ| ≤ 1e-3 over all 3 × 4,096 positions, and
  greedy tokens identical in the speed runs.
- **G2 shadow removes the stall:** median overall decode tok/s of shadow ≥ 0.90 × median of reuse (reuse has no
  boundary work), and the max single-token decode latency ≤ 3 × the median token latency.
- **G3 reuse quality:** reuse is acceptable only if its mean NLL ≤ 1.005 × the exact plan's NLL on each of the 3 chunks.
  Manipulation check: reuse NLL must differ from exact (max |Δ| > 1e-3 at some position), otherwise invalid.
- **G4 split attention:** full-attention decode at 3–4k context is ≥ 1.2× faster with split than with old (median of 3),
  and the per-position NLL of split vs old has max |Δ| ≤ 1e-3 (chunk 1, full attention).
- **G5 SeedPlane vs full at 3–4k context:** the best exact SeedPlane mode has higher median overall decode tok/s than
  full attention with the best kernel. Its NLL cost vs full attention is reported next to it.

Decision rule, fixed now: the default boundary mode is **shadow**. It becomes **reuse** only if reuse passes G3 and is
≥ 5% faster than shadow. All numbers are reported, including failures.

## Addendum 1 (before any V20 data; written after smoke tests on the V19 scenario only)

Smoke tests on the V19 scenario (S64/H32/K4, 296 tokens, not V20 data) showed:
- shadow is exact (296/296 tokens equal to the V19 oracle), and reuse diverges at generated token 61, as expected;
- the first 2-column kernels slowed every mode, because they reserved 2-column shared memory even for 1 column. That was
  fixed with compile-time 1- and 2-column variants plus vec4 shared loads (single-token median 3.27 ms);
- a 2-column (dual) token still costs ~+46% over a single token, so shadow reached only 0.84 × reuse there.

Added candidates (exact, same semantics as rebuild/shadow):
- **shadow-batch**: halo tokens are appended to the shadow window in K/V-only batches of 8 tokens (causal inside the
  batch, one submission per 8 tokens) instead of one dual column per token.
- **rebuild-batch**: the boundary rebuild runs in K/V-only batches of 8 instead of one token per submission.
- Batch modes (and reuse, full) also prefill the prompt in K/V-only batches of 8. Prefill time is reported only.

Gate changes: G1 and G2 apply to **each exact mode** (shadow, shadow-batch, rebuild-batch) and are reported for each.
G5 and the decision rule use the fastest exact mode by median overall decode tok/s. Every mode is reported.
