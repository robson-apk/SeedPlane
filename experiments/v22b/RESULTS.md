# V22b — results

Run: 2026-09-24 on the Windows 5600X + Intel Arc B580, using the isolated `sp_v22b_codex` copy. The full machine-readable
output is in [`gates.json`](gates.json). Criteria are unchanged from [`PROTOCOL.md`](PROTOCOL.md).

## Gate summary

| Gate | Result | Evidence |
| --- | --- | --- |
| G1 distribution | **Fail, 47/48 cases pass** | All 48 cases had zero draws outside reference support. The sole TV failure is `synthetic_s2`, T=0.7, top-p=0.99: TV 0.0467298 > 0.01. No fallbacks. |
| G2 speed | **Pass** | Final fixed-mode graph ratios vs greedy: T=1 1.0040×; k40+p0.9 0.9743×; p0.9 0.9832×; p0.99 0.9760×; k200 0.9727×. Every individual run reported zero fallbacks. |
| G3 greedy regression | **Pass** | Greedy output tokens match the V20 reference in all three rounds. |
| G4 session/CLI | **Pass** | Session continuation crossed position 512 (2,000 context tokens) and matched a fresh prefill token-for-token. CLI converted the source bundle and completed two chat turns with the Windows binary (return code 0). |

The failing G1 case has support size 45,999. For its reference probabilities, the usual multinomial approximation for
the expected empirical TV at one million draws is 0.0467831, nearly identical to the observed 0.0467298. That result is
consistent with sampling noise at this support size. The pre-registered threshold is still recorded as failed; this
observation explains its statistical power and does not change the gate retroactively.

## Implementation under measurement

- Two-level Vulkan histograms select the top-k/top-p boundary. A conservative fixed-point mass guard prevents the
  cumulative mass estimate from crossing below reference support.
- Combined top-k + top-p sorts the at-most-256 candidates on the GPU, applies the exact filters, and performs Gumbel-max
  there. Candidate overflow remains visible and uses the counted host fallback.
- Fixed sampling command graphs omit the unused argmax and dispatches for other modes; `--serve` keeps a dynamic graph.
  Candidate/status bytes copy only for combined top-k+top-p, where the host must detect overflow. Top-k-only also skips
  probability-mass exponentials and atomics.
- Windows MSVC/Vulkan build passed. The macOS and Linux CMake builds compiled the C++ runtime and all shaders, including
  the new candidate sampler.

## Cross-device runtime smoke

On the Apple M4 through MoltenVK, the same 3,000-token prompt and 1,024-token decode matrix also passed the ≥0.95×
per-configuration speed threshold. Median greedy was 64.877 tok/s; measured ratios were T=1 1.0270×, k40+p0.9 1.0328×,
p0.9 1.0134×, p0.99 1.0131×, and k200 1.0156×. There were zero fallbacks. The apparent ratios above 1× reflect
run-to-run variation, not a speedup caused by sampling.
The [raw three-round data](mac_g2.json) includes per-run step time. The initial 64-token smoke also identified the device
as `Apple M4` and completed with 0 fallbacks.

The X79/RX570 was confirmed as `AMD Radeon RX 570 Series (RADV POLARIS10)`. Its final three-round G2 median ratios were T=1
0.9932×, k40+p0.9 0.9777×, p0.9 0.9940×, p0.99 0.9924×, and k200 0.9829×; zero fallbacks in every run. Raw data is in
[`linux_g2.json`](linux_g2.json). This test used a separate `~/sp_v22b_codex` directory; the existing `~/sp_v23` experiment
was not modified.

The B580, M4/MoltenVK, and RX570/RADV also ran the same 200,000-draw k40+p0.9 GPU distribution check for `logits_2000`.
All three returned the same result: support 10, zero outside draws, TV 0.00179688, 40 candidates, and the `gpu-top-k+p`
path. Per-device count files were captured during the runs; the B580 full distribution data is in `gates.json`.

Session continuation via the JSON-lines `--serve` interface also passed on M4 and RX570: both generated 64 tokens on the
first turn, retained 2,000 prompt/context positions, and generated the exact same second-turn tokens as a fresh prefill.
B580 passed the corresponding session test and the Python CLI convert/chat path in G4.

An all-equal logits fixture exercised overflow: the candidate count was 151,936 (>256), `--sample-test-gpu` explicitly
selected `fallback`, and the 10,000 observed draws stayed within the reference nucleus support. This validates fallback
routing; that draw count is not used for a TV-quality claim. The repeatable fixture generator is
[`make_uniform_logits.py`](make_uniform_logits.py), and the summary is [`fallback_overflow.json`](fallback_overflow.json).

## B580 + RX570 independent-request fleet feasibility

A separate SSH/JSONL harness drove the two native `--serve` processes concurrently. Both loaded identical `weights.spw`
(SHA-256 `ef9f3f59f925e46a303193d7b88a979a899c53b5fec2ed0ff679b61fdf3cbe49`). Each interleaved round compared 24
independent 128-token greedy requests on B580 alone against the same 24 requests distributed over B580 + RX570; every
request generated 128 tokens, and all outputs were token-identical across workers.

| Round | B580 alone | B580 + RX570 | Aggregate ratio |
|---|---:|---:|---:|
| 1 | 273.63 tok/s | 377.85 tok/s | 1.3809× |
| 2 | 274.25 tok/s | 378.37 tok/s | 1.3796× |
| 3 | 272.76 tok/s | 378.50 tok/s | 1.3877× |

The median aggregate throughput ratio is **1.3809×**, above the prospective 1.30× throughput target for this workload.
This is a persistent-SSH-stdio feasibility harness, not the product `seedplane` network worker/pool or an acceleration of
one generation. The scheduler assigned 17 requests to B580 and 7 to RX570 in each pool round.

Latency needs two separate readings. For this synchronized burst, all 24 requests are assumed to arrive at batch start;
the p99 completion time from that arrival was 11.20–11.26 s on B580 alone and 8.12–8.13 s on the pool (median ratio
0.724×), because shorter total queueing outweighed the slower RX570 service. By contrast, the p99 worker service/SSH
round-trip was ~0.48–0.51 s for B580-only tasks and ~1.95 s for the pool, with most RX570 service times ~1.02 s; the
median ratio was 4.07×. The arrival process and which p99 definition should govern V26 were not pre-registered for this
exploratory run, so **do not count either reading as formal V26 latency-gate acceptance**. The next formal experiment
must define the request arrival pattern and report both queue-inclusive completion and service latency. Raw values are in
[`fleet_g2.json`](fleet_g2.json); reproduce with [`benchmark_fleet_ssh.py`](benchmark_fleet_ssh.py).
