# V21 — native engine as a usable SeedPlane runtime: results

Protocol: [PROTOCOL.md](PROTOCOL.md), written before implementing. Arc B580, Qwen2.5-0.5B-Instruct, seedplane-bundle/2
with plan S=512 / H=256 / sinks=4, default mode shadow-batch. Raw data: `g1_tokenizer.json` (G1),
`gates_run1.json` (G2–G6, pre-registered run), `posthoc_g3_g4.json` (post-hoc run after the G4 failure),
`gates_final_binary.json` (G2–G6 repeated on the final binary).

## Verdict (pre-registered run)

| Gate | Criterion | Measured | Result |
|---|---|---|---|
| G1 tokenizer | ids = HF `tokenizers` 0.23.2; decode(encode(x)) = NFC(x) | WikiText-2: 298,938/298,938 tokens identical; stress file 454/454; both round trips exact | **pass** |
| G2 no regression | V20 tokens, V19 oracle, chunk-1 NLL ≤ 1e-5 | 1,024/1,024 and 296/296 identical; NLL diff **0.0** | **pass** |
| G3 sampling | TV ≤ 0.01, never outside support | TV ≤ 1.4e-4, 0 outside draws, but **degenerate** (see below) | **pass (uninformative)** |
| G4 sampling cost | sampling ≥ 0.95 × greedy | **10.6 vs 269.2 tok/s (0.04)** | **fail** |
| G5 session = fresh prefill | turn-2 greedy tokens identical | identical (19 tokens) after 1,377 positions, across shard boundaries | **pass** |
| G6 end to end | `convert --native` + `chat --native`, 2 turns | both turns answered through the Python CLI | **pass** |

## Failures and what was done about them

**G4 failed badly on the pre-registered run.** Cause: the logits readback buffer used host-visible, *uncached*
memory, and the sampler's `nth_element` read it thousands of times, so every token paid about 90 ms. Fixes, all after
the G4 data:
1. host-cached memory for readback buffers → 169 tok/s;
2. a top-k min-heap single pass instead of an indirect `nth_element` over 151,936 indices, and top-p limited to tokens
   within 40 nats of the maximum (excluded tail ≤ V·e⁻⁴⁰ ≈ 6e-13 of the mass) → 246 tok/s.

**Post-hoc re-measurement (labelled post-hoc, same protocol):** greedy 264.1 / 267.4 / 269.9 tok/s, sampling
247.0 / 244.5 / 245.7 tok/s, ratio **0.919**. **G4 still fails** against the pre-registered 0.95. The remaining
~0.3 ms per token is the logits copy to the host plus two passes over the vocabulary on the CPU.

**G3 was not really exercised by the pre-registered logits.** At the 3 decode positions (2,999 / 3,500 / 4,020 of the
speed run), the model is nearly deterministic: top-k/top-p keep a single token, and the full distribution has
entropy ≤ 0.12 nats. The post-hoc run repeated G3 on the final sampler with 5 more logit vectors, 3 teacher-forced
WikiText positions (entropy up to 4.2 nats) and 2 synthetic ones (up to 9.9 nats). That gives **15 non-trivial cases** with
support from 2 to 348 tokens under filtering. All pass: max TV 0.0079, 0 draws outside the reference support.

**Final-binary confirmation** (`gates_final_binary.json`, G2–G6 rerun on the binary that is committed): G2, G3,
G5, G6 pass; G4 ratio 0.923 (greedy 266.6 / 266.4 / 267.5, sampling 246.1 / 243.8 / 246.5 tok/s), still a fail.

## Also measured

- **Tokenizer portability:** the same `tokenizer.hpp`, compiled with clang on macOS (`tools/tok_cli.cpp`), produced the
  same 298,938 WikiText-2 tokens as HF `tokenizers` 0.22.2 on the Mac, in 0.5 s. That is an extra check, not a gate.
- **NFC:** equal to Python `unicodedata` (Unicode 14.0) on all 145,854 assigned code points ≥ U+0020 < U+30000, and on
  their NFD form. A 3,000-string random fuzz (14 scripts/blocks) also tokenised identically to HF (88,894 tokens).
- **Build:** the engine passes `clang++ -Wall -Wextra` syntax checks against the Khronos headers. The new
  `CMakeLists.txt` builds on the 5600X (NMake + MSVC + Vulkan SDK), and that binary reproduces the V19 oracle
  (296/296) at 275.6 tok/s.
- **G6 answers are low quality** ("A CPU ... funciona em paralelo" is wrong). The engine behaves correctly here: this is
  what a 0.5B model says, under the SeedPlane plan, at temperature 0.7.

## Not measured / out of scope

- Other GPUs, operating systems at run time (Linux/macOS builds are not run), models beyond Qwen2.5-0.5B.
- GPU-side sampling (the next lever for G4).
- The ~30 ms latency spikes from V20 were not investigated further.
