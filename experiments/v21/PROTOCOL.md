# V21 — native engine becomes a usable SeedPlane runtime: tokenizer, sampling, chat, CLI

Protocol written before implementing or measuring, 2026-09-24.

## Scope

1. **Native tokenizer (C++)**: Qwen2 byte-level BPE from the bundle's `tokenizer.json`. It covers NFC normalisation,
   the pre-tokenizer regex (hand-written matcher with generated Unicode tables), literal matching of special tokens,
   and a streaming UTF-8-safe decoder.
2. **Native sampling**: temperature / top-k / top-p with the same definition as `Qwen2Engine.sample`, plus a seed.
3. **Sessions**: the engine keeps its KV state between chat turns (appending tokens instead of re-prefilling), using the
   V20 default boundary mode (shadow-batch) and batched prefill.
4. **Interfaces**: `qwen_vk --serve` (JSON lines over stdin/stdout), `qwen_vk --chat` (terminal), and
   `seedplane chat --native` / `seedplane generate --native` in the Python CLI. Portable CMake build next to `build.bat`.

## Gates (fixed now)

- **G1 tokenizer exactness:** native `encode` equals HF `tokenizers` 0.23.2 on (a) the whole WikiText-2 test file and
  (b) a stress file with Portuguese, accents in NFD form, CJK, emoji, code, mixed whitespace (`\t`, `\r\n`, U+00A0,
  U+2028, U+001C), contractions (`'S`, `'LL`, `ſ`) and every special token. `decode(encode(x)) == NFC(x)` on both.
  Any mismatch fails G1; the number of mismatching pieces is reported.
- **G2 no regression:** after the refactor, greedy tokens equal the V20 raw `speed_shadow-batch` tokens (1,024) and the
  V19 oracle (296), and the chunk-1 NLL per position differs from V20 shadow-batch by ≤ 1e-5.
- **G3 sampling correctness:** on 3 logit vectors dumped from the engine, the native sampler's empirical distribution
  over 10^6 draws matches the reference distribution (`Qwen2Engine.sample` semantics computed in float64) for
  (T=0.7, k=40, p=0.9), (T=1.0, k=0, p=1.0) and (T=0.7, k=0, p=0.5). It never draws outside the reference support,
  and total variation ≤ 0.01. For the full-vocabulary case, tokens with p < 1e-4 are pooled into one bucket.
- **G4 sampling cost:** decode tok/s with (T=0.7, k=40, p=0.9) ≥ 0.95 × greedy tok/s, same prompt and mode as V20
  speed (median of 3).
- **G5 session continuation is exact:** a 2-turn chat continued in-session produces the same greedy turn-2 tokens (64)
  as re-prefilling the whole conversation from scratch in a fresh session.
- **G6 end to end:** `seedplane convert --native` followed by `seedplane chat --native` (scripted 2 turns over the
  `--serve` protocol) completes on the 5600X/B580 and returns non-empty decoded text for both turns.

Failures are recorded as failures with the measured numbers.
