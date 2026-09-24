# SeedPlane — tools & API

SeedPlane runs an existing Hugging Face causal language model as **independent shard windows** that can be spread over
any mix of devices: CPU cores, GPUs (CUDA, Intel XPU, Apple MPS) and other computers on your network.
Weights are never modified — SeedPlane only changes *which tokens each position attends to* and *where the work runs*.

```bash
python -m pip install --upgrade pip
python -m pip install -e .  # from a clone of this repository (Python ≥ 3.9, PyTorch ≥ 2.2)
seedplane --help
```

---

## 1. The idea in one picture

```text
prompt (L tokens) ──► shard 0 │ shard 1 │ shard 2 │ …      each shard = S tokens it owns
                       window k = [sinks] + [H halo tokens of shard k−1] + [shard k]   (original position ids)
                                     │           │           │
                                  GPU        CPU core     other PC      windows never talk to each other
```

A token inside shard *k* sees, causally, its own shard plus the last *H* tokens of the previous shard
(and optionally the first *K* "sink" tokens). With `S ≥ L` the output is **bit-identical** to the original model.

---

## 2. Command line

### `seedplane convert` — make a bundle
```bash
seedplane convert Qwen/Qwen2.5-0.5B-Instruct ./qwen05.sp --shard 512 --halo 256 --sinks 0
```
Writes the original `model.safetensors` + tokenizer + `seedplane.json` (shard plan and provenance) into `./qwen05.sp`.
Any Hugging Face `AutoModelForCausalLM` checkpoint that accepts `position_ids` works (Qwen2/2.5, Llama, Mistral, …).

### `seedplane serve` — turn a device into a worker
```bash
export SEEDPLANE_AUTHKEY="a long random secret"          # same value on every machine
seedplane serve --bundle ./qwen05.sp --device cpu --port 52000 --threads 1              # loopback only (default)
seedplane serve --bundle ./qwen05.sp --device xpu --port 52002 --host 192.168.1.20       # LAN: explicit key required
```
One worker = one process on one device. Start as many as you have cores/GPUs, on as many machines as you like.

### `seedplane run` — process a long prompt across workers
```bash
seedplane run --bundle ./qwen05.sp --prompt-file long.txt \
              --workers local:xpu:1,local:cpu:4,other-pc:52000,other-pc:52001 \
              --scheduler dynamic
```
`--workers` mixes locally spawned workers (`local:<device>:<count>`) and remote ones (`host:port`).
Output: tokens/s, perplexity of the prompt under the shard plan, and how many windows each worker took.

**Schedulers**
| `--scheduler` | How work is split | Use when |
|---|---|---|
| `dynamic` *(default)* | Pull queue: a worker gets the next window when it finishes. Round-trip time per worker is tracked continuously (network included). **Tail guard:** a worker only gets a window if it can finish it before the rest of the pool would finish the whole remaining queue without it. | Mixed hardware, remote machines, thermal throttling |
| `static` | One upfront split proportional to a 1-window speed probe | Identical workers on one machine |

### `seedplane bench` — quality and speed vs. the original model
```bash
seedplane bench --bundle ./qwen05.sp --text-file corpus.txt --length 4096 --samples 3 --device cpu
```
Reports perplexity with full attention vs. the shard plan, the ratio, and the time of each.

---

## 3. Python API (`seedplane.engine`)

| Function | What it does |
|---|---|
| `ShardPlan(shard=512, halo=256, sinks=0)` | The shard layout. `plan.windows(L)` yields `(core_start, core_end, token_index_array)`. |
| `load_model(path_or_id, device='cpu')` | Loads a Hugging Face causal LM + tokenizer in eval mode. |
| `window_logits(model, ids, index, device)` | Logits of one window, with the original position ids. |
| `nll_full(model, ids, device)` / `nll_shards(model, ids, plan, device)` | Next-token NLL with full attention / with the shard plan (computed on the device). |
| `prefill_full(model, ids, device)` / `prefill_window(model, ids, index, device, last)` | Prompt processing like `llama.cpp -p`: hidden states for all tokens, logits only for the last one. |
| `save_bundle(out_dir, model_id, plan)` | What `seedplane convert` calls. |

Distribution helpers in `seedplane.cli`: `start_workers(spec, bundle)`, `distribute(...)` (static) and
`distribute_dynamic(..., timeline=[])` (dynamic; optional per-window timeline for plotting).

Native scheduling helpers in `seedplane.llama_backend`: `measure_rates(...)` calibrates workers concurrently using
end-to-end time (including network and contention), while `plan_pieces(...)` assigns contiguous pieces. The planner
defaults to the fastest worker alone unless the pool predicts at least a 4% improvement; this safety margin prevents
short-prompt regressions caused by calibration noise and boundary/network overhead. See the measured V15 result.

`run_batch(workers, jobs, want='prefill', estimates=None, timeline=None)` schedules independent `(ids, window)` jobs
over a heterogeneous pool and returns `(ordered_results, seconds, jobs_per_worker)`. Calibrate `estimates` with the
actual job size when sizes are uniform. Its tail guard admits a slower worker only when that job should finish before
the other devices drain the remaining queue, so small queues keep latency on the fastest worker while saturated queues
use every device that can reduce makespan. See the measured [V16 result](../experiments/v16/RESULTS.md).

```python
from seedplane import engine
model, tok = engine.load_model("Qwen/Qwen2.5-0.5B-Instruct")
ids = tok(open("long.txt").read(), return_tensors="np").input_ids[0][:4096]
full, n = engine.nll_full(model, ids, "cpu")
shard, _ = engine.nll_shards(model, ids, engine.ShardPlan(512, 256), "cpu")
print("perplexity cost of sharding:", (shard - full) / n)
```

---

## 4. Worker protocol

Workers are `multiprocessing.connection.Listener`s. The coordinator sends
`('windows', ids, [(core_start, core_end, index), …], want)` with `want ∈ {'nll', 'prefill', 'next'}` and receives
`(results, compute_seconds)`. `('close',)` ends the session; the worker keeps listening for the next coordinator.

**Security.** The transport uses pickle and is not encrypted. Anyone who can reach a worker port and knows or observes
the key can run code on that machine. Workers bind to `127.0.0.1` by default and refuse a non-loopback bind unless
`SEEDPLANE_AUTHKEY` is explicitly set. Use a trusted LAN or an SSH/VPN tunnel, choose a long random key, and never expose
worker ports to the internet. See [SECURITY.md](../SECURITY.md).

---

## 5. What is and is not supported (measured, see `experiments/`)

- ✅ Any Hugging Face causal LM that accepts `position_ids`, unchanged weights.
- ✅ CPU, CUDA, Intel XPU, Apple MPS workers; any mix; other machines over TCP.
- ⚠️ Quality cost depends on how far the text looks back: Qwen2.5-0.5B, 4,096 tokens, S=512/H=256 → +3.7% perplexity on
  short stories, +13% on Wikipedia text (V12). Larger halos cost less quality and more compute.
- ❌ Not a drop-in for tasks that must connect information many shards apart (V7).
- 🧪 Token-by-token generation under the shard plan exists only in the native Vulkan decoder (section 6, Qwen2 family); the
  PyTorch workers do prefill/scoring only.

---

## 6. Native SeedPlane decoder: `seedplane-bundle/2` + `native/vulkan_decode` (V18/V19, experimental)

Token-by-token generation of the SeedPlane model on any Vulkan GPU, with no PyTorch or llama.cpp at run time.

```bash
seedplane convert ./qwen05.sp ./qwen05-native.sp --native            # or an HF dir / hub id; keeps a v1 plan
cmake -S native/vulkan_decode -B native/vulkan_decode/build && cmake --build native/vulkan_decode/build --config Release
#   (Windows without CMake: native\vulkan_decode\build.bat, MSVC + Vulkan SDK; set VULKAN_SDK / VCVARS)
seedplane chat ./qwen05-native.sp --native                           # terminal chat (text in, streamed text out)
seedplane generate ./qwen05-native.sp --native --prompt "Once upon a time" -n 64
native/vulkan_decode/build/qwen_vk ./qwen05-native.sp --chat         # the same chat without Python
```

- **Everything runs in the engine:** the Qwen2 byte-level BPE tokenizer (`tokenizer.hpp`, NFC + the Qwen split regex,
  identical to HF `tokenizers` on all of WikiText-2 in V21), sampling (`--temperature/--top-k/--top-p/--seed`, the
  `Qwen2Engine.sample` definition), the ChatML template, and a persistent session. A new chat turn appends tokens to the
  existing window KV state instead of re-reading the conversation.
- **`--serve` protocol** (what `seedplane.native.NativeEngine` speaks): one JSON request per line on stdin, JSON lines back.
  `{"op":"chat","content":"...","reset":false,"temperature":0.7,"max_new_tokens":256}` streams `{"token":id,"text":"..."}`
  and ends with `{"done":true,"reason":"stop|length|context","generated":n,"prefill_s":…,"decode_tok_s":…,"position":…}`.
  Other ops: `generate` (`text` or `ids`), `tokenize`, `detokenize`, `state`, `reset`.

- **Bundle v2:** `seedplane.json` (architecture, plan, tensor table, hashes) + `weights.spw` (FP16 matrices with fused QKV,
  FP32 norms and biases, 256-byte aligned) + tokenizer files. Qwen2 family only for now (head_dim 64).
- **Semantics:** position *t* attends to its `ShardPlan.windows` window with original position ids. The KV cache holds
  `sinks + halo + shard` slots and is rebuilt in the new window's context at every shard boundary. `--full` switches to
  the original full attention, and `--shard/--halo/--sinks` override the plan.
- **Boundary modes** (`--mode`): `shadow-batch` (default, exact) appends each halo token to the next window in
  K/V-only batches of 8, so the window is ready at the boundary. `rebuild-batch` / `rebuild` re-run sinks + halo at the
  boundary (exact, with a stall). `shadow` uses a second batch column per halo token (exact). `reuse` keeps the
  previous window's halo K/V (approximate, rejected as a default in V20 for worse NLL).
- `--score-file ids.i32 --nll-out nll.f32` scores a token file (teacher-forced NLL per position). `--attn old` selects
  the V19 attention kernel.
- **Measured (Arc B580, Qwen2.5-0.5B, S512/H256/K4, context 3–4k):** shadow-batch 266.9 tok/s vs full attention 250.0,
  KV 19 MB vs 99 MB, prefill of 3,000 tokens 2.5 s. The quality cost of the plan is +8–10% perplexity at 4k.
  See [V18](../experiments/v18/RESULTS.md), [V19](../experiments/v19/RESULTS.md) and [V20](../experiments/v20/RESULTS.md).

## 7. Native backend: `seedplane-worker` on llama.cpp kernels (V13, experimental)

The PyTorch workers above are the reference implementation. For speed, `native/seedplane-worker.cpp` serves the same
windows through the llama.cpp C API, so the math runs on llama.cpp's optimized kernels (CPU, Vulkan, SYCL, CUDA, Metal)
and the model is a GGUF file.

```bash
# build against a llama.cpp checkout (after building libllama there)
c++ -std=c++17 -O3 native/seedplane-worker.cpp -I $LLAMA/include -I $LLAMA/ggml/include \
    -L $LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o seedplane-worker
export SEEDPLANE_AUTHKEY="a long random secret"
./seedplane-worker -m qwen2.5-0.5b-instruct-fp16.gguf --port 54000 --ngl 99                    # loopback
./seedplane-worker -m qwen2.5-0.5b-instruct-fp16.gguf --port 54001 --ngl 0 -t 4 --host 10.0.0.2 # trusted LAN
```

```python
from seedplane import engine, llama_backend as lb
workers = lb.connect(["gpu-box:54000", "gpu-box:54001", "mac.local:54000"])
results, seconds, windows_per_worker = lb.run_windows(workers, ids, engine.ShardPlan(512, 256).windows(len(ids)), want="nll")
```

Binary protocol (little-endian, no pickle). Request: `u32 magic 'SPW1', u32 want (0 prefill, 1 nll, 2 close),
u32 n_tok, u32 core_off, u32 score_from, i32 next_tok, i32 tokens[n], i32 positions[n]`. Response:
`u32 magic, f64 nll_sum, u32 n_scored, i32 argmax_last, f32 compute_ms`. Every window starts from an empty KV cache
and keeps its tokens' original positions.

Measured correctness (V13 C1/C2, Mac CPU): matches the PyTorch engine within 0.6% (single window) and 0.7% (shards)
NLL. That is the F16 vs fp32 difference. Speed results are pending.
