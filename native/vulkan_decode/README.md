# SeedPlane native runtime (`qwen_vk`)

A single executable that runs a SeedPlane bundle (`seedplane-bundle/2`, Qwen2 family) on any Vulkan 1.2 GPU. It
does text in, streamed text out, with shard-window attention, sampling, and chat sessions. No Python, PyTorch or
llama.cpp is needed at run time.

Tested with identical greedy tokens on the Intel Arc B580 (Windows), AMD RX 570 / Mesa RADV (Linux) and Apple M4 via
MoltenVK (macOS). See `experiments/v18`–`v23` for the measurements.

## 1. Get a bundle

```bash
pip install -e .                 # from a clone of the repository
seedplane convert Qwen/Qwen2.5-0.5B-Instruct ./qwen05.sp --native --shard 512 --halo 256 --sinks 4
```

`seedplane convert` accepts a Hugging Face id, a local HF directory or a v1 bundle. It writes `seedplane.json` (plan,
architecture, hashes), `weights.spw` (FP16) and the tokenizer.

## 2. Run

```bash
qwen_vk ./qwen05.sp --chat                     # terminal chat (/reset, /exit)
qwen_vk ./qwen05.sp --serve                    # JSON lines on stdin/stdout (used by `seedplane chat --native`)
qwen_vk ./qwen05.sp --prompt-file ids.i32 -n 256 --runs 3    # benchmark (JSON report)
qwen_vk ./qwen05.sp --score-file ids.i32 --nll-out nll.f32   # teacher-forced NLL
```

Options: `--temperature --top-k --top-p --seed -n`, `--full` (original full attention), `--shard/--halo/--sinks`
(override the plan), `--mode shadow-batch|shadow|rebuild-batch|rebuild|reuse`, `--ctx N`, `--system TEXT`.

The `shaders/` folder must stay next to the executable (or pass `--shaders DIR`).

## 3. Build from source

Requirements: CMake ≥ 3.16, a C++17 compiler, the Vulkan loader/headers and `glslc`.

| Platform | Install | Build |
|---|---|---|
| Linux (Ubuntu) | `sudo apt install libvulkan-dev glslc ninja-build` | `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build` |
| macOS | `brew install molten-vk vulkan-loader vulkan-headers shaderc ninja` | same as Linux |
| Windows | Vulkan SDK + Visual Studio Build Tools | `cmake -S . -B build -A x64 && cmake --build build --config Release`, or `build.bat` |

On macOS the engine enables `VK_KHR_portability_enumeration` / `VK_KHR_portability_subset` automatically. Set
`SP_VK_VALIDATE=1` to run under the Khronos validation layer.
