"""Convert a Qwen2 checkpoint into a native SeedPlane bundle (`seedplane-bundle/2`).

    python -m seedplane.native_bundle <hf_dir | v1_bundle | hub_id> <out.sp> [--shard 512 --halo 256 --sinks 4]

The v2 bundle is what the native engine (`native/vulkan_decode`) runs:

  seedplane.json  architecture, shard plan, tensor table, SHA-256 of sources and weights
  weights.spw     engine-ready tensors: FP16 matrices (QKV fused), FP32 norms and biases, 256-byte aligned
  tokenizer.json  (+ tokenizer_config.json, chat_template.jinja, generation_config.json when present)

Weights are converted, never retrained: BF16/FP32 values are rounded to FP16 (round-to-nearest-even) and the
number of overflows / values flushed to zero is recorded. Only numpy is needed; safetensors files are parsed directly.
"""
import argparse, hashlib, json, shutil, struct
from pathlib import Path

import numpy as np

FORMAT = 'seedplane-bundle/2'
ENGINE = 'seedplane-native-qwen2/1'
ALIGN = 256
MAX_WINDOW = 4096            # sinks + halo + shard must fit the engine's attention kernel
EXTRA_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')


def read_safetensors(path):
    """Map name -> (dtype, shape, raw bytes view) without loading the whole file."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; header = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode='r', offset=8 + n)
    return {k: (v['dtype'], v['shape'], data[v['data_offsets'][0]:v['data_offsets'][1]])
            for k, v in header.items() if k != '__metadata__'}


def as_f32(dtype, raw):
    if dtype == 'F32': return np.frombuffer(raw, dtype='<f4').copy()
    if dtype == 'F16': return np.frombuffer(raw, dtype='<f2').astype(np.float32)
    if dtype == 'BF16': return (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f'unsupported dtype {dtype}')


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk): h.update(block)
    return h.hexdigest()


def resolve_source(src):
    p = Path(src)
    if p.is_dir(): return p
    from huggingface_hub import snapshot_download       # optional: only for hub ids
    return Path(snapshot_download(src, allow_patterns=['*.safetensors', '*.json', '*.jinja']))


def validate_plan(plan):
    s, h, k = plan['shard'], plan['halo'], plan['sinks']
    if s <= 0 or h < 0 or k < 0: raise ValueError(f'invalid plan {plan}')
    if s + h + k > MAX_WINDOW: raise ValueError(f'sinks + halo + shard = {s + h + k} exceeds the engine window {MAX_WINDOW}')


def convert(src, out, shard=None, halo=None, sinks=None):
    src = resolve_source(src); out = Path(out)
    cfg = json.loads((src / 'config.json').read_text())
    if cfg.get('model_type') != 'qwen2': raise ValueError('the native engine currently supports model_type=qwen2 only')
    v1 = src / 'seedplane.json'
    v1_meta = json.loads(v1.read_text()) if v1.exists() else {}
    plan = dict(v1_meta.get('plan', {'shard': 512, 'halo': 256, 'sinks': 4}))
    for key, val in (('shard', shard), ('halo', halo), ('sinks', sinks)):
        if val is not None: plan[key] = val
    validate_plan(plan)

    L, H = cfg['num_hidden_layers'], cfg['hidden_size']; NH, NKV = cfg['num_attention_heads'], cfg['num_key_value_heads']
    I, V = cfg['intermediate_size'], cfg['vocab_size']; HD = H // NH
    if H % NH or NH % NKV or HD != 64 or H % 8 or I % 8 or max(H, I) > 4864:
        raise ValueError('shape not supported by the native kernels (head_dim 64, hidden/intermediate <= 4864)')
    theta = cfg.get('rope_theta', cfg.get('rope_parameters', {}).get('rope_theta', 10000.0))

    files = sorted(src.glob('*.safetensors'))
    if not files: raise FileNotFoundError(f'no safetensors in {src}')
    t = {}
    for f in files:
        for name, v in read_safetensors(f).items():
            if name in t: raise ValueError(f'duplicate tensor {name}')
            t[name] = v
    get = lambda name: as_f32(t[name][0], t[name][2])
    tied = 'lm_head.weight' not in t

    out.mkdir(parents=True, exist_ok=True)
    table, stats, h = {}, {'fp16_overflow': 0, 'fp16_flushed_to_zero': 0}, hashlib.sha256()
    with open(out / 'weights.spw', 'wb') as w:
        def put(name, arr, dtype, shape):
            if dtype == 'f16':
                big = np.abs(arr) > 65504
                data = arr.astype(np.float16)
                stats['fp16_overflow'] += int(big.sum())
                stats['fp16_flushed_to_zero'] += int(((arr != 0) & (data == 0)).sum())
            else:
                data = arr.astype(np.float32)
            pad = (-w.tell()) % ALIGN
            if pad: w.write(b'\0' * pad); h.update(b'\0' * pad)
            raw = data.astype('<f2' if dtype == 'f16' else '<f4').tobytes()
            table[name] = {'dtype': dtype, 'shape': list(shape), 'offset': w.tell(), 'bytes': len(raw)}
            w.write(raw); h.update(raw)

        put('embed', get('model.embed_tokens.weight'), 'f16', (V, H))
        if not tied: put('lm_head', get('lm_head.weight'), 'f16', (V, H))
        put('final_norm', get('model.norm.weight'), 'f32', (H,))
        for l in range(L):
            p = f'model.layers.{l}.'
            cat = lambda *names: np.concatenate([get(p + n) for n in names])
            nq = (NH + 2 * NKV) * HD
            put(f'{l}.qkv_w', cat('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'), 'f16', (nq, H))
            put(f'{l}.qkv_b', cat('self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'), 'f32', (nq,))
            put(f'{l}.o_w', get(p + 'self_attn.o_proj.weight'), 'f16', (H, H))
            put(f'{l}.gate_w', get(p + 'mlp.gate_proj.weight'), 'f16', (I, H))
            put(f'{l}.up_w', get(p + 'mlp.up_proj.weight'), 'f16', (I, H))
            put(f'{l}.down_w', get(p + 'mlp.down_proj.weight'), 'f16', (H, I))
            put(f'{l}.in_norm', get(p + 'input_layernorm.weight'), 'f32', (H,))
            put(f'{l}.post_norm', get(p + 'post_attention_layernorm.weight'), 'f32', (H,))

    for name in EXTRA_FILES:
        if (src / name).exists(): shutil.copyfile(src / name, out / name)
    manifest = {
        'format': FORMAT, 'engine': ENGINE,
        'source': v1_meta.get('source', str(src)),
        'source_files': {f.name: sha256_file(f) for f in files},
        'plan': plan,
        'arch': {'model_type': 'qwen2', 'layers': L, 'hidden': H, 'heads': NH, 'kv_heads': NKV, 'head_dim': HD,
                 'intermediate': I, 'vocab': V, 'rope_theta': theta, 'rms_norm_eps': cfg.get('rms_norm_eps', 1e-6),
                 'max_positions': cfg.get('max_position_embeddings', 32768), 'tied_embeddings': tied},
        'weights': {'file': 'weights.spw', 'sha256': h.hexdigest(), 'align': ALIGN, **stats},
        'tensors': table,
        'note': 'weights rounded to FP16, never retrained; SeedPlane changes which tokens each position attends to',
    }
    (out / 'seedplane.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--shard', type=int); ap.add_argument('--halo', type=int); ap.add_argument('--sinks', type=int)
    a = ap.parse_args(); m = convert(a.src, a.out, a.shard, a.halo, a.sinks)
    print(json.dumps({'out': a.out, 'plan': m['plan'], **m['weights']}, indent=1))


if __name__ == '__main__': main()
