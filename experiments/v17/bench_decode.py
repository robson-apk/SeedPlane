"""Measure SeedPlane Qwen2 decode separately from prefill."""
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from seedplane.qwen_engine import Qwen2Engine


def sync(device):
    if device.startswith('xpu'): torch.xpu.synchronize()
    elif device.startswith('cuda'): torch.cuda.synchronize()
    elif device.startswith('mps'): torch.mps.synchronize()


def main():
    p = argparse.ArgumentParser(); p.add_argument('bundle'); p.add_argument('--device', default='cpu'); p.add_argument('-n', type=int, default=128)
    p.add_argument('--prompt', default='9707,11,1879,0'); a = p.parse_args(); ids = list(map(int, a.prompt.split(',')))
    engine = Qwen2Engine(a.bundle, a.device); cache = engine.new_cache(len(ids) + a.n)
    forward = engine.forward
    sync(a.device)
    t0 = time.perf_counter(); logits, cache = forward(ids, cache); sync(a.device); prefill = time.perf_counter() - t0
    generated = [] ; t0 = time.perf_counter()
    for _ in range(a.n):
        token = int(logits[-1].argmax()); generated.append(token); logits, cache = forward([token], cache)
    sync(a.device); decode = time.perf_counter() - t0
    print(json.dumps({'device': a.device, 'prompt_tokens': len(ids), 'generated_tokens': a.n, 'prefill_seconds': prefill,
                      'decode_seconds': decode, 'decode_tok_s': a.n / decode, 'tokens': generated}, indent=2))


if __name__ == '__main__': main()
