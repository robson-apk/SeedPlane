"""Exploratory profiling of the V8 decoding path (not a pre-registered result). Where does the time go?"""
import sys, time, json, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parent / 'v8')); sys.path.insert(0, str(ROOT.parent / 'v6'))
import v8
from v6 import MASK, SHARD, load_cache

def timeit(fn, n=20):
    fn(); ts = []
    for _ in range(n): t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))

if __name__ == '__main__':
    cache, ckpt = sys.argv[1], sys.argv[2]; L = 1024; HALO = 16
    model = v8._load(ckpt); _, val = load_cache(cache); y = val[:L].clone(); m = torch.rand(L, generator=torch.Generator().manual_seed(1)) < 0.5
    x = y.masked_fill(m, MASK); xn = x.numpy(); out = {}
    for th in (1, 4):
        torch.set_num_threads(th)
        with torch.inference_mode():
            out[f'global_forward_{th}t_ms'] = timeit(lambda: model(x[None], torch.arange(L)[None]))
            def one_shard(): model(x[112:272][None], torch.arange(112, 272)[None])
            out[f'one_shard_forward_{th}t_ms'] = timeit(one_shard)
            wins = torch.stack([x[c0 - HALO:c0 + SHARD + HALO] for c0 in range(128, 768 + 1, 128)])  # 6 interior shards, 160 tokens
            pos = torch.stack([torch.arange(c0 - HALO, c0 + SHARD + HALO) for c0 in range(128, 768 + 1, 128)])
            out[f'batched_6_interior_shards_{th}t_ms'] = timeit(lambda: model(wins, pos))
            out[f'softmax_max_1024x1024_{th}t_ms'] = timeit(lambda: torch.randn(L, 1024).softmax(-1).max(-1))
    torch.set_num_threads(4)
    with ProcessPoolExecutor(4, mp_context=mp.get_context('spawn'), initializer=v8._init, initargs=(ckpt,)) as pool:
        groups = [[(c0, c0 + SHARD) for c0 in range(0, L, SHARD)][i::4] for i in range(4)]
        list(pool.map(v8._shard_job, [(xn, g, L) for g in groups]))
        out['v8_one_step_4workers_ms'] = timeit(lambda: list(pool.map(v8._shard_job, [(xn, g, L) for g in groups])))
        out['ipc_roundtrip_empty_4workers_ms'] = timeit(lambda: list(pool.map(v8._shard_job, [(xn, [], L)] * 4)))
    out['v8_full_decode_traditional_ms'] = timeit(lambda: v8.decode_traditional(model, xn, L), 3)
    print(json.dumps(out, indent=1)); (ROOT / 'results' / 'profile_v8.json').write_text(json.dumps(out, indent=1))
