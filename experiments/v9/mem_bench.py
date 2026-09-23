"""V9 memory benchmark: peak resident memory (main + workers) per method and core count. One fresh process per config.

python mem_bench.py --cache <cache.pt> --ckpt <v6 checkpoint>            (driver: runs every config in a subprocess)
python mem_bench.py --cache ... --ckpt ... --one trad|sp_opt --c N --L N  (single measurement)
"""
import argparse, json, subprocess, sys, threading, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, psutil, torch

ROOT = Path(__file__).resolve().parent


def rss_tree_mb():
    p = psutil.Process(); procs = [p] + p.children(recursive=True); tot = 0
    for q in procs:
        try: tot += q.memory_info().rss
        except psutil.Error: pass
    return tot / 2 ** 20, len(procs)


def one(a):
    import v9
    from v6 import MASK, load_cache
    base_py, _ = rss_tree_mb(); torch.set_num_threads(a.c)
    model = v9.v8._load(a.ckpt); _, val = load_cache(a.cache); L = a.L
    peak = [0.0, 0]; stop = threading.Event()
    def sampler():
        while not stop.is_set():
            v, n = rss_tree_mb(); peak[0] = max(peak[0], v); peak[1] = max(peak[1], n); time.sleep(0.005)
    rng = np.random.default_rng(0); seqs = []
    for s in rng.integers(0, len(val) - L - 1, 3):
        y = val[s:s + L].numpy(); m = rng.random(L) < 0.5; seqs.append(np.where(m, MASK, y))
    if a.one == 'trad':
        v9.decode_trad(model, seqs[0], L); idle, nproc = rss_tree_mb()
        th = threading.Thread(target=sampler); th.start()
        for x in seqs: v9.decode_trad(model, x, L)
        stop.set(); th.join()
    else:
        with ProcessPoolExecutor(a.c, mp_context=mp.get_context('spawn'), initializer=v9._init, initargs=(a.ckpt,)) as pool:
            v9.decode_sp(pool, seqs[0], L, a.c, True); idle, nproc = rss_tree_mb()
            th = threading.Thread(target=sampler); th.start()
            for x in seqs: v9.decode_sp(pool, x, L, a.c, True)
            stop.set(); th.join()
    print(json.dumps({'method': a.one, 'c': a.c, 'L': L, 'python_base_mb': base_py, 'idle_after_load_mb': idle,
                      'peak_mb': peak[0], 'processes': max(peak[1], nproc), 'inference_delta_mb': peak[0] - idle}))


def driver(a):
    rows = []
    for L in (512, 1024):
        for c in (1, 2, 4, 6):
            for meth in ('trad', 'sp_opt'):
                out = subprocess.run([sys.executable, __file__, '--cache', a.cache, '--ckpt', a.ckpt, '--one', meth, '--c', str(c), '--L', str(L)],
                                     capture_output=True, text=True, cwd=ROOT)
                line = [l for l in out.stdout.splitlines() if l.startswith('{')]
                if not line: print('FAILED', meth, c, L, out.stderr[-500:], flush=True); continue
                rows.append(json.loads(line[-1])); print(line[-1], flush=True)
    # attention-score memory per layer (fp32, 8 heads): exact arithmetic
    att = {L: {'global_mb': L * L * 8 * 4 / 2 ** 20, 'per_shard_mb': 160 * 160 * 8 * 4 / 2 ** 20} for L in (512, 1024)}
    (ROOT / 'results' / 'memory.json').write_text(json.dumps({'rows': rows, 'attention_scores_per_layer': att}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--one'); ap.add_argument('--c', type=int, default=1); ap.add_argument('--L', type=int, default=1024)
    a = ap.parse_args(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / 'v6'))
    one(a) if a.one else driver(a)
