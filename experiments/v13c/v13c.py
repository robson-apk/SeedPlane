"""V13c quality x speed frontier (see PROTOCOL.md). Runs on the GPU box; saves results/frontier[_n].json (never overwrites)."""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parents[1]))
from seedplane import engine, llama_backend as lb

L, CHUNKS, REPS = 16384, (3, 4, 5), 3


def start(exe, gguf, port, ctx, extra=()):
    return subprocess.Popen([exe, '-m', gguf, '--dev', 'Vulkan0', '-c', str(ctx), '-b', str(min(ctx, 4096)), '--port', str(port), '--host', '127.0.0.1', *extra],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument(k) for k in ('--exe', '--gguf', '--bundle', '--text')]
    ap.add_argument('--L', type=int, default=16384); ap.add_argument('--chunks', default='3,4,5'); ap.add_argument('--configs', default='')
    ap.add_argument('--tag', default='frontier'); a = ap.parse_args()
    global L, CHUNKS; L = a.L; CHUNKS = tuple(int(c) for c in a.chunks.split(','))
    out = ROOT / 'results'; out.mkdir(exist_ok=True); fp = out / f'{a.tag}.json'; k = 1
    while fp.exists(): fp = out / f'{a.tag}_{k}.json'; k += 1
    from transformers import AutoTokenizer
    ids_all = np.array(AutoTokenizer.from_pretrained(a.bundle)(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)
    chunks = {c: ids_all[c * L:(c + 1) * L] for c in CHUNKS}; res = {'L': L, 'chunks': CHUNKS, 'full': {}, 'configs': []}
    save = lambda: fp.write_text(json.dumps(res, indent=1)); port = 54300
    p = start(a.exe, a.gguf, port, L + 64)
    try:
        w = lb.connect([f'127.0.0.1:{port}'])[0]
        for c, ids in chunks.items():
            r = lb.run_windows([w], ids, [(0, L, np.arange(L))], 'nll')[0][0]; res['full'][c] = r[0] / r[1]
        w.close()
    finally: p.kill()
    save(); print('full', res['full'], flush=True)
    configs = [('windows', 512, 256, 0), ('windows', 512, 256, 4), ('windows', 1024, 1024, 4), ('windows', 2048, 2048, 4), ('windows', 4096, 4096, 4),
               ('span', 512, 256, 4), ('span', 512, 1024, 4), ('span', 512, 2048, 4), ('span', 512, 4096, 4), ('span', 1024, 8192, 4)]
    if a.configs: configs = [(m, int(x), int(y), int(z)) for m, x, y, z in (c.split(':') for c in a.configs.split(','))]
    for mode, S, H, K in configs:
        port += 1; ctx = S + H + K + 64 if mode == 'windows' else H + K + S + 64
        extra = ('--span-chunk', str(S), '--span-keep', str(H), '--span-sinks', str(K)) if mode == 'span' else ()
        p = start(a.exe, a.gguf, port, ctx, extra)
        try:
            w = lb.connect([f'127.0.0.1:{port}'])[0]; row = {'mode': mode, 'S': S, 'H': H, 'sinks': K, 'ppl_ratio': {}}
            for c, ids in chunks.items():
                if mode == 'windows': rs = lb.run_windows([w], ids, list(engine.ShardPlan(S, H, K).windows(L)), 'nll')[0]
                else: rs = lb.run_windows([w], ids, [(0, L, np.arange(L))], 'span_nll')[0]
                row['ppl_ratio'][c] = float(np.exp(sum(r[0] for r in rs) / sum(r[1] for r in rs) - res['full'][c]))
            ids = chunks[CHUNKS[0]]
            wins = list(engine.ShardPlan(S, H, K).windows(L)) if mode == 'windows' else [(0, L, np.arange(L))]
            want = 'prefill' if mode == 'windows' else 'span'
            lb.run_windows([w], ids, wins, want); ts = []
            for _ in range(REPS): t0 = time.perf_counter(); lb.run_windows([w], ids, wins, want); ts.append(time.perf_counter() - t0)
            row['tok_s'] = L / float(np.median(ts)); row['ppl_ratio_mean'] = float(np.mean(list(row['ppl_ratio'].values()))); w.close()
        finally: p.kill()
        res['configs'].append(row); save(); print(mode, S, H, K, round(row['ppl_ratio_mean'], 4), round(row['tok_s']), flush=True)
    print('saved', fp)


if __name__ == '__main__':
    main()
