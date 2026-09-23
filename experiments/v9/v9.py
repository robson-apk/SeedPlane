"""SeedPlane V9: structural optimizations + core scaling 1..6 vs traditional Transformer. See PROTOCOL.md.

python v9.py --cache <cache.pt> --ckpt <v6 checkpoint> [--n_seq 12] [--cores 1,2,3,4,5,6] [--Ls 512,1024]
"""
import argparse, json, math, sys, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'
sys.path.insert(0, str(ROOT.parent / 'v8')); sys.path.insert(0, str(ROOT.parent / 'v6'))
import v8                                  # unchanged V8 implementation (baseline sp_v8)
from v6 import MASK, SHARD, load_cache

HALO, K, RATE, SEEDS = 16, 16, 0.5, (11, 23, 37)
_M = None


# ---------------------------------------------------------------- shared model helpers
def hidden(model, x, pos, pad=None):
    return model.tr(model.emb(x) + model.pos(pos), src_key_padding_mask=pad)


def logits_at(model, h):
    return F.linear(model.norm(h), model.emb.weight) * model.scale + model.bias


def decide(z):
    p = z.float().softmax(-1); p[..., MASK] = 0; conf, pred = p.max(-1); return pred.numpy(), conf.numpy()


# ---------------------------------------------------------------- workers
def _init(ckpt):
    global _M
    torch.set_num_threads(1); _M = v8._load(ckpt); v8._M = _M


def _opt_job(args):
    """O2: owned shards batched by equal window length (no padding: key_padding_mask was slower, see PROTOCOL addendum).
    O1: logits only at masked core positions."""
    x, shards, L = args; out = []; groups = {}
    for c0, c1 in shards:
        q0, q1 = max(0, c0 - HALO), min(L, c1 + HALO); groups.setdefault(q1 - q0, []).append((c0, c1, q0, q1))
    with torch.inference_mode():
        for g in groups.values():
            h = hidden(_M, torch.from_numpy(np.stack([x[q0:q1] for _, _, q0, q1 in g])), torch.from_numpy(np.stack([np.arange(q0, q1) for _, _, q0, q1 in g])))
            for i, (c0, c1, q0, _) in enumerate(g):
                idx = np.nonzero(x[c0:c1] == MASK)[0]
                if len(idx) == 0: continue
                pred, conf = decide(logits_at(_M, h[i, c0 - q0 + idx])); out.append((c0 + idx, pred, conf))
    return out


# ---------------------------------------------------------------- decoders
def commit(x, pred, conf, t, L):
    v8.commit(x, pred, conf, t, L)


def decode_trad(model, x0, L):
    x = x0.copy(); pos = torch.arange(L)[None]
    with torch.inference_mode():
        for t in range(K):
            idx = np.nonzero(x == MASK)[0]; pred = np.zeros(L, np.int64); conf = np.full(L, -1.0, np.float32)
            h = hidden(model, torch.from_numpy(x)[None], pos)[0]
            pr, cf = decide(logits_at(model, h[idx])); pred[idx] = pr; conf[idx] = cf; commit(x, pred, conf, t, L)
    return x


def groups_rr(L, c):
    shards = [(c0, c0 + SHARD) for c0 in range(0, L, SHARD)]; return [shards[i::c] for i in range(c)]


def groups_lpt(L, c):
    """O3: longest-processing-time assignment by window length (edge shards are cheaper)."""
    shards = [(c0, c0 + SHARD) for c0 in range(0, L, SHARD)]
    cost = lambda s: min(L, s[1] + HALO) - max(0, s[0] - HALO)
    g = [[] for _ in range(c)]; load = [0] * c
    for s in sorted(shards, key=cost, reverse=True):
        j = int(np.argmin(load)); g[j].append(s); load[j] += cost(s)
    return g


def decode_sp(pool, x0, L, c, opt):
    x = x0.copy(); groups = groups_lpt(L, c) if opt else groups_rr(L, c)
    for t in range(K):
        pred = np.zeros(L, np.int64); conf = np.full(L, -1.0, np.float32)
        if opt:
            for res in pool.map(_opt_job, [(x, g, L) for g in groups]):
                for idx, pr, cf in res: pred[idx] = pr; conf[idx] = cf
        else:
            for res in pool.map(v8._shard_job, [(x, g, L) for g in groups]):
                for c0, pr, cf in res: pred[c0:c0 + SHARD] = pr; conf[c0:c0 + SHARD] = cf
        commit(x, pred, conf, t, L)
    return x


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_seq', type=int, default=12); ap.add_argument('--cores', default='1,2,3,4,5,6'); ap.add_argument('--Ls', default='512,1024')
    a = ap.parse_args(); cores = [int(v) for v in a.cores.split(',')]; Ls = [int(v) for v in a.Ls.split(',')]
    model = v8._load(a.ckpt); _, val = load_cache(a.cache); rows = []
    for c in cores:
        torch.set_num_threads(c)
        with ProcessPoolExecutor(c, mp_context=mp.get_context('spawn'), initializer=_init, initargs=(a.ckpt,)) as pool:
            for L in Ls:
                w = np.full(L, MASK, np.int64); decode_trad(model, w, L); decode_sp(pool, w, L, c, False); decode_sp(pool, w, L, c, True)  # warm
                for seed in SEEDS:
                    g = torch.Generator().manual_seed(seed + L)
                    for i, s in enumerate(torch.randint(0, len(val) - L - 1, (a.n_seq,), generator=g).tolist()):
                        y = val[s:s + L].numpy(); m = torch.rand(L, generator=g).numpy() < RATE; x0 = np.where(m, MASK, y)
                        order = list(np.random.default_rng(1000 * seed + 10 * c + i).permutation(['trad', 'sp_v8', 'sp_opt']))
                        r = {'c': c, 'L': L, 'seed': seed, 'seq': i, 'n_masked': int(m.sum()), 'order': order}; out = {}
                        for meth in order:
                            t0 = time.perf_counter()
                            xf = decode_trad(model, x0, L) if meth == 'trad' else decode_sp(pool, x0, L, c, meth == 'sp_opt')
                            r[f'{meth}_s'] = time.perf_counter() - t0; r[f'{meth}_correct'] = int(((xf == y) & m).sum()); out[meth] = xf
                        r['agree_opt_v8'] = float((out['sp_opt'][m] == out['sp_v8'][m]).mean()); rows.append(r)
                    print(f'c={c} L={L} seed={seed} done', flush=True)
    RES.mkdir(exist_ok=True); (RES / 'rows.json').write_text(json.dumps(rows)); analyze(rows)


def analyze(rows=None):
    rows = rows or json.loads((RES / 'rows.json').read_text()); out = {'per': {}, 'curves': {}, 'criteria': {}}
    cores = sorted({r['c'] for r in rows}); Ls = sorted({r['L'] for r in rows})
    def ratio_ci(a, b, seed):
        ix = np.random.default_rng(seed).integers(0, len(a), (2000, len(a))); return np.quantile(a[ix].sum(1) / b[ix].sum(1), [.025, .975]).tolist()
    for L in Ls:
        for c in cores:
            for seed in SEEDS:
                r = [x for x in rows if x['L'] == L and x['c'] == c and x['seed'] == seed]
                if not r: continue
                T = {k: np.array([x[f'{k}_s'] for x in r]) for k in ('trad', 'sp_v8', 'sp_opt')}; n = sum(x['n_masked'] for x in r)
                out['per'][f'L{L}_c{c}_s{seed}'] = {
                    **{f'{k}_ms_median': 1000 * float(np.median(v)) for k, v in T.items()},
                    'opt_over_trad': float(T['sp_opt'].sum() / T['trad'].sum()), 'opt_over_trad_ci': ratio_ci(T['sp_opt'], T['trad'], seed),
                    'opt_over_v8': float(T['sp_opt'].sum() / T['sp_v8'].sum()), 'opt_over_v8_ci': ratio_ci(T['sp_opt'], T['sp_v8'], seed),
                    **{f'{k}_acc': sum(x[f'{k}_correct'] for x in r) / n for k in T}, 'agree_opt_v8': float(np.mean([x['agree_opt_v8'] for x in r]))}
        for k in ('trad', 'sp_v8', 'sp_opt'):
            med = {c: 1000 * float(np.median([x[f'{k}_s'] for x in rows if x['L'] == L and x['c'] == c])) for c in cores}
            out['curves'][f'L{L}_{k}'] = {'ms': med, 'speedup_vs_1core': {c: med[cores[0]] / med[c] for c in cores}, 'efficiency': {c: med[cores[0]] / med[c] / c for c in cores}}
    if 1024 in Ls:
        P = lambda c, s: out['per'].get(f'L1024_c{c}_s{s}')
        out['criteria']['P1'] = all(P(c, s)['opt_over_trad_ci'][1] < 1 for c in cores for s in SEEDS)
        out['criteria']['P2'] = all(P(4, s)['opt_over_v8_ci'][1] < 0.90 for s in SEEDS) if 4 in cores else None
        out['criteria']['P3'] = all(P(c, s)['agree_opt_v8'] >= 0.99 and abs(P(c, s)['sp_opt_acc'] - P(c, s)['sp_v8_acc']) <= 0.003 for c in cores for s in SEEDS)
    (RES / 'analysis.json').write_text(json.dumps(out, indent=1)); print(json.dumps({'criteria': out['criteria'], 'curves': out['curves']}, indent=1))


if __name__ == '__main__':
    main()
