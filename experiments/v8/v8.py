"""SeedPlane V8: head-to-head on the SAME model (V6 checkpoint): speed AND quality. See PROTOCOL.md.

python v8.py --cache <tinystories_word1024_cache.pt> --ckpt <v6_mdlm_d256_l6_seed1.pt> [--n_seq 32] [--quick]
"""
import argparse, json, math, sys, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'
sys.path.insert(0, str(ROOT.parent / 'v6'))
from v6 import MDLM, load_cache, MASK, SHARD  # same model class and constants as V6

HALO = 16; K = 16; RATE = 0.5; SEEDS = (11, 23, 37); WORKERS = 4
_M = None


def _load(ckpt):
    ck = torch.load(ckpt, map_location='cpu'); m = MDLM(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval(); return m


def _init(ckpt):
    global _M
    torch.set_num_threads(1); _M = _load(ckpt)


def _shard_job(args):
    """Worker: own shards only; each shard sees itself + HALO tokens of each neighbor. Returns (pred, conf) of owned cores."""
    x, shards, L = args; out = []
    with torch.inference_mode():
        for c0, c1 in shards:
            q0, q1 = max(0, c0 - HALO), min(L, c1 + HALO)
            z = _M(torch.from_numpy(x[q0:q1])[None], torch.arange(q0, q1)[None])[0, c0 - q0:c1 - q0].float()
            p = z.softmax(-1); p[:, MASK] = 0; conf, pred = p.max(-1); out.append((c0, pred.numpy(), conf.numpy()))
    return out


def commit(x, pred, conf, t, L):
    masked = x == MASK; conf = np.where(masked, conf, -1.0)
    for c0 in range(0, L, SHARD):
        c1 = c0 + SHARD; mc = masked[c0:c1]; n = int(math.ceil(mc.sum() / (K - t)))
        if n == 0: continue
        order = np.argsort(-conf[c0:c1], kind='stable')[:n]; order = order[mc[order]]
        x[c0 + order] = pred[c0 + order]


def decode_traditional(model, x0, L):
    x = x0.copy()
    with torch.inference_mode():
        for t in range(K):
            p = model(torch.from_numpy(x)[None], torch.arange(L)[None])[0].float().softmax(-1); p[:, MASK] = 0
            conf, pred = p.max(-1); commit(x, pred.numpy(), conf.numpy(), t, L)
    return x


def decode_seedplane(pool, x0, L):
    x = x0.copy(); shards = [(c0, c0 + SHARD) for c0 in range(0, L, SHARD)]
    groups = [shards[i::WORKERS] for i in range(WORKERS)]
    for t in range(K):
        pred = np.empty(L, np.int64); conf = np.empty(L, np.float32)
        for res in pool.map(_shard_job, [(x, g, L) for g in groups]):
            for c0, pr, cf in res: pred[c0:c0 + SHARD] = pr; conf[c0:c0 + SHARD] = cf
        commit(x, pred, conf, t, L)
    return x


def onestep_nll(model, y, rate, g, L):
    m = torch.rand(L, generator=g) < rate; x = y.masked_fill(m, MASK)
    with torch.inference_mode():
        zg = model(x[None], torch.arange(L)[None])[0].float()
        parts = []
        for c0 in range(0, L, SHARD):
            q0, q1 = max(0, c0 - HALO), min(L, c0 + SHARD + HALO)
            parts.append(model(x[q0:q1][None], torch.arange(q0, q1)[None])[0, c0 - q0:c0 - q0 + SHARD].float())
        zs = torch.cat(parts)
    return float(F.cross_entropy(zg[m], y[m], reduction='sum')), float(F.cross_entropy(zs[m], y[m], reduction='sum')), int(m.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_seq', type=int, default=32); ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
    torch.set_num_threads(4); model = _load(a.ckpt); _, val = load_cache(a.cache); rows = []
    Ls = (512,) if a.quick else (512, 1024)
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('spawn'), initializer=_init, initargs=(a.ckpt,)) as pool:
        list(pool.map(_shard_job, [(np.full(160, MASK, np.int64), [(0, 128)], 160)] * WORKERS))  # warm workers
        for L in Ls:
            decode_traditional(model, np.full(L, MASK, np.int64), L)  # warm
            for seed in SEEDS:
                g = torch.Generator().manual_seed(seed + L)
                starts = torch.randint(0, len(val) - L - 1, (a.n_seq,), generator=g)
                for i, s in enumerate(starts.tolist()):
                    y = val[s:s + L].clone(); m = torch.rand(L, generator=g) < RATE; x0 = y.masked_fill(m, MASK).numpy()
                    order = ['traditional', 'seedplane'] if np.random.default_rng(seed * 1000 + i).random() < 0.5 else ['seedplane', 'traditional']
                    r = {'L': L, 'seed': seed, 'seq': i, 'n_masked': int(m.sum()), 'order': order}
                    for cond in order:
                        t0 = time.perf_counter()
                        xf = decode_traditional(model, x0, L) if cond == 'traditional' else decode_seedplane(pool, x0, L)
                        r[f'{cond}_s'] = time.perf_counter() - t0
                        r[f'{cond}_correct'] = int(((torch.from_numpy(xf) == y) & m).sum())
                    for rate in (0.15, 0.5, 0.9):
                        ng, ns, n = onestep_nll(model, y, rate, g, L); r[f'nll{rate}_trad'] = ng; r[f'nll{rate}_seed'] = ns; r[f'nll{rate}_n'] = n
                    rows.append(r)
                print('done L', L, 'seed', seed, flush=True)
    RES.mkdir(exist_ok=True); (RES / 'rows.json').write_text(json.dumps(rows)); analyze(rows, Ls)


def analyze(rows, Ls):
    out = {'per': {}, 'criteria': {}}
    for L in Ls:
        for seed in SEEDS:
            r = [x for x in rows if x['L'] == L and x['seed'] == seed]
            n = np.array([x['n_masked'] for x in r], float); ct = np.array([x['traditional_correct'] for x in r], float); cs = np.array([x['seedplane_correct'] for x in r], float)
            tt = np.array([x['traditional_s'] for x in r]); ts = np.array([x['seedplane_s'] for x in r])
            acc_t, acc_s = ct.sum() / n.sum(), cs.sum() / n.sum()
            ix = np.random.default_rng(seed).integers(0, len(r), (2000, len(r)))
            acc_ci = np.quantile((cs[ix].sum(1) - ct[ix].sum(1)) / n[ix].sum(1), [.025, .975]).tolist()
            ratio_ci = np.quantile(ts[ix].sum(1) / tt[ix].sum(1), [.025, .975]).tolist()
            e = {'n_seq': len(r), 'masked_tokens': int(n.sum()), 'acc_traditional': acc_t, 'acc_seedplane': acc_s, 'acc_diff_pp': 100 * (acc_s - acc_t), 'acc_diff_ci_pp': [100 * v for v in acc_ci],
                 'time_traditional_ms_median': 1000 * float(np.median(tt)), 'time_seedplane_ms_median': 1000 * float(np.median(ts)), 'time_ratio': float(ts.sum() / tt.sum()), 'time_ratio_ci': ratio_ci}
            for rate in (0.15, 0.5, 0.9):
                nt = sum(x[f'nll{rate}_trad'] for x in r); ns = sum(x[f'nll{rate}_seed'] for x in r); nn = sum(x[f'nll{rate}_n'] for x in r)
                e[f'nll{rate}_traditional'] = nt / nn; e[f'nll{rate}_seedplane'] = ns / nn
            out['per'][f'L{L}_s{seed}'] = e
    if 1024 in Ls:
        P = [out['per'][f'L1024_s{s}'] for s in SEEDS]
        out['criteria']['Q'] = all(p['acc_diff_ci_pp'][0] > -0.5 and all(p[f'nll{r}_seedplane'] <= 1.01 * p[f'nll{r}_traditional'] for r in (0.15, 0.5, 0.9)) for p in P)
        out['criteria']['S'] = all(p['time_ratio_ci'][1] < 0.90 for p in P)
        out['verdict'] = {(True, True): 'SAME_OR_BETTER_QUALITY_AND_FASTER', (True, False): 'SAME_QUALITY_NOT_FASTER',
                          (False, True): 'FASTER_BUT_QUALITY_WORSE', (False, False): 'NEITHER'}[(out['criteria']['Q'], out['criteria']['S'])]
    (RES / 'analysis.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
