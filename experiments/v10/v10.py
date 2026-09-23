"""SeedPlane V10: heterogeneous devices (GPU + CPU cores + remote Mac cores). See PROTOCOL.md.

python v10.py serve --port P --device cpu|xpu --ckpt C          # a worker endpoint (one per core / GPU)
python v10.py run --ckpt C --cache D --mac HOST:PORT,HOST:PORT   # coordinator (starts local endpoints itself)
python v10.py analyze
"""
import argparse, json, math, subprocess, sys, time
from multiprocessing.connection import Listener, Client
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'
sys.path.insert(0, str(ROOT.parent / 'v8')); sys.path.insert(0, str(ROOT.parent / 'v6'))
from v6 import MASK, SHARD, load_cache
import v8

HALO, K, RATE, SEEDS, AUTH = 16, 16, 0.5, (11, 23, 37), b'seedplane-v10'


# ---------------------------------------------------------------- endpoint (worker server)
def serve(a):
    torch.set_num_threads(1); dev = torch.device(a.device)
    model = v8._load(a.ckpt).to(dev); CH = 64 if a.device != 'cpu' else 10 ** 9
    def sync():
        if dev.type == 'xpu': torch.xpu.synchronize()
    def decide(z):
        p = z.float().softmax(-1); p[..., MASK] = 0; c, pr = p.max(-1); sync(); return pr.cpu().numpy(), c.cpu().numpy()
    def run_shards(xs, units, L):
        groups = {}
        for k, (pi, c0) in enumerate(units):
            q0, q1 = max(0, c0 - HALO), min(L, c0 + SHARD + HALO); groups.setdefault(q1 - q0, []).append((k, pi, c0, q0, q1))
        out = []
        with torch.inference_mode():
            for g in groups.values():
                for i in range(0, len(g), CH):
                    sub = g[i:i + CH]
                    xb = torch.from_numpy(np.stack([xs[pi, q0:q1] for _, pi, _, q0, q1 in sub]).astype(np.int64)).to(dev)
                    pb = torch.from_numpy(np.stack([np.arange(q0, q1) for *_, q0, q1 in sub])).to(dev)
                    h = model.tr(model.emb(xb) + model.pos(pb))
                    cores = torch.stack([h[j, c0 - q0:c0 - q0 + SHARD] for j, (_, _, c0, q0, _) in enumerate(sub)])
                    pr, cf = decide(F.linear(model.norm(cores), model.emb.weight) * model.scale + model.bias)
                    for j, (_, pi, c0, _, _) in enumerate(sub):
                        idx = np.nonzero(xs[pi, c0:c0 + SHARD] == MASK)[0]; out.append((pi, c0 + idx, pr[j, idx], cf[j, idx]))
        return out
    def run_global(xs, L):
        out = []
        with torch.inference_mode():
            for i in range(0, len(xs), max(1, CH // 8)):
                xb = torch.from_numpy(xs[i:i + CH // 8 or 1].astype(np.int64)).to(dev); pos = torch.arange(L, device=dev).expand(len(xb), -1)
                pr, cf = decide(model(xb, pos))
                for j in range(len(xb)):
                    idx = np.nonzero(xs[i + j] == MASK)[0]; out.append((i + j, idx, pr[j, idx], cf[j, idx]))
        return out
    with Listener(('0.0.0.0', a.port), authkey=AUTH) as lst:
        print(f'endpoint ready {a.device}:{a.port}', flush=True)
        while True:
            conn = lst.accept()
            try:
                while True:
                    msg = conn.recv()
                    if msg[0] == 'close': break
                    t0 = time.perf_counter()
                    if msg[0] == 'shards': res = run_shards(msg[1], msg[2], msg[3])
                    elif msg[0] == 'global': res = run_global(msg[1], msg[2])
                    else: res = 'pong'
                    conn.send((res, time.perf_counter() - t0))
            except (EOFError, ConnectionResetError): pass
            finally: conn.close()


# ---------------------------------------------------------------- coordinator
class EP:
    def __init__(self, name, group, host, port):
        self.name, self.group, self.host, self.port = name, group, host, port
        for _ in range(120):
            try: self.conn = Client((host, port), authkey=AUTH); break
            except OSError: time.sleep(0.5)
        else: raise RuntimeError(f'cannot reach {name}')
    def call(self, *msg): self.conn.send(msg)
    def get(self): return self.conn.recv()


def calibrate(eps, x_pages, L, n_units, n_pages):
    """Units/s for shard jobs and pages/s for global jobs, measured on each endpoint with a job of the given size."""
    units = [(p % len(x_pages), c0) for p in range(n_pages) for c0 in range(0, L, SHARD)][:n_units]
    sp, gl = {}, {}
    for e in eps:
        for _ in range(2): e.call('shards', x_pages, units, L); e.get()          # warm
        t = []; [(e.call('shards', x_pages, units, L), t.append(e.get()[1])) for _ in range(3)]; sp[e.name] = len(units) / np.median(t)
        e.call('global', x_pages[:n_pages], L); e.get()
        t = []; [(e.call('global', x_pages[:n_pages], L), t.append(e.get()[1])) for _ in range(3)]; gl[e.name] = n_pages / np.median(t)
    return sp, gl


def split(items, eps, speed):
    load = {e.name: 0.0 for e in eps}; out = {e.name: [] for e in eps}
    for it in items:
        best = min(eps, key=lambda e: (load[e.name] + 1) / speed[e.name]); out[best.name].append(it); load[best.name] += 1
    return out


def decode(eps, x0, L, method, speed):
    x = x0.copy(); P = len(x); steps = []
    if method == 'sp': plan = split([(p, c0) for p in range(P) for c0 in range(0, L, SHARD)], eps, speed)
    else: plan = split(list(range(P)), eps, speed)
    t_all = time.perf_counter()
    for t in range(K):
        ts = time.perf_counter(); pred = np.zeros((P, L), np.int64); conf = np.full((P, L), -1.0, np.float32); busy = []
        active = [e for e in eps if plan[e.name]]
        for e in active:
            if method == 'sp':
                pages = sorted({p for p, _ in plan[e.name]}); remap = {p: i for i, p in enumerate(pages)}
                e._pages = pages; e.call('shards', x[pages].astype(np.int16), [(remap[p], c0) for p, c0 in plan[e.name]], L)
            else:
                e._pages = plan[e.name]; e.call('global', x[plan[e.name]].astype(np.int16), L)
        for e in active:
            res, comp = e.get(); busy.append(comp)
            for pi, idx, pr, cf in res: pred[e._pages[pi], idx] = pr; conf[e._pages[pi], idx] = cf
        for p in range(P): v8.commit(x[p], pred[p], conf[p], t, L)
        steps.append({'wall': time.perf_counter() - ts, 'max_compute': max(busy)})
    return x, time.perf_counter() - t_all, steps


def run(a):
    torch.set_num_threads(1); _, val = load_cache(a.cache); L = 1024; py = sys.executable; procs = []
    local = [('gpu', 'gpu', 'xpu', 51000)] + [(f'cpu{i}', 'cpu4', 'cpu', 51001 + i) for i in range(4)]
    for name, grp, dev, port in local:
        procs.append(subprocess.Popen([py, __file__, 'serve', '--port', str(port), '--device', dev, '--ckpt', a.ckpt]))
    eps = {name: EP(name, grp, '127.0.0.1', port) for name, grp, _, port in local}
    for i, hp in enumerate(a.mac.split(',')):
        h, p = hp.split(':'); eps[f'mac{i}'] = EP(f'mac{i}', 'mac2', h, int(p))
    sets = {'gpu': ['gpu'], 'cpu4': ['cpu4'], 'gpu+cpu4': ['gpu', 'cpu4'], 'cpu4+mac2': ['cpu4', 'mac2'], 'gpu+cpu4+mac2': ['gpu', 'cpu4', 'mac2']}
    seeds = SEEDS
    if a.quick: sets = {k: v for k, v in sets.items() if k in ('gpu', 'gpu+cpu4+mac2')}; seeds = SEEDS[:1]
    rows = []; calib = {}
    def pages(seed, n):
        g = np.random.default_rng(seed); out = []
        for s in g.integers(0, len(val) - L - 1, n):
            y = val[s:s + L].numpy(); m = g.random(L) < RATE; out.append((y, m))
        return out
    for mode, n_pages in ((('latency', 1), ('throughput', 4)) if a.quick else (('latency', 1), ('throughput', 32))):
        for sname, groups in sets.items():
            E = [e for e in eps.values() if e.group in groups]
            cal_pages = np.stack([np.where(m, MASK, y) for y, m in pages(999, max(n_pages, 4))]).astype(np.int16)
            sp_speed, gl_speed = calibrate(E, cal_pages, L, n_units=8 * min(n_pages, 8), n_pages=min(n_pages, 8))
            calib[f'{mode}_{sname}'] = {'sp_units_per_s': sp_speed, 'global_pages_per_s': gl_speed}
            for seed in seeds:
                nl = 2 if a.quick else 8
                batches = [pages(seed, nl)[i:i + 1] for i in range(nl)] if mode == 'latency' else [pages(seed, n_pages)] * (1 if a.quick else 3)
                for bi, batch in enumerate(batches):
                    x0 = np.stack([np.where(m, MASK, y) for y, m in batch]); ys = np.stack([y for y, _ in batch]); ms = np.stack([m for _, m in batch])
                    r = {'mode': mode, 'set': sname, 'seed': seed, 'batch': bi, 'n_masked': int(ms.sum()), 'pages': len(batch)}
                    methods = ['sp', 'trad'] if (mode == 'throughput' or sname in ('gpu',)) else ['sp']
                    for meth in np.random.default_rng(seed * 100 + bi).permutation(methods):
                        xf, dt, steps = decode(E, x0, L, meth, sp_speed if meth == 'sp' else gl_speed)
                        r[f'{meth}_s'] = dt; r[f'{meth}_correct'] = int(((xf == ys) & ms).sum())
                        r[f'{meth}_sync_s'] = float(sum(s['wall'] - s['max_compute'] for s in steps)); r[f'{meth}_tokens'] = xf[ms].tolist()
                    rows.append(r)
                print(f'{mode} {sname} seed {seed} done', flush=True)
    for e in eps.values():
        try: e.call('close')
        except Exception: pass
    for p in procs: p.terminate()
    RES.mkdir(exist_ok=True); (RES / ('rows_quick.json' if a.quick else 'rows.json')).write_text(json.dumps({'rows': rows, 'calibration': calib}))
    if not a.quick: analyze()
    else: print(json.dumps([{k: v for k, v in r.items() if not k.endswith('tokens')} for r in rows], indent=0)[:3000])


def analyze(_=None):
    d = json.loads((RES / 'rows.json').read_text()); rows = d['rows']; out = {'cells': {}, 'criteria': {}, 'calibration': d['calibration']}
    ref = {(r['mode'], r['seed'], r['batch']): r['sp_tokens'] for r in rows if r['set'] == 'gpu'}
    for r in rows:
        k = f"{r['mode']}|{r['set']}|{r['seed']}"; c = out['cells'].setdefault(k, {'sp_s': [], 'trad_s': [], 'masked': [], 'agree': [], 'sp_sync_s': [], 'sp_correct': 0, 'trad_correct': 0})
        c['sp_s'].append(r['sp_s']); c['masked'].append(r['n_masked']); c['sp_sync_s'].append(r['sp_sync_s']); c['sp_correct'] += r['sp_correct']
        if 'trad_s' in r: c['trad_s'].append(r['trad_s']); c['trad_correct'] += r['trad_correct']
        rt = ref[(r['mode'], r['seed'], r['batch'])]; c['agree'].append(float(np.mean(np.array(r['sp_tokens']) == np.array(rt))))
    summ = {}
    for k, c in out['cells'].items():
        s = {'sp_ms_median': 1000 * float(np.median(c['sp_s'])), 'sp_tok_s': float(sum(c['masked']) / sum(c['sp_s'])),
             'sp_sync_share': float(sum(c['sp_sync_s']) / sum(c['sp_s'])), 'agree_vs_gpu': float(np.mean(c['agree'])),
             'sp_acc': c['sp_correct'] / sum(c['masked']), 'sp_s': c['sp_s']}
        if c['trad_s']: s.update({'trad_ms_median': 1000 * float(np.median(c['trad_s'])), 'trad_tok_s': float(sum(c['masked']) / sum(c['trad_s'])), 'trad_acc': c['trad_correct'] / sum(c['masked']), 'trad_s': c['trad_s']})
        summ[k] = s
    out['cells'] = summ
    def ci_ratio(a, b, seed):
        a, b = np.array(a), np.array(b); ix = np.random.default_rng(seed).integers(0, len(a), (2000, len(a))); return np.quantile(a[ix].sum(1) / b[ix].sum(1), [.025, .975]).tolist()
    multi = [s for s in ('gpu+cpu4', 'cpu4+mac2', 'gpu+cpu4+mac2')]
    best = min(multi, key=lambda s: np.mean([summ[f'latency|{s}|{sd}']['sp_ms_median'] for sd in SEEDS]))
    out['criteria']['best_latency_set'] = best
    out['criteria']['H1'] = all(ci_ratio(summ[f'latency|{best}|{sd}']['sp_s'], summ[f'latency|gpu|{sd}']['sp_s'], sd)[1] < 0.90 for sd in SEEDS)
    out['criteria']['H2'] = all(ci_ratio(summ[f'latency|{best}|{sd}']['sp_s'], summ[f'latency|gpu|{sd}']['trad_s'], sd)[1] < 1.0 for sd in SEEDS)
    out['criteria']['H3'] = all(summ[f'throughput|gpu+cpu4+mac2|{sd}']['sp_tok_s'] >= 1.10 * summ[f'throughput|gpu|{sd}']['sp_tok_s'] for sd in SEEDS)
    out['criteria']['H4'] = all(summ[f'throughput|gpu+cpu4+mac2|{sd}']['sp_tok_s'] >= summ[f'throughput|gpu+cpu4+mac2|{sd}']['trad_tok_s'] for sd in SEEDS)
    out['criteria']['Q'] = all(v['agree_vs_gpu'] >= 0.99 for v in summ.values())
    for v in summ.values(): v.pop('sp_s', None); v.pop('trad_s', None)
    (RES / 'analysis.json').write_text(json.dumps(out, indent=1)); print(json.dumps({'criteria': out['criteria']}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['serve', 'run', 'analyze'])
    ap.add_argument('--port', type=int); ap.add_argument('--device', default='cpu'); ap.add_argument('--ckpt'); ap.add_argument('--cache'); ap.add_argument('--mac', default=''); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(); {'serve': serve, 'run': run, 'analyze': analyze}[a.mode](a)
