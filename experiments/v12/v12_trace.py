"""Record REAL scheduling timelines (which window ran on which device, when) for the README animation.

python v12_trace.py --bundle <bundle> --text <txt> --plan 512,256,0 --workers local:xpu:1,local:cpu:4,HOST:PORT,HOST:PORT --tokens 16384
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parents[1] / 'seedplane')); import engine, cli


def static_timeline(clients, ids, wins):
    """Static proportional split (the V10 scheduler): one message per worker; record each worker's block."""
    speed = {}
    for name, c in clients:
        c.send(('windows', ids, wins[:1], 'prefill')); c.recv(); c.send(('windows', ids, wins[:1], 'prefill')); speed[name] = 1.0 / max(c.recv()[1], 1e-6)
    load = {n: 0.0 for n, _ in clients}; assign = {n: [] for n, _ in clients}
    for k in range(len(wins)):
        best = min(clients, key=lambda nc: (load[nc[0]] + 1) / speed[nc[0]])[0]; assign[best].append(k); load[best] += 1
    t0 = time.perf_counter(); tl = []
    for name, c in clients:
        if assign[name]: c.send(('windows', ids, [wins[k] for k in assign[name]], 'prefill'))
    from multiprocessing.connection import wait
    pending = {c: n for n, c in clients if assign[n]}
    while pending:
        for conn in wait(list(pending)):
            conn.recv(); n = pending.pop(conn); tl.append({'worker': n, 'windows': assign[n], 'start': 0.0, 'end': time.perf_counter() - t0})
    return tl, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle'); ap.add_argument('--text'); ap.add_argument('--plan'); ap.add_argument('--workers'); ap.add_argument('--tokens', type=int, default=16384)
    a = ap.parse_args(); S, H, K = map(int, a.plan.split(','))
    from transformers import AutoTokenizer
    ids = np.array(AutoTokenizer.from_pretrained(a.bundle)(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)[:a.tokens]
    wins = list(engine.ShardPlan(S, H, K).windows(len(ids))); procs, clients = cli.start_workers(a.workers, a.bundle); out = {'tokens': len(ids), 'windows': len(wins), 'workers': [n for n, _ in clients]}
    try:
        cli.distribute_dynamic(clients, ids, wins[:4], 'prefill')                    # warm every worker
        out['static'], out['static_s'] = static_timeline(clients, ids, wins)
        tl = []; _, out['dynamic_s'], out['dynamic_counts'] = cli.distribute_dynamic(clients, ids, wins, 'prefill', timeline=tl); out['dynamic'] = tl
        gpu = [(n, c) for n, c in clients if n.startswith('xpu')]
        if gpu: _, out['gpu_only_s'], _ = cli.distribute_dynamic(gpu, ids, wins, 'prefill')
    finally:
        for _, c in clients:
            try: c.send(('close',))
            except Exception: pass
        for p in procs: p.terminate()
    (ROOT / 'results' / 'trace.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k not in ('static', 'dynamic')}, indent=1))


if __name__ == '__main__':
    main()
