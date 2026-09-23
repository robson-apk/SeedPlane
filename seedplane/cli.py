"""seedplane — run existing Hugging Face models with SeedPlane shard windows, on any mix of devices.

  seedplane convert Qwen/Qwen2.5-0.5B-Instruct ./qwen05.sp --shard 512 --halo 256 --sinks 4
  seedplane serve   --bundle ./qwen05.sp --device cpu --port 52000        # turn this machine/core/GPU into a worker
  seedplane run     --bundle ./qwen05.sp --prompt-file long.txt --workers local:cpu:4,local:xpu:1,10.0.0.92:52000
  seedplane bench   --bundle ./qwen05.sp --text-file corpus.txt --length 4096

A worker is one process on one device (a CPU core, a GPU, or another computer over the LAN).
`run` splits the prompt into shard windows, sends each window to a worker in proportion to its measured speed,
and stitches the results. Windows are independent, so there is no synchronization between workers.
"""
import argparse, json, subprocess, sys, time
from multiprocessing.connection import Listener, Client
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine

AUTH = b'seedplane-cli'


def plan_of(bundle):
    cfg = json.loads((Path(bundle) / 'seedplane.json').read_text())['plan']; return engine.ShardPlan(**cfg)


# ------------------------------------------------------------------ convert
def cmd_convert(a):
    out = engine.save_bundle(a.out, a.model, engine.ShardPlan(a.shard, a.halo, a.sinks))
    print(f'bundle written to {out}  (weights unchanged + seedplane.json)')


# ------------------------------------------------------------------ serve (a worker)
def cmd_serve(a):
    torch.set_num_threads(a.threads); model, _ = engine.load_model(a.bundle, a.device)
    with Listener((a.host, a.port), authkey=AUTH) as lst:
        print(f'seedplane worker ready on {a.host}:{a.port} ({a.device}, {a.threads} thread(s))', flush=True)
        while True:
            conn = lst.accept()
            try:
                while True:
                    msg = conn.recv()
                    if msg[0] == 'close': break
                    t0 = time.perf_counter()
                    if msg[0] == 'windows':                      # (ids, [(c0, c1, index), ...]) -> NLL / next-token argmax per window
                        _, ids, wins, want = msg; res = []
                        for c0, c1, idx in wins:
                            if want == 'prefill':
                                res.append(engine.prefill_window(model, ids, idx, a.device, last=(c1 == len(ids)))); continue
                            z = engine.window_logits(model, ids, idx, a.device)[-(c1 - c0):]
                            if want == 'nll':
                                last = min(c1, len(ids) - 1); y = torch.as_tensor(ids[c0 + 1:last + 1], device=z.device)
                                res.append(float(F.cross_entropy(z[:last - c0], y, reduction='sum')) if last > c0 else 0.0)
                            else: res.append(int(z[-1].argmax()))
                        if a.device == 'xpu': torch.xpu.synchronize()
                        conn.send((res, time.perf_counter() - t0))
                    else: conn.send(('pong', 0.0))
            except (EOFError, ConnectionResetError): pass
            finally: conn.close()


# ------------------------------------------------------------------ run / bench (the coordinator)
def start_workers(spec, bundle):
    """spec: 'local:cpu:4,local:xpu:1,HOST:PORT' -> list of (name, Client). Local workers are spawned here."""
    procs, conns, port = [], [], 53000
    for item in spec.split(','):
        parts = item.split(':')
        if parts[0] == 'local':
            dev, n = parts[1], int(parts[2])
            for _ in range(n):
                procs.append(subprocess.Popen([sys.executable, __file__, 'serve', '--bundle', bundle, '--device', dev, '--port', str(port), '--host', '127.0.0.1']))
                conns.append((f'{dev}:{port}', '127.0.0.1', port)); port += 1
        else:
            conns.append((item, parts[0], int(parts[1])))
    clients = []
    for name, h, p in conns:
        for _ in range(240):
            try: clients.append((name, Client((h, p), authkey=AUTH))); break
            except OSError: time.sleep(0.5)
        else: raise RuntimeError(f'worker {name} unreachable')
    return procs, clients


def distribute(clients, ids, wins, want):
    """Calibrate each worker on one window, then assign windows proportionally to speed; returns results in window order."""
    speed = {}
    for name, c in clients:
        c.send(('windows', ids, wins[:1], want)); c.recv(); c.send(('windows', ids, wins[:1], want)); speed[name] = 1.0 / max(c.recv()[1], 1e-6)
    load = {n: 0.0 for n, _ in clients}; assign = {n: [] for n, _ in clients}
    for k in range(len(wins)):
        best = min(clients, key=lambda nc: (load[nc[0]] + 1) / speed[nc[0]])[0]; assign[best].append(k); load[best] += 1
    t0 = time.perf_counter(); out = [None] * len(wins)
    for name, c in clients:
        if assign[name]: c.send(('windows', ids, [wins[k] for k in assign[name]], want))
    for name, c in clients:
        if assign[name]:
            res, _ = c.recv()
            for k, v in zip(assign[name], res): out[k] = v
    return out, time.perf_counter() - t0, {n: len(v) for n, v in assign.items()}


def distribute_dynamic(clients, ids, wins, want, prior=None):
    """Pull-based work queue with a tail guard.
    Each worker gets one window at a time and asks for more when done (fast devices naturally take more).
    Per-window time of each worker is tracked (EMA, seeded by `prior` or a 1-window probe). A worker only receives a
    window if it would finish it before the rest of the pool would finish the whole remaining queue without it
    (so a slow core or a far-away machine never becomes the straggler that holds the last window)."""
    from multiprocessing.connection import wait
    est = dict(prior or {})
    for name, c in clients:
        if name not in est:
            c.send(('windows', ids, wins[:1], want)); c.recv(); t1 = time.perf_counter(); c.send(('windows', ids, wins[:1], want)); c.recv(); est[name] = time.perf_counter() - t1
    queue = list(range(len(wins))); out = [None] * len(wins); busy = {}; count = {n: 0 for n, _ in clients}
    by_conn = {c: n for n, c in clients}; t0 = time.perf_counter(); sent_at = {}

    def worth_it(name):
        others = [n for n, _ in clients if n != name]
        if not others: return True
        rate_others = sum(1.0 / est[n] for n in others)
        return est[name] <= len(queue) / rate_others + 1e-9

    def feed(name, c):
        if queue and (worth_it(name) or all(n not in busy for n, _ in clients if n != name)):
            k = queue.pop(0); c.send(('windows', ids, [wins[k]], want)); busy[name] = k; sent_at[name] = time.perf_counter(); return True
        return False

    for name, c in sorted(clients, key=lambda nc: est[nc[0]]):
        feed(name, c)
    while busy:
        for conn in wait([c for n, c in clients if n in busy]):
            name = by_conn[conn]; res, _ = conn.recv(); k = busy.pop(name); out[k] = res[0]; count[name] += 1
            est[name] = 0.7 * est[name] + 0.3 * (time.perf_counter() - sent_at[name])   # round-trip time, network included
            feed(name, conn)
        for name, c in clients:
            if name not in busy and queue: feed(name, c)
    return out, time.perf_counter() - t0, count


def cmd_run(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.bundle)
    ids = np.array(tok(Path(a.prompt_file).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)[: a.max_tokens]
    plan = plan_of(a.bundle); wins = list(plan.windows(len(ids))); procs, clients = start_workers(a.workers, a.bundle)
    try:
        nll, dt, split = (distribute_dynamic if a.scheduler == 'dynamic' else distribute)(clients, ids, wins, 'nll')
        print(json.dumps({'tokens': len(ids), 'windows': len(wins), 'seconds': dt, 'tokens_per_second': len(ids) / dt,
                          'perplexity': float(np.exp(sum(nll) / (len(ids) - 1))), 'windows_per_worker': split}, indent=1))
    finally:
        for _, c in clients:
            try: c.send(('close',))
            except Exception: pass
        for p in procs: p.terminate()


def cmd_bench(a):
    dev = a.device; model, tok = engine.load_model(a.bundle, dev); plan = plan_of(a.bundle)
    ids = np.array(tok(Path(a.text_file).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)
    rng = np.random.default_rng(0); res = []
    for _ in range(a.samples):
        s = int(rng.integers(0, len(ids) - a.length - 1)); seq = ids[s:s + a.length]
        t0 = time.perf_counter(); nf, n = engine.nll_full(model, seq, dev); tf = time.perf_counter() - t0
        t0 = time.perf_counter(); ns, _ = engine.nll_shards(model, seq, plan, dev); ts = time.perf_counter() - t0
        res.append((nf, ns, n, tf, ts))
    nf, ns, n, tf, ts = map(np.sum, zip(*res))
    print(json.dumps({'device': dev, 'length': a.length, 'samples': a.samples, 'plan': plan.__dict__,
                      'ppl_full': float(np.exp(nf / n)), 'ppl_seedplane': float(np.exp(ns / n)), 'ppl_ratio': float(np.exp((ns - nf) / n)),
                      'seconds_full': float(tf), 'seconds_seedplane': float(ts), 'speedup': float(tf / ts)}, indent=1))


def main():
    ap = argparse.ArgumentParser(prog='seedplane', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('convert', help='make a SeedPlane bundle from a Hugging Face model'); c.add_argument('model'); c.add_argument('out')
    c.add_argument('--shard', type=int, default=512); c.add_argument('--halo', type=int, default=256); c.add_argument('--sinks', type=int, default=4)
    s = sub.add_parser('serve', help='run a worker on this device'); s.add_argument('--bundle', required=True); s.add_argument('--device', default='cpu')
    s.add_argument('--port', type=int, default=52000); s.add_argument('--host', default='0.0.0.0'); s.add_argument('--threads', type=int, default=1)
    r = sub.add_parser('run', help='process a long prompt across workers'); r.add_argument('--bundle', required=True); r.add_argument('--prompt-file', required=True)
    r.add_argument('--workers', default='local:cpu:4'); r.add_argument('--scheduler', choices=['dynamic', 'static'], default='dynamic'); r.add_argument('--max-tokens', type=int, default=8192)
    b = sub.add_parser('bench', help='quality + speed vs the original full attention'); b.add_argument('--bundle', required=True); b.add_argument('--text-file', required=True)
    b.add_argument('--device', default='cpu'); b.add_argument('--length', type=int, default=4096); b.add_argument('--samples', type=int, default=3)
    a = ap.parse_args(); {'convert': cmd_convert, 'serve': cmd_serve, 'run': cmd_run, 'bench': cmd_bench}[a.cmd](a)


if __name__ == '__main__':
    main()
