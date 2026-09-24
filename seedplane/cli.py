"""seedplane — run existing Hugging Face models with SeedPlane shard windows, on any mix of devices.

  seedplane convert Qwen/Qwen2.5-0.5B-Instruct ./qwen05.sp --shard 512 --halo 256
  seedplane convert ./qwen05.sp ./qwen05-native.sp --native      # engine-ready bundle for native/vulkan_decode
  seedplane chat     ./qwen05-native.sp --native                   # chat on the native runtime (Vulkan GPU)
  seedplane generate ./qwen05-native.sp --native --prompt "Once upon a time" -n 64
  seedplane serve   --bundle ./qwen05.sp --device cpu --port 52000        # turn this machine/core/GPU into a worker
  seedplane run     --bundle ./qwen05.sp --prompt-file long.txt --workers local:xpu:1,local:cpu:4,OTHER-PC:52000
  seedplane bench   --bundle ./qwen05.sp --text-file corpus.txt --length 4096

A worker is one process on one device (a CPU core, a GPU, or another computer over the LAN).
`run` splits the prompt into shard windows and hands them out from a queue: each worker asks for the next window when it
finishes, and a tail guard keeps slow workers from holding the last windows. Windows are independent, so workers never wait
for each other.

SECURITY: ``worker`` uses the bounded authenticated protocol v1 (JSON + HMAC). ``serve`` is the legacy research worker
and still uses Python pickle: keep it on loopback/trusted LAN only. Never expose either worker directly to the internet.
"""
import argparse, json, os, platform, shutil, socket, subprocess, sys, time, uuid
from multiprocessing.connection import Listener, Client
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import engine                                   # installed package: `seedplane ...`
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import engine   # plain script: `python seedplane/cli.py ...`

DEFAULT_AUTH = 'seedplane-loopback-only'


def authkey():
    """Authentication key shared by coordinators and workers.

    The fallback is intentionally restricted to loopback workers by cmd_serve;
    listening on another interface requires an explicit secret.
    """
    return os.environ.get('SEEDPLANE_AUTHKEY', DEFAULT_AUTH).encode()


def is_loopback(host):
    return host in {'127.0.0.1', '::1', 'localhost'}


def plan_of(bundle):
    cfg = json.loads((Path(bundle) / 'seedplane.json').read_text())['plan']; return engine.ShardPlan(**cfg)


def _config_dir():
    return Path(os.environ.get('SEEDPLANE_CONFIG_DIR', Path.home() / '.config' / 'seedplane'))


def _identity(name):
    try: from .cluster import stable_id
    except ImportError: from cluster import stable_id
    env = os.environ.get(f'SEEDPLANE_{name.upper()}_ID')
    return str(uuid.UUID(env)) if env else stable_id(_config_dir() / f'{name}_id', name)


def _endpoint(value):
    host, sep, port = value.rpartition(':')
    if not sep or not host:
        raise argparse.ArgumentTypeError('address must be HOST:PORT')
    try: port = int(port)
    except ValueError as exc: raise argparse.ArgumentTypeError('port must be an integer') from exc
    if not 1 <= port <= 65535: raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return host, port


# ------------------------------------------------------------------ safe protocol-v1 worker / devices / doctor
def cmd_worker(a):
    try: from .cluster import WorkerServer, cluster_key
    except ImportError: from cluster import WorkerServer, cluster_key
    if not is_loopback(a.host) and 'SEEDPLANE_CLUSTER_KEY' not in os.environ:
        raise SystemExit('refusing non-loopback worker without SEEDPLANE_CLUSTER_KEY (minimum 16 characters)')
    server = WorkerServer(a.host, a.port, cluster_key(), worker_id=_identity('worker'), cluster_id=_identity('cluster'))
    try: server.serve_forever()
    except KeyboardInterrupt: pass


def cmd_devices(a):
    try: from .cluster import cluster_key, probe
    except ImportError: from cluster import cluster_key, probe
    key, cluster_id, coordinator_id = cluster_key(), _identity('cluster'), _identity('coordinator')
    addresses = a.address or [('127.0.0.1', 52100)]
    rows = []
    for host, port in addresses:
        t0 = time.perf_counter()
        try:
            info = probe(host, port, key, cluster_id, coordinator_id, timeout=a.timeout)
            cap = info['capabilities']
            rows.append({'address': f'{host}:{port}', 'status': 'ready', 'latency_ms': round((time.perf_counter() - t0) * 1000, 2),
                         'system': cap.get('system', '?'), 'machine': cap.get('machine', '?'),
                         'backends': cap.get('backends', []), 'hostname': cap.get('hostname', '?')})
        except Exception as exc:
            rows.append({'address': f'{host}:{port}', 'status': 'error', 'error': str(exc)[:200]})
    if a.json:
        print(json.dumps({'protocol': 1, 'devices': rows}, indent=2)); return
    print(f"{'ADDRESS':<24} {'STATUS':<8} {'HOST':<18} {'SYSTEM/ARCH':<22} BACKENDS")
    for row in rows:
        detail = row.get('error', ','.join(row.get('backends', [])))
        print(f"{row['address']:<24} {row['status']:<8} {row.get('hostname', '-'):<18} "
              f"{(row.get('system', '-') + '/' + row.get('machine', '-')):<22} {detail}")
    if a.action == 'test' and any(row['status'] != 'ready' for row in rows):
        raise SystemExit(1)


def _doctor_checks():
    checks = []
    def add(name, status, detail, fix=''):
        checks.append({'name': name, 'status': status, 'detail': detail, 'fix': fix})
    add('platform', 'ok', f'{platform.system()} {platform.release()} {platform.machine()}')
    add('python', 'ok' if sys.version_info >= (3, 9) else 'error', platform.python_version(), 'Install Python 3.9 or newer.')
    usage = shutil.disk_usage(Path.home())
    free_gib = usage.free / 2**30
    add('disk', 'ok' if free_gib >= 10 else 'warn', f'{free_gib:.1f} GiB free', 'Free at least 10 GiB for models and builds.')
    key = os.environ.get('SEEDPLANE_CLUSTER_KEY', '')
    add('cluster-key', 'ok' if len(key) >= 16 else 'warn', 'configured' if key else 'not configured',
        'Set SEEDPLANE_CLUSTER_KEY to at least 16 random characters before LAN use.')
    try:
        import torch
        backends = ['cpu']
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): backends.append('mps')
        if torch.cuda.is_available(): backends.append('cuda')
        if hasattr(torch, 'xpu') and torch.xpu.is_available(): backends.append('xpu')
        add('torch', 'ok', f'{torch.__version__}; backends={",".join(backends)}')
    except Exception as exc:
        add('torch', 'error', f'{type(exc).__name__}: {exc}', 'Install SeedPlane with its declared dependencies.')
    vulkaninfo, glslc = shutil.which('vulkaninfo'), shutil.which('glslc')
    add('vulkan-tools', 'ok' if vulkaninfo and glslc else 'warn', f'vulkaninfo={vulkaninfo or "missing"}; glslc={glslc or "missing"}',
        'Install a Vulkan SDK only when using/building the native Vulkan runtime.')
    add('legacy-serve', 'warn', 'uses multiprocessing.connection/pickle', 'Prefer `seedplane worker`; keep `serve` on a trusted LAN only.')
    return checks


def cmd_doctor(a):
    checks = _doctor_checks()
    if a.json:
        print(json.dumps({'checks': checks}, indent=2)); return
    for check in checks:
        icon = {'ok': 'OK', 'warn': 'WARN', 'error': 'ERROR'}[check['status']]
        print(f"[{icon:5}] {check['name']}: {check['detail']}")
        if check['fix'] and check['status'] != 'ok': print(f"        → {check['fix']}")
    if any(c['status'] == 'error' for c in checks): raise SystemExit(1)


# ------------------------------------------------------------------ convert
def cmd_convert(a):
    if a.native:
        try: from . import native_bundle
        except ImportError: import native_bundle
        m = native_bundle.convert(a.model, a.out, a.shard, a.halo, a.sinks)
        print(f"native bundle written to {a.out}  ({m['format']}, plan {m['plan']}, weights sha256 {m['weights']['sha256'][:16]}...)")
        return
    plan = engine.ShardPlan(512 if a.shard is None else a.shard, 256 if a.halo is None else a.halo, a.sinks or 0)
    out = engine.save_bundle(a.out, a.model, plan)
    print(f'bundle written to {out}  (weights unchanged + seedplane.json)')


# ------------------------------------------------------------------ native chat / generate
def _native_engine(a):
    try: from . import native
    except ImportError: import native
    sampling = {'temperature': a.temperature, 'top_k': a.top_k, 'top_p': a.top_p, 'seed': a.seed}
    return native, native.NativeEngine(a.bundle, engine=a.engine, mode=a.mode, full=a.full, ctx=a.ctx, max_new_tokens=a.max_new_tokens,
                                       system=getattr(a, 'system', None), **sampling)


def cmd_chat(a):
    if not a.native:
        raise SystemExit('seedplane chat runs the native runtime: add --native (PyTorch reference chat: seedplane-chat <hf_dir>)')
    native, eng = _native_engine(a)
    with eng: native.chat_repl(eng)


def cmd_generate(a):
    if not a.native:
        raise SystemExit('seedplane generate runs the native runtime: add --native')
    text = Path(a.prompt_file).read_text(encoding='utf-8') if a.prompt_file else a.prompt
    native, eng = _native_engine(a)
    with eng:
        for piece in eng.generate(text): print(piece, end='', flush=True)
        s = eng.last
        print(f"\n[{s['generated']} tokens, {s['decode_tok_s']:.1f} tok/s, prompt {s['prompt_tokens']} tokens in {s['prefill_s']:.2f}s, {s['reason']}]",
              file=sys.stderr)


# ------------------------------------------------------------------ serve (a worker)
def cmd_serve(a):
    if not is_loopback(a.host) and 'SEEDPLANE_AUTHKEY' not in os.environ:
        raise SystemExit('refusing non-loopback worker without SEEDPLANE_AUTHKEY; set a long random secret or use --host 127.0.0.1')
    torch.set_num_threads(a.threads); model, _ = engine.load_model(a.bundle, a.device)
    with Listener((a.host, a.port), authkey=authkey()) as lst:
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
    if not spec.strip(): raise ValueError('at least one worker is required')
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
            try: clients.append((name, Client((h, p), authkey=authkey()))); break
            except OSError: time.sleep(0.5)
        else: raise RuntimeError(f'worker {name} unreachable')
    return procs, clients


def distribute(clients, ids, wins, want):
    """Calibrate each worker on one window, then assign windows proportionally to speed; returns results in window order."""
    if not clients: raise ValueError('at least one worker is required')
    if not wins: return [], 0.0, {n: 0 for n, _ in clients}
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


def distribute_dynamic(clients, ids, wins, want, prior=None, timeline=None):
    """Pull-based work queue with a tail guard.
    Each worker gets one window at a time and asks for more when done (fast devices naturally take more).
    Per-window time of each worker is tracked (EMA, seeded by `prior` or a 1-window probe). A worker only receives a
    window if it would finish it before the rest of the pool would finish the whole remaining queue without it
    (so a slow core or a far-away machine never becomes the straggler that holds the last window)."""
    if not clients: raise ValueError('at least one worker is required')
    if not wins: return [], 0.0, {n: 0 for n, _ in clients}
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
            k = queue.pop(0); c.send(('windows', ids, [wins[k]], want)); busy[name] = k; sent_at[name] = time.perf_counter()
            return True
        return False

    for name, c in sorted(clients, key=lambda nc: est[nc[0]]):
        feed(name, c)
    while busy:
        for conn in wait([c for n, c in clients if n in busy]):
            name = by_conn[conn]; res, _ = conn.recv(); k = busy.pop(name); out[k] = res[0]; count[name] += 1
            if timeline is not None: timeline.append({'worker': name, 'window': k, 'start': sent_at[name] - t0, 'end': time.perf_counter() - t0})
            est[name] = 0.7 * est[name] + 0.3 * (time.perf_counter() - sent_at[name])   # round-trip time, network included
            feed(name, conn)
        for name, c in clients:
            if name not in busy and queue: feed(name, c)
    return out, time.perf_counter() - t0, count


def cmd_run(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.bundle)
    ids = np.array(tok(Path(a.prompt_file).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)[: a.max_tokens]
    if len(ids) < 2: raise SystemExit('prompt must contain at least two tokens for perplexity scoring')
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
    if len(ids) <= a.length: raise SystemExit(f'text must contain more than --length ({a.length}) tokens')
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
    c.add_argument('--shard', type=int, help='default 512 (native: keep a v1 source plan, else 512)')
    c.add_argument('--halo', type=int, help='default 256'); c.add_argument('--sinks', type=int, help='default 0 (native from scratch: 4)')
    c.add_argument('--native', action='store_true', help='write a seedplane-bundle/2 for the native engine (source: HF dir, v1 bundle or hub id)')
    s = sub.add_parser('serve', help='run a worker on this device'); s.add_argument('--bundle', required=True); s.add_argument('--device', default='cpu')
    s.add_argument('--port', type=int, default=52000); s.add_argument('--host', default='127.0.0.1'); s.add_argument('--threads', type=int, default=1)
    w = sub.add_parser('worker', help='run a safe protocol-v1 control-plane worker')
    w.add_argument('--host', default='127.0.0.1'); w.add_argument('--port', type=int, default=52100)
    d = sub.add_parser('devices', help='list or test protocol-v1 workers')
    d.add_argument('action', nargs='?', choices=['list', 'test'], default='list')
    d.add_argument('--address', action='append', type=_endpoint, metavar='HOST:PORT')
    d.add_argument('--timeout', type=float, default=5.0); d.add_argument('--json', action='store_true')
    doc = sub.add_parser('doctor', help='check this machine and give actionable fixes'); doc.add_argument('--json', action='store_true')
    r = sub.add_parser('run', help='process a long prompt across workers'); r.add_argument('--bundle', required=True); r.add_argument('--prompt-file', required=True)
    r.add_argument('--workers', default='local:cpu:4'); r.add_argument('--scheduler', choices=['dynamic', 'static'], default='dynamic'); r.add_argument('--max-tokens', type=int, default=8192)
    b = sub.add_parser('bench', help='quality + speed vs the original full attention'); b.add_argument('--bundle', required=True); b.add_argument('--text-file', required=True)
    b.add_argument('--device', default='cpu'); b.add_argument('--length', type=int, default=4096); b.add_argument('--samples', type=int, default=3)
    for name, helptext in (('chat', 'interactive chat on the native runtime'), ('generate', 'continue a text prompt on the native runtime')):
        n = sub.add_parser(name, help=helptext); n.add_argument('bundle', help='seedplane-bundle/2 directory')
        n.add_argument('--native', action='store_true', help='use the native Vulkan runtime (required)')
        n.add_argument('--engine', help='path to qwen_vk (default: $SEEDPLANE_NATIVE_ENGINE or the in-repo build)')
        n.add_argument('--mode', help='boundary mode (default shadow-batch)'); n.add_argument('--full', action='store_true', help='original full attention')
        n.add_argument('--ctx', type=int, help='maximum positions'); n.add_argument('-n', '--max-new-tokens', type=int, default=256)
        n.add_argument('--temperature', type=float, default=0.7); n.add_argument('--top-k', type=int, default=40)
        n.add_argument('--top-p', type=float, default=0.9); n.add_argument('--seed', type=int, default=1)
        if name == 'chat': n.add_argument('--system', default='You are a helpful assistant.')
        else: n.add_argument('--prompt', default=''); n.add_argument('--prompt-file')
    a = ap.parse_args()
    {'convert': cmd_convert, 'serve': cmd_serve, 'worker': cmd_worker, 'devices': cmd_devices, 'doctor': cmd_doctor,
     'run': cmd_run, 'bench': cmd_bench, 'chat': cmd_chat, 'generate': cmd_generate}[a.cmd](a)


if __name__ == '__main__':
    main()
