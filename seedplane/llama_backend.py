"""Client for native `seedplane-worker` processes (llama.cpp kernels) + the dynamic, tail-guarded window scheduler.

    from seedplane import llama_backend as lb
    workers = lb.connect(['127.0.0.1:54000', 'gpu-box:54000', 'mac.local:54001'])
    nll, secs, per_worker = lb.run_windows(workers, ids, plan.windows(len(ids)), want='nll')
"""
import os, select, socket, struct, subprocess, time
import numpy as np

MAGIC = 0x31575053                      # "SPW1"
REQ = struct.Struct('<IIIIIi')           # magic, want, n_tok, core_off, score_from, next_tok
RESP = struct.Struct('<IdIif')           # magic, nll_sum, n_scored, argmax_last, compute_ms
WANT = {'prefill': 0, 'nll': 1, 'close': 2, 'span': 3, 'span_nll': 4}   # span*: KV halo reuse (worker --span-chunk/--span-keep)
DEFAULT_AUTH = 'seedplane-loopback-only'


class Worker:
    def __init__(self, host, port, key=None, timeout=600, slot=0):
        self.name = f'{host}:{port}' + (f'#{slot}' if slot else ''); key = (key or os.environ.get('SEEDPLANE_AUTHKEY', DEFAULT_AUTH)).encode()
        deadline = time.time() + timeout
        while True:
            try: self.s = socket.create_connection((host, port), timeout=timeout); break
            except OSError:
                if time.time() > deadline: raise
                time.sleep(0.5)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.sendall(struct.pack('<I', len(key)) + key)
        if struct.unpack('<I', self._read(4))[0] != 1: raise PermissionError(f'{self.name}: bad SEEDPLANE_AUTHKEY')

    def _read(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.s.recv(n - len(buf))
            if not chunk: raise ConnectionError(self.name)
            buf += chunk
        return buf

    def send_window(self, ids, win, want, score_from=0):
        c0, c1, idx = win; tok = np.ascontiguousarray(ids[idx], dtype=np.int32); pos = np.array(idx, dtype=np.int32)
        gap = np.flatnonzero(np.diff(pos) != 1)
        if len(gap):   # sinks + halo: llama.cpp needs consecutive positions in a batch -> place the sinks right before
            g = gap[0] + 1; pos[:g] = pos[g] - g + np.arange(g)   # the halo (StreamingLLM); halo/core distances stay original
        core_off = len(idx) - (c1 - c0); nxt = int(ids[c1]) if (want in ('nll', 'span_nll') and c1 < len(ids)) else -1
        self.s.sendall(REQ.pack(MAGIC, WANT[want], len(tok), core_off, max(0, score_from - (c0 - core_off)), nxt) + tok.tobytes() + pos.tobytes())

    def recv(self):
        magic, nll, n, argmax, ms = RESP.unpack(self._read(RESP.size))
        if magic != MAGIC: raise ConnectionError(f'{self.name}: invalid response magic 0x{magic:08x}')
        return nll, n, argmax, ms

    def close(self):
        try: self.s.sendall(REQ.pack(MAGIC, 2, 0, 0, 0, -1))
        except OSError: pass
        self.s.close()


def connect(addrs, key=None):
    """addrs: ['host:port', 'host:port*4', ...]; '*N' opens N connections = N parallel slots of one worker process."""
    out = []
    for a in addrs:
        a, n = (a.split('*') + ['1'])[:2]; host, port = a.rsplit(':', 1)
        out += [Worker(host, int(port), key, slot=k) for k in range(int(n))]
    return out


def spawn(exe, gguf, port, dev='CPU', slots=1, threads=1, ctx=4096, extra_env=None):
    """Start a local worker process (returns Popen); pair with connect([f'127.0.0.1:{port}*{slots}'])."""
    env = dict(os.environ, **(extra_env or {}))
    return subprocess.Popen([exe, '-m', gguf, '--port', str(port), '--dev', dev, '--slots', str(slots), '-t', str(threads), '-c', str(ctx), '--host', '127.0.0.1'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def calibrate(workers, ids, win, want='prefill'):
    """Pre-bench ONCE (at startup, models stay loaded): round-trip seconds of one window per worker, network included.
    Stored on each worker (w.est) and reused by every later request instead of re-probing inside the request."""
    for w in workers:
        w.send_window(ids, win, want); w.recv()
        t = time.perf_counter(); w.send_window(ids, win, want); w.recv(); w.est = time.perf_counter() - t
    return {w.name: w.est for w in workers}


def measure_rates(workers, ids, win, want='prefill', repeats=2):
    """Measure concurrent end-to-end throughput for planning.

    Unlike the worker-reported ``compute_ms``, this includes serialization,
    network round trips and coordinator delay. All workers are exercised at
    once so shared CPU/driver contention is represented as well.
    """
    if not workers: raise ValueError('at least one worker is required')
    if repeats < 1: raise ValueError('repeats must be positive')
    n_tokens = len(win[2])
    if n_tokens < 1: raise ValueError('calibration window must contain tokens')
    work = {w.name: [win] for w in workers}
    run_pieces(workers, ids, work, want)  # warm kernels and caches
    timeline = []
    run_pieces(workers, ids, {w.name: [win] * repeats for w in workers}, want, timeline)
    elapsed = {w.name: 0.0 for w in workers}
    for row in timeline: elapsed[row['worker']] += row['end'] - row['start']
    return {name: repeats * n_tokens / seconds for name, seconds in elapsed.items() if seconds > 0}


def run_windows(workers, ids, wins, want='prefill', score_from=0, timeline=None):
    """Dynamic pull queue with a tail guard (same policy as seedplane.cli.distribute_dynamic).
    Uses w.est from calibrate() when present; only uncalibrated workers are probed here (and that probe is timed)."""
    wins = list(wins)
    todo = [w for w in workers if getattr(w, 'est', None) is None]
    if todo: calibrate(todo, ids, wins[0], want)
    est = {w.name: w.est for w in workers}
    queue = list(range(len(wins))); out = [None] * len(wins); busy, sent = {}, {}; count = {w.name: 0 for w in workers}; t0 = time.perf_counter()

    def worth_it(w):
        others = [o for o in workers if o is not w]
        return not others or est[w.name] <= len(queue) / sum(1.0 / est[o.name] for o in others) + 1e-9

    def feed(w):
        if queue and (worth_it(w) or not any(o.name in busy for o in workers if o is not w)):
            k = queue.pop(0); w.send_window(ids, wins[k], want, score_from); busy[w.name] = k; sent[w.name] = time.perf_counter()

    for w in sorted(workers, key=lambda w: est[w.name]): feed(w)
    by_sock = {w.s: w for w in workers}
    while busy:
        ready, _, _ = select.select([w.s for w in workers if w.name in busy], [], [])
        for s in ready:
            w = by_sock[s]; r = w.recv(); k = busy.pop(w.name); out[k] = r; count[w.name] += 1
            est[w.name] = 0.7 * est[w.name] + 0.3 * (time.perf_counter() - sent[w.name])
            if timeline is not None: timeline.append({'worker': w.name, 'window': k, 'start': sent[w.name] - t0, 'end': time.perf_counter() - t0})
            feed(w)
        for w in workers:
            if w.name not in busy and queue: feed(w)
    return out, time.perf_counter() - t0, count


def plan_pieces(L, rates, S=512, H=256, min_core=16, span=False, min_gain=0.04):
    """Size each device's piece so that ALL devices finish together (no device is dropped by policy).

    rates: {worker_name: tok/s measured on ~S+H token windows}. Each window costs (core + halo) / rate; a device gets
    as many full S-core windows as fit in the common finish time T, plus one partial window if >= min_core tokens fit.
    Returns ({name: [(c0, c1), ...]}, T). A device whose smallest piece would still end after T gets nothing FOR THIS
    prompt. If the whole pool's predicted gain over the fastest worker is below ``min_gain`` (4% by default), the
    fastest worker runs alone. The margin covers calibration/model error observed on short spans and avoids turning a
    mathematically tiny gain into a real network regression. Set it explicitly when a different risk margin is known.
    """
    if L < 0 or S <= 0 or H < 0 or min_core <= 0 or not 0 <= min_gain < 1:
        raise ValueError('invalid length, shard, halo, minimum core, or gain threshold')
    if not rates or any(r <= 0 for r in rates.values()): raise ValueError('worker rates must be positive')
    def capacity(r, T):
        if span:                                 # one contiguous span per device: the halo is paid once, not per window
            c = int(r * T - H); return c if c >= min_core else 0
        budget = r * T; n = int(budget // (S + H)); rest = budget - n * (S + H) - H
        return n * S + (int(rest) if rest >= min_core else 0)
    lo, hi = 0.0, (L + H * (L // S + 1)) / max(rates.values())
    for _ in range(60):                          # bisection on the common finish time T
        T = (lo + hi) / 2
        if sum(capacity(r, T) for r in rates.values()) >= L: hi = T
        else: lo = T
    fastest = max(rates, key=rates.get)
    solo_T = (L + H if span else L + H * max(0, (L + S - 1) // S - 1)) / rates[fastest] if L else 0.0
    if solo_T and (solo_T - hi) / solo_T < min_gain:
        return {n: ([(0, L)] if n == fastest and L else []) for n in rates}, solo_T
    caps = {n: capacity(r, hi) for n, r in rates.items()}; out = {n: [] for n in rates}; c0 = 0
    for n in sorted(rates, key=lambda n: rates[n]):  # slow devices take the first, small pieces; the fastest takes the rest
        left = min(caps[n], L - c0)
        while left > 0 and c0 < L:
            c1 = min(L, c0 + (left if span else min(S, left))); out[n].append((c0, c1)); left -= c1 - c0; c0 = c1
    return out, hi


def windows_from_pieces(pieces, H=256, sinks=0):
    """Turn core ranges into SeedPlane windows (sinks + halo + core, original positions)."""
    res = {}
    for n, spans in pieces.items():
        res[n] = []
        for c0, c1 in spans:
            h0 = max(0, c0 - H); idx = np.arange(h0, c1)
            if sinks and h0 > 0: idx = np.concatenate([np.arange(min(sinks, h0)), idx])
            res[n].append((c0, c1, idx))
    return res


def run_pieces(workers, ids, wins_by_worker, want='prefill', timeline=None):
    """Every worker streams its own window list; all run concurrently. Returns ({name: results}, seconds)."""
    by = {w.name: w for w in workers}; todo = {n: list(ws) for n, ws in wins_by_worker.items() if ws}
    out = {n: [] for n in todo}; sent = {}; t0 = time.perf_counter()
    for n in todo: by[n].send_window(ids, todo[n][0], want); sent[n] = time.perf_counter()
    active = {by[n].s: n for n in todo}
    while active:
        ready, _, _ = select.select(list(active), [], [])
        for s in ready:
            n = active[s]; out[n].append(by[n].recv())
            if timeline is not None: timeline.append({'worker': n, 'start': sent[n] - t0, 'end': time.perf_counter() - t0})
            todo[n].pop(0)
            if todo[n]: by[n].send_window(ids, todo[n][0], want); sent[n] = time.perf_counter()
            else: del active[s]
    return out, time.perf_counter() - t0
