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
WANT = {'prefill': 0, 'nll': 1, 'close': 2}


class Worker:
    def __init__(self, host, port, key=None, timeout=600, slot=0):
        self.name = f'{host}:{port}' + (f'#{slot}' if slot else ''); key = (key or os.environ.get('SEEDPLANE_AUTHKEY', 'seedplane-local-default')).encode()
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
        c0, c1, idx = win; tok = np.ascontiguousarray(ids[idx], dtype=np.int32); pos = np.ascontiguousarray(idx, dtype=np.int32)
        core_off = len(idx) - (c1 - c0); nxt = int(ids[c1]) if (want == 'nll' and c1 < len(ids)) else -1
        self.s.sendall(REQ.pack(MAGIC, WANT[want], len(tok), core_off, max(0, score_from - (c0 - core_off)), nxt) + tok.tobytes() + pos.tobytes())

    def recv(self):
        magic, nll, n, argmax, ms = RESP.unpack(self._read(RESP.size)); return nll, n, argmax, ms

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


def run_windows(workers, ids, wins, want='prefill', score_from=0, timeline=None):
    """Dynamic pull queue with a tail guard (same policy as seedplane.cli.distribute_dynamic)."""
    wins = list(wins); est = {}
    for w in workers:                                   # 1-window probe per worker (round trip, network included)
        w.send_window(ids, wins[0], want, score_from); w.recv()
        t = time.perf_counter(); w.send_window(ids, wins[0], want, score_from); w.recv(); est[w.name] = time.perf_counter() - t
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


def plan_pieces(L, rates, S=512, H=256, min_core=16):
    """Size each device's piece so that ALL devices finish together (no device is dropped by policy).

    rates: {worker_name: tok/s measured on ~S+H token windows}. Each window costs (core + halo) / rate; a device gets
    as many full S-core windows as fit in the common finish time T, plus one partial window if >= min_core tokens fit.
    Returns ({name: [(c0, c1), ...]}, T). A device whose smallest piece would still end after T gets nothing FOR THIS
    prompt (it stays loaded and takes the next request) - reported, never hidden.
    """
    def capacity(r, T):
        budget = r * T; n = int(budget // (S + H)); rest = budget - n * (S + H) - H
        return n * S + (int(rest) if rest >= min_core else 0)
    lo, hi = 0.0, (L + H * (L // S + 1)) / max(rates.values())
    for _ in range(60):                          # bisection on the common finish time T
        T = (lo + hi) / 2
        if sum(capacity(r, T) for r in rates.values()) >= L: hi = T
        else: lo = T
    caps = {n: capacity(r, hi) for n, r in rates.items()}; out = {n: [] for n in rates}; c0 = 0
    for n in sorted(rates, key=lambda n: rates[n]):  # slow devices take the first, small pieces; the fastest takes the rest
        left = min(caps[n], L - c0)
        while left > 0 and c0 < L:
            c1 = min(L, c0 + min(S, left)); out[n].append((c0, c1)); left -= c1 - c0; c0 = c1
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
