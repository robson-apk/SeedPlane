"""V13 speed / memory / quality: SeedPlane windows on llama.cpp kernels vs llama.cpp native. See PROTOCOL.md (P1-P8, M1).

python v13_speed.py --exe seedplane-worker.exe --bench llama-bench.exe --gguf model.gguf --bundle <tokenizer dir> --text wiki.txt
                    [--mac 10.0.0.92:54110] [--mac-rate 1250] [--Ls 4096,8192,16384,32768]
Saves after every step to results/speed.json (never overwrites a previous file: results/speed_<n>.json).
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np, psutil
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parents[1]))
from seedplane import engine, llama_backend as lb

S, H, REPS = 512, 256, 3
BUF = re.compile(r'(KV|compute) buffer size\s*=\s*([\d.]+) MiB')


def start(exe, gguf, port, dev, slots=1, threads=1, ctx=1024, log=None):
    f = open(log, 'w') if log else subprocess.DEVNULL
    p = subprocess.Popen([exe, '-m', gguf, '--dev', dev, '--slots', str(slots), '-t', str(threads), '-c', str(ctx), '-b', str(min(ctx, 2048)),
                          '--port', str(port), '--host', '127.0.0.1'], stdout=f, stderr=subprocess.STDOUT)
    return p


def runtime_mib(log):
    txt = Path(log).read_text(errors='ignore'); return sum(float(m.group(2)) for m in BUF.finditer(txt))


def rates_concurrent(workers, ids):
    """Each connection processes one S+H window at the SAME time (contention included) -> tok/s per connection."""
    win = next(w for w in engine.ShardPlan(S, H, 0).windows(4 * S) if w[0] > 0)
    lb.run_pieces(workers, ids, {w.name: [win] for w in workers})                         # warm
    out, _ = lb.run_pieces(workers, ids, {w.name: [win] * 2 for w in workers})
    return {n: 2 * len(win[2]) / (sum(r[3] for r in rs) / 1e3) for n, rs in out.items()}


def timed(fn):
    fn(); ts = []
    for _ in range(REPS): t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def llama_bench(exe, gguf, L, extra):
    out = subprocess.run([exe, '-m', gguf, '-p', str(L), '-n', '0', '-r', str(REPS), '-o', 'json'] + extra, capture_output=True, text=True, timeout=3600)
    try: return float(json.loads(out.stdout)[0]['avg_ts'])
    except Exception: return {'error': (out.stderr or out.stdout)[-300:]}


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument(k) for k in ('--exe', '--bench', '--gguf', '--bundle', '--text')]
    ap.add_argument('--mac', default=''); ap.add_argument('--Ls', default='4096,8192,16384,32768'); ap.add_argument('--parts', default='native,sp,mem,quality')
    a = ap.parse_args(); Ls = [int(x) for x in a.Ls.split(',')]; parts = a.parts.split(',')
    RES = ROOT / 'results'; RES.mkdir(exist_ok=True); fp = RES / 'speed.json'; k = 1
    while fp.exists(): fp = RES / f'speed_{k}.json'; k += 1
    res = {'S': S, 'H': H, 'reps': REPS, 'native': {}, 'sp': {}, 'mem': {}, 'quality': {}, 'notes': []}
    save = lambda: fp.write_text(json.dumps(res, indent=1))
    from transformers import AutoTokenizer
    ids_all = np.array(AutoTokenizer.from_pretrained(a.bundle)(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)

    if 'native' in parts:   # llama.cpp native, full attention, same build
        for L in Ls:
            res['native'][f'gpu|{L}'] = llama_bench(a.bench, a.gguf, L, ['-ngl', '99', '-dev', 'Vulkan0']); save(); print('native gpu', L, res['native'][f'gpu|{L}'], flush=True)
        for L in [x for x in Ls if x <= 8192]:
            res['native'][f'cpu6|{L}'] = llama_bench(a.bench, a.gguf, L, ['-ngl', '0', '-t', '6', '-nopo', '1']); save(); print('native cpu', L, flush=True)
        res['notes'].append('native CPU uses -nopo 1: without it llama.cpp offloads big prompt matmuls to the GPU even at -ngl 0 (run 1 measured 3610 tok/s that way).')
        res['notes'].append('native CPU skipped for L > 8192 (quadratic; > 1 h). Not a result.')

    if 'sp' in parts:
        procs = [start(a.exe, a.gguf, 54000, 'Vulkan0'), start(a.exe, a.gguf, 54001, 'CPU', slots=3, threads=2),
                 start(a.exe, a.gguf, 54002, 'CPU', slots=2, threads=2)]
        try:
            gpu = lb.connect(['127.0.0.1:54000']); cpu3 = lb.connect(['127.0.0.1:54001*3']); cpu2 = lb.connect(['127.0.0.1:54002*2'])
            mac = lb.connect([a.mac]) if a.mac else []
            sets = {'gpu': gpu, 'gpu+cpu4': gpu + cpu2, 'gpu+cpu6': gpu + cpu3}
            if mac: sets['gpu+cpu4+mac'] = gpu + cpu2 + mac; sets['gpu+cpu6+mac'] = gpu + cpu3 + mac
            # CPU-only scaling (P5), L=4096: 1, 2, 3 slots x 2 threads in one process with shared weights
            ids = ids_all[:4096]; wins = list(engine.ShardPlan(S, H, 0).windows(4096))
            lb.calibrate(gpu + cpu3 + cpu2 + mac, ids_all, wins[1])          # pre-bench once; models stay warm
            alone = rates_concurrent(cpu3[:1], ids_all)[cpu3[0].name]
            for n in (1, 2, 3):
                t = timed(lambda: lb.run_windows(cpu3[:n], ids, wins, 'prefill')); res['sp'][f'cpu{2 * n}t|4096'] = {'tok_s': 4096 / t, 'sum_alone_tok_s': n * alone, 'P5_efficiency': 4096 / t / (n * alone)}
            res['sp']['cpu_slot_alone_tok_s'] = alone; save(); print('cpu scaling done', flush=True)
            for L in Ls:
                ids = ids_all[:L]
                for name, ws in sets.items():
                    rates = rates_concurrent(ws, ids_all)
                    pieces, T = lb.plan_pieces(L, rates, S, H); wb = lb.windows_from_pieces(pieces, H)
                    tl = []
                    t = timed(lambda: lb.run_pieces(ws, ids, wb, 'prefill'))
                    out, tt = lb.run_pieces(ws, ids, wb, 'prefill', timeline=tl)
                    busy = max(sum(r[3] for r in rs) for rs in out.values()) / 1e3
                    row = {'tok_s': L / t, 'predicted_tok_s': L / T, 'rates': rates, 'tokens_per_worker': {n: sum(c1 - c0 for c0, c1 in p) for n, p in pieces.items()},
                           'overhead_frac': max(0.0, 1 - busy / tt)}
                    if len(ws) > 1: row['dynamic_tok_s'] = L / timed(lambda: lb.run_windows(ws, ids, list(engine.ShardPlan(S, H, 0).windows(L)), 'prefill'))
                    if name.endswith('mac') and L == 16384: row['timeline'] = tl
                    res['sp'][f'{name}|{L}'] = row; save(); print('sp', name, L, round(row['tok_s']), 'pred', round(row['predicted_tok_s']), row['tokens_per_worker'], flush=True)
                    # span mode (adendo 3): one contiguous span per device, halo KV reused instead of recomputed
                    pieces, T = lb.plan_pieces(L, rates, S, H, span=True); wb = lb.windows_from_pieces(pieces, H); tl = []
                    t = timed(lambda: lb.run_pieces(ws, ids, wb, 'span'))
                    out, tt = lb.run_pieces(ws, ids, wb, 'span', timeline=tl); busy = max(sum(r[3] for r in rs) for rs in out.values()) / 1e3
                    row = {'tok_s': L / t, 'predicted_tok_s': L / T, 'tokens_per_worker': {n: sum(c1 - c0 for c0, c1 in p) for n, p in pieces.items()},
                           'overhead_frac': max(0.0, 1 - busy / tt)}
                    if name.endswith('mac') and L == 16384: row['timeline'] = tl
                    res['sp'][f'{name}|span|{L}'] = row; save(); print('span', name, L, round(row['tok_s']), 'pred', round(row['predicted_tok_s']), row['tokens_per_worker'], flush=True)
            for w in gpu + cpu3 + cpu2 + mac: w.close()
        finally:
            for p in procs: p.kill()

    if 'mem' in parts:   # M1: runtime buffers (KV + compute) that llama.cpp itself reports, window ctx vs full ctx
        port = 54100
        for dev, Lm in [('Vulkan0', L) for L in Ls if L >= 8192] + [('CPU', 8192)]:
            for tag, ctx in (('seedplane', 1024), ('full', Lm + 64)):
                log = str(ROOT / 'results' / f'mem_{dev}_{tag}_{Lm}.log'); port += 1; p = start(a.exe, a.gguf, port, dev, ctx=ctx, log=log)
                try:
                    w = lb.connect([f'127.0.0.1:{port}'])[0]; ids = ids_all[:Lm]
                    wins = [(0, Lm, np.arange(Lm))] if tag == 'full' else list(engine.ShardPlan(S, H, 0).windows(Lm))
                    lb.run_windows([w], ids, wins, 'prefill'); peak = psutil.Process(p.pid).memory_info(); w.close()
                    res['mem'][f'{dev}|{tag}|{Lm}'] = {'runtime_mib': runtime_mib(log), 'proc_peak_mib': getattr(peak, 'peak_wset', peak.rss) / 2**20}
                finally: p.kill()
                save(); print('mem', dev, tag, Lm, res['mem'][f'{dev}|{tag}|{Lm}'], flush=True)

    if 'quality' in parts:   # P3: same engine, NLL windows / NLL full, L=16384, 3 WikiText chunks, GPU
        Lq = 16384; p1 = start(a.exe, a.gguf, 54020, 'Vulkan0', ctx=Lq + 64); p2 = start(a.exe, a.gguf, 54021, 'Vulkan0')
        try:
            wf, ww = lb.connect(['127.0.0.1:54020', '127.0.0.1:54021'])
            for c in range(3):
                ids = ids_all[c * Lq:(c + 1) * Lq]
                full = lb.run_windows([wf], ids, [(0, Lq, np.arange(Lq))], 'nll')[0][0]
                sh = lb.run_windows([ww], ids, list(engine.ShardPlan(S, H, 0).windows(Lq)), 'nll')[0]
                sp_ = lb.run_windows([ww], ids, [(0, Lq, np.arange(Lq))], 'span_nll')[0][0]
                res['quality'][f'chunk{c}'] = {'full_nll_tok': full[0] / full[1], 'sp_nll_tok': sum(r[0] for r in sh) / sum(r[1] for r in sh),
                                               'span_nll_tok': sp_[0] / sp_[1], 'n_scored': [full[1], sum(r[1] for r in sh), sp_[1]]}
                save(); print('quality', c, res['quality'][f'chunk{c}'], flush=True)
            wf.close(); ww.close()
        finally: p1.kill(); p2.kill()
    print('saved', fp)


if __name__ == '__main__':
    main()
