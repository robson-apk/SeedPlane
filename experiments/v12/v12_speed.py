"""V12 speed: prompt processing (L=4096) — our engine (full vs SeedPlane windows, any device mix) vs llama.cpp. See PROTOCOL.md.

python v12_speed.py --bundle <seedplane bundle> --gguf <fp16 gguf> --llama <dir with cpu/ sycl/ vulkan/> --text <txt> --plan S,H,K [--mac HOST:PORT,HOST:PORT]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'
sys.path.insert(0, str(ROOT.parents[1] / 'seedplane')); import engine, cli

L, REPS = 4096, 3


def median_time(fn, reps=REPS):
    fn(); ts = []
    for _ in range(reps):
        if torch.xpu.is_available(): torch.xpu.synchronize()
        t0 = time.perf_counter(); fn()
        if torch.xpu.is_available(): torch.xpu.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def llama_bench(exe, gguf, extra):
    out = subprocess.run([str(exe), '-m', str(gguf), '-p', str(L), '-n', '0', '-r', str(REPS), '-o', 'json'] + extra, capture_output=True, text=True, timeout=1800)
    try:
        d = json.loads(out.stdout); return float(d[0]['avg_ts'])
    except Exception:
        return {'error': (out.stderr or out.stdout)[-400:]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle'); ap.add_argument('--gguf'); ap.add_argument('--llama'); ap.add_argument('--text'); ap.add_argument('--plan'); ap.add_argument('--mac', default='')
    a = ap.parse_args(); S, H, K = map(int, a.plan.split(','))
    from transformers import AutoTokenizer
    ids = np.array(AutoTokenizer.from_pretrained(a.bundle)(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)[:L]
    plan = engine.ShardPlan(S, H, K); wins = list(plan.windows(L)); res = {'L': L, 'plan': [S, H, K], 'ours_full': {}, 'ours_seedplane': {}, 'llama_cpp': {}}
    # --- our engine, original full attention (1 process, N threads / GPU)
    for dev, threads in [('cpu', 1), ('cpu', 2), ('cpu', 4), ('cpu', 6), ('xpu', 1)]:
        torch.set_num_threads(threads); model, _ = engine.load_model(a.bundle, dev)
        res['ours_full'][f'{dev}{threads if dev == "cpu" else ""}'] = L / median_time(lambda: engine.prefill_full(model, ids, dev)); del model
        print('full', dev, threads, flush=True)
    torch.set_num_threads(1)
    # --- our engine, SeedPlane windows distributed over workers
    configs = {'cpu1': 'local:cpu:1', 'cpu2': 'local:cpu:2', 'cpu4': 'local:cpu:4', 'cpu6': 'local:cpu:6', 'gpu': 'local:xpu:1', 'gpu+cpu4': 'local:xpu:1,local:cpu:4'}
    if a.mac: configs['cpu4+mac2'] = 'local:cpu:4,' + a.mac
    for name, spec in configs.items():
        procs, clients = cli.start_workers(spec, a.bundle)
        try:
            ts = [cli.distribute(clients, ids, wins, 'prefill')[1] for _ in range(REPS + 1)][1:]
            res['ours_seedplane'][name] = L / float(np.median(ts))
        finally:
            for _, c in clients:
                try: c.send(('close',))
                except Exception: pass
            for p in procs: p.terminate()
        print('seedplane', name, flush=True)
    # --- llama.cpp native (same model, F16 GGUF)
    lp = Path(a.llama)
    for t in (1, 2, 4, 6): res['llama_cpp'][f'cpu{t}'] = llama_bench(lp / 'cpu' / 'llama-bench.exe', a.gguf, ['-t', str(t), '-ngl', '0'])
    res['llama_cpp']['gpu_sycl'] = llama_bench(lp / 'sycl' / 'llama-bench.exe', a.gguf, ['-ngl', '99'])
    res['llama_cpp']['gpu_vulkan'] = llama_bench(lp / 'vulkan' / 'llama-bench.exe', a.gguf, ['-ngl', '99'])
    RES.mkdir(exist_ok=True); (RES / 'speed.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
