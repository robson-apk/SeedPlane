"""V12 speed: prompt processing (L=4096) — our engine (full vs SeedPlane windows, any device mix) vs llama.cpp. See PROTOCOL.md.

python v12_speed.py --bundle <seedplane bundle> --gguf <fp16 gguf> --llama <dir with cpu/ sycl/ vulkan/> --text <txt> --plan S,H,K [--mac HOST:PORT,HOST:PORT]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, psutil, torch
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


def llama_bench(exe, gguf, extra, oneapi=False):
    cmd = [str(exe), '-m', str(gguf), '-p', str(L), '-n', '0', '-r', str(REPS), '-o', 'json'] + extra
    if oneapi:   # SYCL builds need the oneAPI runtime DLLs on PATH
        cmd = ['cmd', '/c', 'call', r'E:\oneAPI\setvars.bat', '>nul', '2>&1', '&&'] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    try:
        d = json.loads(out.stdout); return float(d[0]['avg_ts'])
    except Exception:
        return {'error': (out.stderr or out.stdout)[-400:]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle'); ap.add_argument('--gguf'); ap.add_argument('--llama'); ap.add_argument('--text'); ap.add_argument('--plan'); ap.add_argument('--mac', default='')
    ap.add_argument('--part', default='full,seedplane,llama')  # phases run in separate processes (PROTOCOL adendo 2)
    ap.add_argument('--synapse-llama', help='optional path to local tuned llama.cpp builds')
    a = ap.parse_args(); parts = a.part.split(','); S, H, K = map(int, a.plan.split(','))
    from transformers import AutoTokenizer
    ids = np.array(AutoTokenizer.from_pretrained(a.bundle)(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)[:L]
    plan = engine.ShardPlan(S, H, K); wins = list(plan.windows(L)); RES.mkdir(exist_ok=True); fp = RES / 'speed.json'
    res = json.loads(fp.read_text()) if fp.exists() else {'L': L, 'plan': [S, H, K], 'ours_full': {}, 'ours_seedplane': {}, 'llama_cpp': {}}
    save = lambda: fp.write_text(json.dumps(res, indent=1))
    # --- our engine, original full attention (1 process, N threads / GPU)
    for dev, threads in ([('cpu', 1), ('cpu', 2), ('cpu', 4), ('cpu', 6), ('xpu', 1)] if 'full' in parts else []):
        torch.set_num_threads(threads); model, _ = engine.load_model(a.bundle, dev)
        res['ours_full'][f'{dev}{threads if dev == "cpu" else ""}'] = L / median_time(lambda: engine.prefill_full(model, ids, dev)); del model
        print('full', dev, threads, flush=True); save()
    torch.set_num_threads(1)
    # --- our engine, SeedPlane windows distributed over workers
    configs = {'cpu1': 'local:cpu:1', 'cpu2': 'local:cpu:2', 'cpu4': 'local:cpu:4', 'cpu6': 'local:cpu:6', 'gpu': 'local:xpu:1', 'gpu+cpu4': 'local:xpu:1,local:cpu:4'}
    if a.mac: configs['cpu4+mac2'] = 'local:cpu:4,' + a.mac; configs['gpu+cpu4+mac2'] = 'local:xpu:1,local:cpu:4,' + a.mac
    for name, spec in (configs.items() if 'seedplane' in parts else []):
        n_cpu = sum(int(x.split(':')[2]) for x in spec.split(',') if x.startswith('local:cpu')); need = 2600 * n_cpu + 1500 * ('xpu' in spec)
        free = psutil.virtual_memory().available >> 20
        if free < need + 1500:   # a worker OOM crash leaves the coordinator waiting forever (V12 run 1) -> record instead
            res['ours_seedplane'][f'{name}|skipped'] = f'RAM: {free} MB free < {need} MB needed + 1500 margin'; save(); print('skip', name, flush=True); continue
        procs, clients = cli.start_workers(spec, a.bundle)
        try:
            for sched, fn in (('static', cli.distribute), ('dynamic', cli.distribute_dynamic)):
                if sched == 'dynamic' and len(clients) == 1: continue
                ts = [fn(clients, ids, wins, 'prefill')[1] for _ in range(REPS + 1)][1:]
                res['ours_seedplane'][f'{name}|{sched}'] = L / float(np.median(ts))
        finally:
            for _, c in clients:
                try: c.send(('close',))
                except Exception: pass
            for p in procs: p.terminate()
        print('seedplane', name, flush=True); save()
    # --- llama.cpp native (same model, F16 GGUF)
    if 'llama' not in parts: return
    lp = Path(a.llama)
    for t in (1, 2, 4, 6): res['llama_cpp'][f'cpu{t}'] = llama_bench(lp / 'cpu' / 'llama-bench.exe', a.gguf, ['-t', str(t), '-ngl', '0'])
    res['llama_cpp']['gpu_sycl_official'] = llama_bench(lp / 'sycl' / 'llama-bench.exe', a.gguf, ['-ngl', '99'])
    res['llama_cpp']['gpu_vulkan_official'] = llama_bench(lp / 'vulkan' / 'llama-bench.exe', a.gguf, ['-ngl', '99'])
    if a.synapse_llama:
        syn = Path(a.synapse_llama)
        for b_ in ('build', 'build_dnn', 'build_vk_submitstats'):
            exe = syn / b_ / 'bin' / 'llama-bench.exe'
            if exe.exists(): res['llama_cpp'][f'gpu_synapse_{b_}'] = llama_bench(exe, a.gguf, ['-ngl', '99'], oneapi='vk' not in b_)
    save(); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
