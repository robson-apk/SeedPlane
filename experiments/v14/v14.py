"""V14: exact layer pipeline over B580 + CPU + Mac via llama.cpp RPC, with the SeedPlane planner. See PROTOCOL.md."""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parents[1]))
from seedplane.planner import Device, plan_pipeline, _split

B = r"F:\seedplane_build\llama.cpp\build_vk\bin"; G = r"C:\Users\Windows 11\seedplane_v12\llama\qwen2.5-0.5b-instruct-fp16.gguf"
RPC = "127.0.0.1:50052,10.0.0.92:50053"; N_LAYERS, HIDDEN, LAYER_GB = 24, 896, 0.04
PROMPT = "The history of the printing press begins in the fifteenth century, when"


def bench(devs, ts=None, p='4096,16384', n='128', reps='3'):
    cmd = [B + r"\llama-bench.exe", '-m', G, '--rpc', RPC, '-dev', devs, '-ngl', '99', '-p', p, '-n', n, '-r', reps, '-o', 'json']
    if ts: cmd += ['-ts', ts]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    try: return {f"{'pp' if x['n_prompt'] else 'tg'}{x['n_prompt'] or x['n_gen']}": x['avg_ts'] for x in json.loads(r.stdout)}
    except Exception: return {'error': (r.stderr or r.stdout)[-400:]}


def generate(devs, ts):
    cmd = [B + r"\llama-completion.exe", '-m', G, '--rpc', RPC, '-dev', devs, '-ngl', '99', '-p', PROMPT, '-n', '64', '--temp', '0',
           '--seed', '1', '-no-cnv', '--no-display-prompt']
    if ts: cmd += ['-ts', ts]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, encoding='utf-8', errors='replace'); return r.stdout.strip()


def min1(split):
    s = list(split)
    while 0 in s: i = s.index(0); j = s.index(max(s)); s[i] += 1; s[j] -= 1
    return s


def main():
    out = ROOT / 'results'; out.mkdir(exist_ok=True); fp = out / 'v14.json'; k = 1
    while fp.exists(): fp = out / f'v14_{k}.json'; k += 1
    res = {'pre_bench_pp512': {}, 'conditions': {}, 'E1': {}}; save = lambda: fp.write_text(json.dumps(res, indent=1))
    srv = subprocess.Popen([B + r"\ggml-rpc-server.exe", '-H', '127.0.0.1', '-p', '50052', '-d', 'CPU', '-t', '5'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        for d in ('Vulkan0', 'RPC0', 'RPC1'):
            res['pre_bench_pp512'][d] = bench(d, p='512', n='0', reps='2').get('pp512'); save(); print('pre', d, res['pre_bench_pp512'][d], flush=True)
        r = res['pre_bench_pp512']; mem = {'Vulkan0': 11, 'RPC0': 12, 'RPC1': 16}; link = {'Vulkan0': 16, 'RPC0': 10, 'RPC1': 1}
        for name, devs in (('gpu+cpu', ['Vulkan0', 'RPC0']), ('gpu+cpu+mac', ['Vulkan0', 'RPC0', 'RPC1'])):
            ds = [Device(d, r[d], mem[d], link[d]) for d in devs]; p = plan_pipeline(ds, N_LAYERS, LAYER_GB, HIDDEN)
            m1 = min1(_split(ds, N_LAYERS, LAYER_GB)); res['conditions'][name] = {'planner': p.layers, 'planner_pred_tok_s': p.tok_s, 'min1': m1}
            for tag, ts in (('default', None), ('planner', '/'.join(map(str, p.layers))), ('min1', '/'.join(map(str, m1)))):
                res['conditions'][name][tag + '_result'] = bench(','.join(devs), ts); save(); print(name, tag, ts, res['conditions'][name][tag + '_result'], flush=True)
        res['conditions']['gpu'] = {'result': bench('Vulkan0')}; save(); print('gpu', res['conditions']['gpu'], flush=True)
        m1 = res['conditions']['gpu+cpu+mac']['min1']
        a = generate('Vulkan0', None); b = generate('Vulkan0,RPC0,RPC1', ','.join(map(str, m1)))
        res['E1'] = {'gpu_text': a, 'mesh_text': b, 'min1_split': m1, 'identical': a == b}; save(); print('E1 identical:', a == b, flush=True)
    finally:
        srv.kill()
    print('saved', fp)


if __name__ == '__main__':
    main()
