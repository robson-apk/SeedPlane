"""V23 gates on the X79 (RX 570 / RADV / Ubuntu). Run from the clean checkout:

    ../venv/bin/python experiments/v23/run_gates.py <workdir>      (workdir holds build_gcc, build_clang, qwen05.sp, data)
"""
import glob, json, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

W = Path(sys.argv[1]); src = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(src))
exe = str(W / 'build_gcc' / 'qwen_vk'); B = str(W / 'qwen05.sp'); data = W / 'data'; out = W / 'v23'; out.mkdir(exist_ok=True)
res = {'commit': (W / 'commit.txt').read_text().strip()}
def save(): (out / 'gates.json').write_text(json.dumps(res, indent=1))
def run(args, env=None, exe_=exe):
    r = subprocess.run([exe_, B, *map(str, args)], capture_output=True, text=True, env=env)
    if r.returncode: raise SystemExit(f'{args}: {r.stderr[-600:]}')
    return r
def bench(*args, exe_=exe): return json.loads(run(args, exe_=exe_).stdout)

# G1 builds, G2 shaders
res['G1'] = {'gcc': (W / 'build_gcc' / 'qwen_vk').exists(), 'clang': (W / 'build_clang' / 'qwen_vk').exists()}
res['G1']['pass'] = res['G1']['gcc'] and res['G1']['clang']
cm = (src / 'native' / 'vulkan_decode' / 'CMakeLists.txt').read_text()
expected = 4 + len(cm.split('foreach(s ')[1].split(')')[0].split())            # gemv1/2 + swiglu1/2 + singles
spv = sorted(Path(p).name for p in glob.glob(str(W / 'build_gcc' / 'shaders' / '*.spv')))
res['G2'] = {'spv_files': len(spv), 'expected': expected, 'pass': len(spv) == expected}
save()

# G3 validation layers (Khronos layer from the LunarG SDK tarball if present)
layer_dirs = glob.glob(str(W / 'vksdk' / '*' / 'x86_64' / 'share' / 'vulkan' / 'explicit_layer.d'))
if layer_dirs:
    env = dict(os.environ, SP_VK_VALIDATE='1', VK_LAYER_PATH=layer_dirs[0],
               LD_LIBRARY_PATH=str(Path(layer_dirs[0]).parents[2] / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    r = subprocess.run([exe, B, '-n', '8', '--runs', '1', '--full'], capture_output=True, text=True, env=env)
    errs = [l for l in r.stderr.splitlines() if 'VUID' in l or 'Validation Error' in l]
    loaded = 'VK_LAYER_KHRONOS_validation' not in r.stderr or r.returncode == 0
    res['G3'] = {'executed': True, 'returncode': r.returncode, 'validation_errors': len(errs), 'first_errors': errs[:5], 'stderr_tail': r.stderr[-400:]}
    res['G3']['pass'] = r.returncode == 0 and not errs
else:
    res['G3'] = {'executed': False, 'reason': 'Khronos validation layer not available (no sudo; SDK tarball not extracted)', 'pass': None}
save()

# G5 tokens: full attention vs V18 CPU FP32 oracle; V19 plan vs V19 window oracle
v18 = json.load(open(src / 'experiments' / 'v18' / 'cpu_fp32_oracle.json'))['tokens']
full = bench('-n', 128, '--runs', 1, '--full', '--dump-at', 3, '--dump-dir', out)
v19 = json.load(open(src / 'experiments' / 'v19' / 'oracle.json'))['tokens']
plan = bench('-n', 296, '--runs', 1, '--shard', 64, '--halo', 32, '--sinks', 4)
res['G4'] = {'devices': sorted({full['device'], plan['device']}), 'pass': all('RX 570' in d for d in (full['device'], plan['device']))}
res['G5'] = {'full_first8_equal': full['runs'][0]['tokens'][:8] == v18[:8], 'full_128_equal': full['runs'][0]['tokens'] == v18,
             'plan_296_equal': plan['runs'][0]['tokens'] == v19,
             'full_first_div': next((i for i, (a, b) in enumerate(zip(full['runs'][0]['tokens'], v18)) if a != b), None),
             'plan_first_div': next((i for i, (a, b) in enumerate(zip(plan['runs'][0]['tokens'], v19)) if a != b), None)}
res['G5']['pass'] = res['G5']['full_first8_equal'] and res['G5']['full_128_equal'] and res['G5']['plan_296_equal']
save()

# G6 logits vs CPU FP32 oracle computed here (PyTorch Qwen2Engine on the same HF weights)
import torch
from seedplane.qwen_engine import Qwen2Engine
hf = (W / 'hf_path.txt').read_text().strip(); torch.set_num_threads(16)
eng = Qwen2Engine(hf, 'cpu', torch.float32); lg, _ = eng.forward([9707, 11, 1879, 0], eng.new_cache(8))
ref = lg[-1].numpy().astype(np.float32); got = np.fromfile(out / 'logits_3.f32', np.float32)
res['G6'] = {'max_abs_logit_error': float(np.abs(ref - got).max()), 'argmax_equal': int(ref.argmax()) == int(got.argmax())}
res['G6']['pass'] = res['G6']['max_abs_logit_error'] <= 0.10
save()

# G7 NLL on WikiText chunk 1 vs the B580 (V20 shadow-batch)
run(['--score-file', data / 'quality_1.i32', '--nll-out', out / 'nll_q1.f32', '--runs', 1])
rx = np.fromfile(out / 'nll_q1.f32', np.float32); b580 = np.load(src / 'experiments' / 'v20' / 'raw' / 'nll_per_position.npz')['nll_q1_shadow-batch']
res['G7'] = {'mean_nll_rx570': float(rx.mean()), 'mean_nll_b580': float(b580.mean()), 'mean_abs_diff': float(abs(rx.mean() - b580.mean())),
             'max_pos_abs_diff': float(np.abs(rx - b580).max()), 'positions': int(rx.size)}
res['G7']['pass'] = res['G7']['mean_abs_diff'] <= 1e-3 and res['G7']['max_pos_abs_diff'] <= 0.05
save()

# G8 stability + G9 VRAM (sysfs sampled while 3 speed runs execute in one process)
vram_file = next(iter(glob.glob('/sys/class/drm/card*/device/mem_info_vram_used')), None); samples = []; stop = False
def sampler():
    while not stop:
        if vram_file: samples.append((time.time(), int(open(vram_file).read())))
        time.sleep(0.05)
th = threading.Thread(target=sampler); th.start(); t0 = time.time()
speed = bench('--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 3)
stop = True; th.join(); t1 = time.time()
toks = [r['tokens'] for r in speed['runs']]
v20 = json.load(open(src / 'experiments' / 'v20' / 'raw' / 'speed_shadow-batch.json'))['runs'][0]['tokens']
res['G8'] = {'runs_identical': all(t == toks[0] for t in toks), 'equal_b580_v20_reported': toks[0] == v20,
             'first_div_vs_b580': next((i for i, (a, b) in enumerate(zip(toks[0], v20)) if a != b), None), 'device': speed['device']}
res['G8']['pass'] = res['G8']['runs_identical']
if samples:
    d = t1 - t0; early = [v for t, v in samples if 0.2 * d <= t - t0 <= 0.4 * d]; late = [v for t, v in samples if t - t0 >= 0.8 * d]
    res['G9'] = {'vram_peak_20_40pct_mib': max(early) / 2**20, 'vram_peak_last20pct_mib': max(late) / 2**20, 'samples': len(samples)}
    res['G9']['pass'] = max(late) <= 1.05 * max(early)
else: res['G9'] = {'pass': None, 'reason': 'no amdgpu VRAM sysfs'}
save()

# Reported: full vs SeedPlane speed, sampling, latency, spikes, clang build parity
fullspd = bench('--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 3, '--full')
samp = {name: bench('--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 1, *a)['runs'][0]['decode_tok_s']
        for name, a in (('T0.7_k40_p0.9', ['--temperature', 0.7, '--top-k', 40, '--top-p', 0.9]), ('T1.0', ['--temperature', 1.0]),
                        ('T0.7_p0.9', ['--temperature', 0.7, '--top-p', 0.9]))}
clang = bench('--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 1, exe_=str(W / 'build_clang' / 'qwen_vk'))
med = lambda j, k: float(np.median([r[k] for r in j['runs']]))
res['reported'] = {'seedplane_tok_s': [r['decode_tok_s'] for r in speed['runs']], 'full_tok_s': [r['decode_tok_s'] for r in fullspd['runs']],
                   'prefill_s_median': med(speed, 'prefill_seconds'), 'lat_median_ms': med(speed, 'lat_median_ms'),
                   'lat_p99_ms': med(speed, 'lat_p99_ms'), 'lat_max_ms': [r['lat_max_ms'] for r in speed['runs']],
                   'sampling_tok_s': samp, 'clang_tok_s': clang['runs'][0]['decode_tok_s'], 'clang_tokens_equal_gcc': clang['runs'][0]['tokens'] == toks[0],
                   'load_seconds': speed['load_seconds'], 'subgroup_size': speed['subgroup_size'], 'shared_bytes': speed['shared_bytes'],
                   'kv_bytes_per_cache': speed['kv_bytes_per_cache']}
save(); print(json.dumps({k: v.get('pass') for k, v in res.items() if isinstance(v, dict) and 'pass' in v}))
