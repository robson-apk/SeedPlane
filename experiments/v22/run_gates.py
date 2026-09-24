"""V22 gates on the 5600X/B580: G1 GPU-assisted sampler distribution, G2 speed (interleaved runs), G3 greedy tokens,
G4 = V21 G5/G6 on the new binary.

    python experiments/v22/run_gates.py <sp_v19 dir> <wikitext2_test.txt>
"""
import json, statistics as stt, subprocess, sys
from pathlib import Path
import numpy as np

root, wiki = Path(sys.argv[1]), sys.argv[2]; exe = str(root / 'native' / 'build' / 'qwen_vk.exe'); v1 = root / 'run' / 'qwen05_v1.sp'
work = root / 'v22'; work.mkdir(exist_ok=True); v21 = root / 'v21'; data = root / 'v20' / 'data'; res = {}
def save(): (work / 'gates.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
def reference(logits, T, k, p):
    z = logits.astype(np.float64) / T
    if k > 0: z = np.where(z < np.sort(z)[::-1][k - 1], -np.inf, z)
    if p < 1:
        order = np.argsort(-z, kind='stable'); q = np.exp(z[order] - z[order][0]); q /= q.sum()
        drop = np.cumsum(q) - q > p; z[order[drop]] = -np.inf
    q = np.exp(z - z.max()); return q / q.sum()

# ---- G1: same 8 logit vectors as the V21 post-hoc G3, 3 configurations, 10^6 draws each
sources = ['2999', '3500', '4020', '1000', '2000', '3000', 'synthetic_s2', 'synthetic_s4']
cases = [(0.7, 40, 0.9), (1.0, 0, 1.0), (0.7, 0, 0.5)]; out = []
for src in sources:
    f = v21 / f'logits_{src}.f32'; lg = np.fromfile(f, np.float32)
    for T, k, p in cases:
        cf = work / f'counts_{src}_{T}_{k}_{p}.u32'; n = 1_000_000
        r = subprocess.run([exe, str(v1), '--sample-test-gpu', str(f), str(T), str(k), str(p), str(n), '11', str(cf)], capture_output=True, text=True)
        if r.returncode: raise SystemExit(r.stderr)
        info = json.loads(r.stdout); cnt = np.fromfile(cf, np.uint32); emp = cnt / n; ref = reference(lg, T, k, p)
        if k == 0 and p >= 1:
            tail = ref < 1e-4; tv = 0.5 * (np.abs(emp[~tail] - ref[~tail]).sum() + abs(emp[tail].sum() - ref[tail].sum()))
        else: tv = 0.5 * np.abs(emp - ref).sum()
        out.append({'logits': src, 'T': T, 'k': k, 'p': p, 'path': info['path'], 'candidates': info['candidates'],
                    'support': int((ref > 0).sum()), 'outside_draws': int(cnt[ref == 0].sum()), 'tv': float(tv)})
res['G1'] = {'cases': out, 'fallbacks': sum(c['path'] == 'fallback' for c in out), 'max_tv': max(c['tv'] for c in out),
             'pass': all(c['outside_draws'] == 0 and c['tv'] <= 0.01 for c in out)}
save(); print('G1', res['G1']['pass'], res['G1']['max_tv'], file=sys.stderr, flush=True)

# ---- G2 speed (interleaved: each round runs every configuration once) + G3 greedy tokens
cfgs = {'greedy': [], 'T0.7_k40_p0.9': ['--temperature', '0.7', '--top-k', '40', '--top-p', '0.9', '--seed', '3'],
        'T1.0_k0_p1': ['--temperature', '1.0', '--seed', '3'], 'T0.7_k0_p0.9': ['--temperature', '0.7', '--top-p', '0.9', '--seed', '3']}
runs = {c: [] for c in cfgs}
for rnd in range(3):
    for c, a in cfgs.items():
        r = subprocess.run([exe, str(v1), '--prompt-file', str(data / 'speed_prompt.i32'), '-n', '1024', '--runs', '1'] + a, capture_output=True, text=True)
        if r.returncode: raise SystemExit(r.stderr)
        runs[c].append(json.loads(r.stdout)['runs'][0])
med = {c: stt.median(r['decode_tok_s'] for r in runs[c]) for c in cfgs}
res['G2'] = {'tok_s': {c: [r['decode_tok_s'] for r in runs[c]] for c in cfgs}, 'median': med,
             'ratio_vs_greedy': {c: med[c] / med['greedy'] for c in cfgs if c != 'greedy'},
             'lat_median_ms': {c: [r['lat_median_ms'] for r in runs[c]] for c in cfgs},
             'fallbacks': {c: [r['fallbacks'] for r in runs[c]] for c in cfgs}}
res['G2']['pass'] = all(v >= 0.95 for v in res['G2']['ratio_vs_greedy'].values())
ref = json.load(open(root / 'v20' / 'out' / 'speed_shadow-batch.json'))['runs'][0]['tokens']
res['G3'] = {'greedy_tokens_equal_v20_all_rounds': all(r['tokens'] == ref for r in runs['greedy'])}
res['G3']['pass'] = res['G3']['greedy_tokens_equal_v20_all_rounds']
save(); print('G2', res['G2']['ratio_vs_greedy'], file=sys.stderr, flush=True)

# ---- G4: V21 G5 / G6 on the new binary
r = subprocess.run([sys.executable, str(root / 'experiments' / 'v21' / 'run_gates.py'), str(root), wiki], capture_output=True, text=True,
                   encoding='utf-8', cwd=root)
g = json.load(open(v21 / 'gates.json', encoding='utf-8'))
res['G4'] = {'v21_G5': g['G5'], 'v21_G6': g['G6'], 'v21_G4_ratio_same_binary': g['G4']['ratio'], 'pass': g['G5']['pass'] and g['G6']['pass']}
save(); print(json.dumps({k: v['pass'] for k, v in res.items()}))
