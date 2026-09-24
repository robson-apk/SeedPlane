"""V21 post-hoc run after the G4 failure (sampler fixed): G3 again on the final sampler, extended with high-entropy
logits (teacher-forced WikiText positions) and synthetic logits, and G4 again (median of 3). Labelled post-hoc.

    python experiments/v21/posthoc_g3_g4.py <sp_v19 dir>
"""
import json, statistics as stt, subprocess, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]); exe = str(root / 'native' / 'build' / 'qwen_vk.exe'); v1 = root / 'run' / 'qwen05_v1.sp'
work = root / 'v21'; data = root / 'v20' / 'data'; res = {}
def bench(*args):
    r = subprocess.run([exe, str(v1), *map(str, args)], capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr)
    return json.loads(r.stdout)
def reference(logits, T, k, p):
    z = logits.astype(np.float64) / T
    if k > 0: z = np.where(z < np.sort(z)[::-1][k - 1], -np.inf, z)
    if p < 1:
        order = np.argsort(-z, kind='stable'); q = np.exp(z[order] - z[order][0]); q /= q.sum()
        drop = np.cumsum(q) - q > p; z[order[drop]] = -np.inf
    q = np.exp(z - z.max()); return q / q.sum()

# logits: the 3 pre-registered decode positions + 3 teacher-forced WikiText positions (high entropy) + 2 synthetic
bench('--score-file', data / 'quality_1.i32', '--runs', 1, '--dump-at', '1000,2000,3000', '--dump-dir', work)
rng = np.random.default_rng(21)
for name, scale in (('synthetic_s2', 2.0), ('synthetic_s4', 4.0)):
    (rng.standard_normal(151936) * scale).astype(np.float32).tofile(work / f'logits_{name}.f32')
sources = {'decode_2999': 'logits_2999.f32', 'decode_3500': 'logits_3500.f32', 'decode_4020': 'logits_4020.f32',
           'wiki_1000': 'logits_1000.f32', 'wiki_2000': 'logits_2000.f32', 'wiki_3000': 'logits_3000.f32',
           'synthetic_s2': 'logits_synthetic_s2.f32', 'synthetic_s4': 'logits_synthetic_s4.f32'}
cases = [(0.7, 40, 0.9), (1.0, 0, 1.0), (0.7, 0, 0.5)]; out = []
for src, f in sources.items():
    lg = np.fromfile(work / f, np.float32)
    for T, k, p in cases:
        cf = work / f'counts_{src}_{T}_{k}_{p}.u32'; n = 1_000_000
        subprocess.run([exe, '--sample-test', str(work / f), str(T), str(k), str(p), str(n), '7', str(cf)], check=True)
        cnt = np.fromfile(cf, np.uint32); emp = cnt / n; ref = reference(lg, T, k, p)
        if k == 0 and p >= 1:
            tail = ref < 1e-4; tv = 0.5 * (np.abs(emp[~tail] - ref[~tail]).sum() + abs(emp[tail].sum() - ref[tail].sum()))
        else: tv = 0.5 * np.abs(emp - ref).sum()
        ent = float(-(ref[ref > 0] * np.log(ref[ref > 0])).sum())
        out.append({'logits': src, 'T': T, 'k': k, 'p': p, 'support': int((ref > 0).sum()), 'entropy_nats': ent,
                    'outside_draws': int(cnt[ref == 0].sum()), 'tv': float(tv)})
pre = [c for c in out if c['logits'].startswith('decode_')]
res['G3_posthoc'] = {'cases': out, 'pass_preregistered_logits': all(c['outside_draws'] == 0 and c['tv'] <= 0.01 for c in pre),
                     'pass_all_logits': all(c['outside_draws'] == 0 and c['tv'] <= 0.01 for c in out),
                     'nontrivial_cases': sum(c['support'] > 1 for c in out)}
greedy = bench('--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 3)
samp = bench('--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 3, '--temperature', 0.7, '--top-k', 40, '--top-p', 0.9, '--seed', 3)
gm = stt.median(r['decode_tok_s'] for r in greedy['runs']); sm = stt.median(r['decode_tok_s'] for r in samp['runs'])
res['G4_posthoc'] = {'greedy_tok_s': [r['decode_tok_s'] for r in greedy['runs']], 'sampling_tok_s': [r['decode_tok_s'] for r in samp['runs']],
                     'ratio': sm / gm, 'pass_threshold_0.95': sm >= 0.95 * gm,
                     'sampled_tokens_distinct_from_greedy': samp['runs'][0]['tokens'] != greedy['runs'][0]['tokens']}
(work / 'posthoc.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
