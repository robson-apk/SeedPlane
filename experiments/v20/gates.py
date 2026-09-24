"""Evaluate the V20 gates (PROTOCOL.md + Addendum 1) from the raw outputs in <out>/."""
import json, statistics as stt, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); EXACT = ('shadow', 'shadow-batch', 'rebuild-batch'); ALL = ('full_split', 'full_old', 'rebuild') + EXACT + ('reuse',)
J = lambda tag: json.load(open(out / f'{tag}.json'))
if (out / 'nll_per_position.npz').exists():                 # archived form (experiments/v20/raw)
    _z = np.load(out / 'nll_per_position.npz'); nll = {(q, m): _z[f'nll_q{q}_{m}'] for q in (1, 2, 3) for m in ALL}
else:
    nll = {(q, m): np.fromfile(out / f'nll_q{q}_{m}.f32', np.float32) for q in (1, 2, 3) for m in ALL}
mean = {(q, m): J(f'quality{q}_{m}')['runs'][0]['nll_mean'] for q in (1, 2, 3) for m in ALL}
spd = {m: J(f'speed_{m}') for m in ALL}
med = {m: stt.median(r['decode_tok_s'] for r in spd[m]['runs']) for m in ALL}
maxdiff = lambda a, b: max(float(np.abs(nll[(q, a)] - nll[(q, b)]).max()) for q in (1, 2, 3))
res = {'coverage': {'positions_scored_per_chunk': int(nll[(1, 'rebuild')].size), 'chunks': 3,
                    'boundaries_scored_per_chunk': J('quality1_rebuild')['runs'][0]['boundaries'],
                    'speed_boundaries_in_decode': spd['rebuild']['runs'][0]['boundaries']},
       'median_decode_tok_s': med,
       'runs_decode_tok_s': {m: [r['decode_tok_s'] for r in spd[m]['runs']] for m in ALL},
       'prefill_seconds_median': {m: stt.median(r['prefill_seconds'] for r in spd[m]['runs']) for m in ALL},
       'lat_ms': {m: {'median': [r['lat_median_ms'] for r in spd[m]['runs']], 'p99': [r['lat_p99_ms'] for r in spd[m]['runs']],
                      'max': [r['lat_max_ms'] for r in spd[m]['runs']], 'max_pos': [r['lat_max_pos'] for r in spd[m]['runs']]} for m in ALL},
       'nll_mean': {m: [mean[(q, m)] for q in (1, 2, 3)] for m in ALL},
       'kv_bytes_per_cache': {m: spd[m]['kv_bytes_per_cache'] for m in ALL}}
ref_tokens = spd['rebuild']['runs'][0]['tokens']
res['G1'] = {m: {'max_abs_nll_diff_vs_rebuild': maxdiff(m, 'rebuild'),
                 'tokens_equal_rebuild_all_runs': all(r['tokens'] == ref_tokens for r in spd[m]['runs'])} for m in EXACT}
for m in EXACT: res['G1'][m]['pass'] = res['G1'][m]['max_abs_nll_diff_vs_rebuild'] <= 1e-3 and res['G1'][m]['tokens_equal_rebuild_all_runs']
res['G1_rebuild_runs_self_consistent'] = all(r['tokens'] == ref_tokens for r in spd['rebuild']['runs'])
res['G2'] = {}
for m in EXACT:
    ratio = med[m] / med['reuse']; worst = max(r['lat_max_ms'] / r['lat_median_ms'] for r in spd[m]['runs'])
    res['G2'][m] = {'ratio_vs_reuse': ratio, 'worst_max_over_median_latency': worst, 'pass_speed': ratio >= 0.90, 'pass_latency': worst <= 3.0}
res['G2_latency_same_metric_other_modes'] = {m: max(r['lat_max_ms'] / r['lat_median_ms'] for r in spd[m]['runs']) for m in ('full_split', 'full_old', 'rebuild', 'reuse')}
rq = [mean[(q, 'reuse')] / mean[(q, 'rebuild')] for q in (1, 2, 3)]
res['G3'] = {'reuse_over_exact_nll_per_chunk': rq, 'manip_max_abs_diff': maxdiff('reuse', 'rebuild'),
             'pass': all(x <= 1.005 for x in rq) and maxdiff('reuse', 'rebuild') > 1e-3}
d_old = float(np.abs(nll[(1, 'full_split')] - nll[(1, 'full_old')]).max())
res['G4'] = {'speedup_split_over_old': med['full_split'] / med['full_old'], 'max_abs_nll_diff_q1': d_old,
             'nll_diff_all_chunks': maxdiff('full_split', 'full_old')}
res['G4']['pass'] = res['G4']['speedup_split_over_old'] >= 1.2 and d_old <= 1e-3
best_exact = max(EXACT, key=lambda m: med[m]); best_full = max(('full_split', 'full_old'), key=lambda m: med[m])
res['G5'] = {'best_exact': best_exact, 'best_exact_tok_s': med[best_exact], 'best_full': best_full, 'best_full_tok_s': med[best_full],
             'speedup': med[best_exact] / med[best_full], 'pass': med[best_exact] > med[best_full],
             'nll_cost_exact_vs_full_per_chunk': [mean[(q, 'rebuild')] / mean[(q, 'full_split')] - 1 for q in (1, 2, 3)],
             'nll_cost_reuse_vs_full_per_chunk': [mean[(q, 'reuse')] / mean[(q, 'full_split')] - 1 for q in (1, 2, 3)]}
default = best_exact
if res['G3']['pass'] and med['reuse'] >= 1.05 * med[best_exact]: default = 'reuse'
res['decision_default_mode'] = default
print(json.dumps(res, indent=1))
