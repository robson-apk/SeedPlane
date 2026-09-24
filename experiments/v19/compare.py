"""Evaluate the V19 gates from the raw outputs (paths relative to the run directory given as argv[1])."""
import json, statistics, sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1]); v18 = json.load(open(Path(__file__).resolve().parents[1] / 'v18' / 'b580_vk_decode.json'))
conv = {k: json.load(open(d / f'{k}.sp' / 'seedplane.json'))['weights'] for k in ('qwen05_hf', 'qwen05_v1')}
plan = json.load(open(d / 'native_plan.json')); fullr = json.load(open(d / 'native_full.json')); orc = json.load(open(d / 'oracle' / 'oracle.json'))
P = plan['prompt_tokens']; dumps = [3, 63, 64, 65, 128, 200, 299]

def first_div(a, b): return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
errs = {p: float(np.abs(np.fromfile(d / 'dump_plan' / f'logits_{p}.f32', np.float32) - np.fromfile(d / 'oracle' / f'logits_{p}.f32', np.float32)).max()) for p in dumps}
manip = float(np.abs(np.fromfile(d / 'dump_plan' / 'logits_65.f32', np.float32) - np.fromfile(d / 'dump_full' / 'logits_65.f32', np.float32)).max())
runs = plan['runs']; steady = [r['steady_tok_s'] for r in runs]
g2_upto = 130 - P + 1
res = {
    'G1_sha_hf': conv['qwen05_hf']['sha256'], 'G1_sha_v1': conv['qwen05_v1']['sha256'],
    'G1_pass': conv['qwen05_hf']['sha256'] == conv['qwen05_v1']['sha256'],
    'fp16_stats': {k: {'overflow': v['fp16_overflow'], 'flushed': v['fp16_flushed_to_zero']} for k, v in conv.items()},
    'G2_first_divergence_per_run': [first_div(r['tokens'], orc['tokens']) for r in runs],
    'G2_pass': all(r['tokens'][:g2_upto] == orc['tokens'][:g2_upto] for r in runs),
    'G3_logit_err_by_pos': errs, 'G3_pass': max(errs.values()) <= 0.10,
    'G4_manip_max_diff_pos65': manip, 'G4_full_equals_v18_first128': fullr['runs'][0]['tokens'][:128] == v18['runs'][0]['tokens'][:128],
    'G4_full_first_div_vs_v18': first_div(fullr['runs'][0]['tokens'], v18['runs'][0]['tokens']),
    'plan_vs_full_first_token_div': first_div(runs[0]['tokens'], fullr['runs'][0]['tokens']),
    'G5_steady_tok_s': steady, 'G5_median': statistics.median(steady), 'G5_pass': statistics.median(steady) >= 245,
    'overall_tok_s_plan': [r['decode_tok_s'] for r in runs], 'overall_tok_s_full': [r['decode_tok_s'] for r in fullr['runs']],
    'rebuilds': runs[0]['rebuilds'], 'rebuild_submits': runs[0]['rebuild_submits'],
    'kv_bytes_plan': plan['kv_bytes'], 'kv_bytes_full': fullr['kv_bytes'], 'load_seconds': plan['load_seconds'],
}
res['G4_pass'] = manip > 0.01 and res['G4_full_equals_v18_first128']
print(json.dumps(res, indent=1))
