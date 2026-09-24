import json, numpy as np, statistics
d = r'C:\Users\Windows 11\sp_vk' + '\\'
o = json.load(open(d + 'oracle.json')); v = json.load(open(d + 'vk_run1.json'))
a = np.fromfile(d + 'vk_logits.f32', np.float32); b = np.fromfile(d + 'oracle_logits.f32', np.float32)
err = np.abs(a - b)
rt = [r['decode_tok_s'] for r in v['runs']]
div = []
for r in v['runs']:
    t = r['tokens']; k = next((i for i, (x, y) in enumerate(zip(t, o['tokens'])) if x != y), None); div.append(k)
print(json.dumps({'G1_first8_identical': all(r['tokens'][:8] == o['tokens'][:8] for r in v['runs']),
  'G2_max_abs_logit_err': float(err.max()), 'mean_abs_logit_err': float(err.mean()), 'G2_pass': bool(err.max() <= 0.10),
  'argmax_same': int(a.argmax()) == int(b.argmax()),
  'runs_tok_s': rt, 'G3_median_tok_s': statistics.median(rt), 'G3_pass': statistics.median(rt) > 296,
  'first_divergence_vs_oracle_per_run': div, 'runs_identical_to_each_other': len({tuple(r['tokens']) for r in v['runs']}) == 1}, indent=1))
