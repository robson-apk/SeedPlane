"""PROTOCOL addendum 4: try training seeds 1,2,3; keep the first that passes the sanity gate; archive failures."""
import json, os, subprocess, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); py = sys.executable; log = []
for seed in (1, 2, 3):
    with open(f'train_s{seed}.log', 'w') as f:
        subprocess.run([py, 'v6.py', 'train', '--cache', 'cache.pt', '--steps', '10000', '--seed', str(seed)], stdout=f, stderr=subprocess.STDOUT, check=True)
    subprocess.run([py, 'diag_model.py', 'cache.pt'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    s = json.load(open('results/model_sanity.json')); l15 = s['by_rate']['0.15']['loss']; ok = l15 <= s['unigram_entropy'] - 1.0
    log.append({'seed': seed, 'loss_at_0.15': l15, 'unigram': s['unigram_entropy'], 'gate_pass': ok}); print(log[-1], flush=True)
    if ok: break
    for a, b in [('v6_model.pt', f'run4_s{seed}_v6_model.pt'), ('results/model_sanity.json', f'results/run4_s{seed}_model_sanity.json'), ('results/train_log.json', f'results/run4_s{seed}_train_log.json')]:
        os.replace(a, b)
json.dump(log, open('results/seed_attempts.json', 'w'), indent=1)
print('GATE_PASSED' if log[-1]['gate_pass'] else 'ALL_SEEDS_FAILED', flush=True)
