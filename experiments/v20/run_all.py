"""Run the full V20 matrix on the B580 and store raw outputs in <out>/ (JSON per config, NLL arrays per chunk)."""
import json, subprocess, sys, time
from pathlib import Path

exe, bundle, data, out = map(Path, sys.argv[1:5]); out.mkdir(parents=True, exist_ok=True)
MODES = {'full_split': ['--full'], 'full_old': ['--full', '--attn', 'old'], 'rebuild': ['--mode', 'rebuild'],
         'rebuild-batch': ['--mode', 'rebuild-batch'], 'shadow': ['--mode', 'shadow'], 'shadow-batch': ['--mode', 'shadow-batch'],
         'reuse': ['--mode', 'reuse']}

def run(tag, args):
    t = time.time(); r = subprocess.run([str(exe), str(bundle)] + args, capture_output=True, text=True)
    if r.returncode: raise SystemExit(f'{tag} failed: {r.stderr[-800:]}')
    (out / f'{tag}.json').write_text(r.stdout); print(f'{tag} done in {time.time() - t:.1f}s', flush=True)

for q in (1, 2, 3):
    for m, a in MODES.items():
        run(f'quality{q}_{m}', a + ['--score-file', str(data / f'quality_{q}.i32'), '--nll-out', str(out / f'nll_q{q}_{m}.f32'), '--runs', '1'])
for m, a in MODES.items():
    run(f'speed_{m}', a + ['--prompt-file', str(data / 'speed_prompt.i32'), '-n', '1024', '--runs', '3'])
