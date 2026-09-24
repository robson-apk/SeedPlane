"""V21 gates G2-G6 on the 5600X/B580. Paths are the V19/V20 working directories on that machine.

    python experiments/v21/run_gates.py <sp_v19 dir> <wikitext2_test.txt>
"""
import json, statistics as stt, subprocess, sys, time
from pathlib import Path
import numpy as np

root, wiki = Path(sys.argv[1]), Path(sys.argv[2]); sys.path.insert(0, str(root))
from seedplane.native import NativeEngine
exe = str(root / 'native' / 'build' / 'qwen_vk.exe'); v1 = root / 'run' / 'qwen05_v1.sp'; hfb = root / 'run' / 'qwen05_hf.sp'
work = root / 'v21'; work.mkdir(exist_ok=True); data = root / 'v20' / 'data'; v20 = root / 'v20' / 'out'
res = {}
def save(tag):
    (work / 'gates.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8'); print(tag, 'done', flush=True, file=sys.stderr)

def bench(bundle, *args):
    r = subprocess.run([exe, str(bundle), *map(str, args)], capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr)
    return json.loads(r.stdout)

# ---- G2 no regression
a = bench(v1, '--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 1)
ref = json.load(open(v20 / 'speed_shadow-batch.json'))['runs'][0]['tokens']
b = bench(hfb, '-n', 296, '--runs', 1)
orc = json.load(open(root / 'run' / 'oracle' / 'oracle.json'))['tokens']
bench(v1, '--score-file', data / 'quality_1.i32', '--nll-out', work / 'nll_q1.f32', '--runs', 1)
d = float(np.abs(np.fromfile(work / 'nll_q1.f32', np.float32) - np.fromfile(v20 / 'nll_q1_shadow-batch.f32', np.float32)).max())
res['G2'] = {'speed_tokens_equal_v20': a['runs'][0]['tokens'] == ref, 'v19_tokens_equal_oracle': b['runs'][0]['tokens'] == orc,
             'nll_q1_max_abs_diff_vs_v20': d, 'mode': a['mode']}
res['G2']['pass'] = res['G2']['speed_tokens_equal_v20'] and res['G2']['v19_tokens_equal_oracle'] and d <= 1e-5

save('G2')
# ---- G3 sampler vs reference definition (float64), 1e6 draws per case
dumps = [2999, 3500, 4020]
bench(v1, '--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 1, '--dump-at', ','.join(map(str, dumps)), '--dump-dir', work)
def reference(logits, T, k, p):
    z = logits.astype(np.float64) / T
    if k > 0: z = np.where(z < np.sort(z)[::-1][k - 1], -np.inf, z)
    if p < 1:
        order = np.argsort(-z, kind='stable'); q = np.exp(z[order] - z[order][0]); q /= q.sum()
        drop = np.cumsum(q) - q > p; z[order[drop]] = -np.inf
    q = np.exp(z - z.max()); return q / q.sum()
cases = [(0.7, 40, 0.9), (1.0, 0, 1.0), (0.7, 0, 0.5)]; g3 = []
for pos in dumps:
    lg = np.fromfile(work / f'logits_{pos}.f32', np.float32)
    for T, k, p in cases:
        out = work / f'counts_{pos}_{T}_{k}_{p}.u32'; n = 1_000_000
        subprocess.run([exe, '--sample-test', str(work / f'logits_{pos}.f32'), str(T), str(k), str(p), str(n), '7', str(out)], check=True)
        emp = np.fromfile(out, np.uint32).astype(np.float64) / n; ref_p = reference(lg, T, k, p)
        outside = int((np.fromfile(out, np.uint32)[ref_p == 0]).sum())
        if k == 0 and p >= 1:                                   # pool the long tail into one bucket
            tail = ref_p < 1e-4; tv = 0.5 * (np.abs(emp[~tail] - ref_p[~tail]).sum() + abs(emp[tail].sum() - ref_p[tail].sum()))
        else: tv = 0.5 * np.abs(emp - ref_p).sum()
        g3.append({'pos': pos, 'T': T, 'k': k, 'p': p, 'support': int((ref_p > 0).sum()), 'outside_draws': outside, 'tv': float(tv)})
res['G3'] = {'cases': g3, 'pass': all(c['outside_draws'] == 0 and c['tv'] <= 0.01 for c in g3)}

save('G3')
# ---- G4 sampling cost (median of 3, same prompt and mode as V20 speed)
greedy = bench(v1, '--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 3)
samp = bench(v1, '--prompt-file', data / 'speed_prompt.i32', '-n', 1024, '--runs', 3, '--temperature', 0.7, '--top-k', 40, '--top-p', 0.9, '--seed', 3)
gm = stt.median(r['decode_tok_s'] for r in greedy['runs']); sm = stt.median(r['decode_tok_s'] for r in samp['runs'])
res['G4'] = {'greedy_tok_s': [r['decode_tok_s'] for r in greedy['runs']], 'sampling_tok_s': [r['decode_tok_s'] for r in samp['runs']],
             'ratio': sm / gm, 'pass': sm >= 0.95 * gm}

save('G4')
# ---- G5 in-session continuation == fresh prefill (greedy), across a shard boundary
text = wiki.read_text(encoding='utf-8')[50000:53500]
with NativeEngine(v1, engine=exe, temperature=0) as eng:
    t1 = ''.join(eng.chat('Summarize this text in two sentences:\n' + text, reset=True, max_new_tokens=64)); s1 = dict(eng.last)
    t2 = ''.join(eng.chat('Now give it a short title.', max_new_tokens=64)); s2 = dict(eng.last)
    toks = eng.state()['tokens']; gen2 = toks[len(toks) - s2['generated']:]; prefix = toks[:len(toks) - s2['generated']]
    t2b = ''.join(eng.generate(ids=prefix, reset=True, max_new_tokens=64))
    toks_b = eng.state()['tokens']; gen2_fresh = toks_b[len(prefix):]
res['G5'] = {'turn1_tokens': s1['generated'], 'turn2_tokens': s2['generated'], 'conversation_positions': len(toks),
             'crossed_boundary_512': len(prefix) > 512, 'equal': gen2 == gen2_fresh, 'turn2_text': t2, 'turn2_text_fresh': t2b}
res['G5']['pass'] = res['G5']['equal']

save('G5')
# ---- G6 end to end through the Python CLI: convert --native, then chat --native with 2 scripted turns
cli_sp = work / 'cli.sp'
t0 = time.time()
r = subprocess.run([sys.executable, '-m', 'seedplane.cli', 'convert', str(root.parent / 'seedplane_v12' / 'qwen05.sp'), str(cli_sp), '--native'],
                   capture_output=True, text=True, cwd=root)
conv_ok = r.returncode == 0
r2 = subprocess.run([sys.executable, '-m', 'seedplane.cli', 'chat', str(cli_sp), '--native', '--engine', exe, '-n', '48', '--seed', '5'],
                    input='Olá! Em uma frase, o que é uma GPU?\nE uma CPU?\n', capture_output=True, text=True, encoding='utf-8', cwd=root)
answers = [l.split('Qwen> ', 1)[1] for l in r2.stdout.splitlines() if 'Qwen> ' in l]
res['G6'] = {'convert_ok': conv_ok, 'convert_out': r.stdout.strip()[-300:], 'chat_returncode': r2.returncode, 'answers': answers,
             'stderr_tail': r2.stderr[-300:], 'seconds': time.time() - t0}
res['G6']['pass'] = conv_ok and r2.returncode == 0 and len(answers) == 2 and all(a.strip() for a in answers)
save('G6')
print(json.dumps(res, indent=1, ensure_ascii=False))
