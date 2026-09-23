"""V12 quality: perplexity of Qwen2.5-0.5B with full attention vs SeedPlane shard windows (no retraining). See PROTOCOL.md.

python v12_quality.py --model <hf id or path> --tinystories <txt> --wikitext <txt>
"""
import argparse, json, math, sys, itertools
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'
sys.path.insert(0, str(ROOT.parents[1] / 'seedplane')); import engine

SEEDS = (11, 23, 37); LS = (1024, 2048, 4096); N_SEQ = 8
VARIANTS = {'full': None, **{f'sp_S{S}_H{H}_k{k}': engine.ShardPlan(S, H, k) for S, H, k in itertools.product((256, 512), (64, 256), (0, 4))}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct'); ap.add_argument('--tinystories'); ap.add_argument('--wikitext')
    a = ap.parse_args(); dev = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    model, tok = engine.load_model(a.model, dev)
    corpora = {'tinystories': Path(a.tinystories).read_text(encoding='utf-8', errors='ignore')[:3_000_000], 'wikitext': Path(a.wikitext).read_text(encoding='utf-8', errors='ignore')}
    rows = []
    for cname, text in corpora.items():
        ids = np.array(tok(text).input_ids, dtype=np.int64); print(cname, 'tokens', len(ids), flush=True)
        for L in LS:
            for seed in SEEDS:
                starts = np.random.default_rng(seed + L).integers(0, len(ids) - L - 1, N_SEQ)
                for i, s in enumerate(starts):
                    seq = ids[s:s + L]; r = {'corpus': cname, 'L': L, 'seed': seed, 'seq': i}
                    for vname, plan in VARIANTS.items():
                        r[vname], r['n'] = engine.nll_full(model, seq, dev) if plan is None else engine.nll_shards(model, seq, plan, dev)
                    rows.append(r)
                print(cname, L, seed, 'done', flush=True)
    RES.mkdir(exist_ok=True); (RES / 'quality_rows.json').write_text(json.dumps(rows)); analyze(rows)


def analyze(rows=None):
    rows = rows or json.loads((RES / 'quality_rows.json').read_text()); out = {'ppl': {}, 'ratio': {}}
    for c in ('tinystories', 'wikitext'):
        for L in LS:
            for seed in SEEDS:
                r = [x for x in rows if x['corpus'] == c and x['L'] == L and x['seed'] == seed]; n = sum(x['n'] for x in r)
                ppl = {v: math.exp(sum(x[v] for x in r) / n) for v in VARIANTS}; out['ppl'][f'{c}|{L}|{seed}'] = ppl
                out['ratio'][f'{c}|{L}|{seed}'] = {v: ppl[v] / ppl['full'] for v in VARIANTS if v != 'full'}
    passing = [v for v in VARIANTS if v != 'full' and all(out['ratio'][f'{c}|4096|{s}'][v] <= 1.05 for c in ('tinystories', 'wikitext') for s in SEEDS)]
    out['Q1_passing_variants'] = passing; out['Q1'] = bool(passing)
    best = min((v for v in VARIANTS if v != 'full'), key=lambda v: np.mean([out['ratio'][f'{c}|4096|{s}'][v] for c in ('tinystories', 'wikitext') for s in SEEDS]))
    out['best_variant_L4096'] = best
    (RES / 'quality_analysis.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'Q1': out['Q1'], 'passing': passing, 'best': best}, indent=1))
    for k in sorted(out['ratio']):
        if '|4096|' in k: print(k, {v: round(x, 3) for v, x in out['ratio'][k].items()})


if __name__ == '__main__':
    main()
