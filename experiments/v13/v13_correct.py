"""V13 C1/C2: native seedplane-worker (llama.cpp) NLL vs PyTorch engine NLL. See PROTOCOL.md.

python v13_correct.py --bundle <sp bundle> --text <txt> --worker HOST:PORT [--L 1024] [--plan 512,256,0] [--native-ref <file>]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parents[1]))
from seedplane import engine, llama_backend as lb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle'); ap.add_argument('--text'); ap.add_argument('--worker'); ap.add_argument('--L', type=int, default=1024)
    ap.add_argument('--plan', default='512,256,0'); ap.add_argument('--chunks', type=int, default=2); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(); S, H, K = map(int, a.plan.split(',')); torch.set_num_threads(a.threads)
    model, tok = engine.load_model(a.bundle, 'cpu')
    allids = np.array(tok(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)
    w = lb.connect([a.worker])[0]; rows = []
    for c in range(a.chunks):
        ids = allids[c * a.L:(c + 1) * a.L]
        one = [(0, a.L, np.arange(a.L))]
        (nat_full, n_full, _, _), = lb.run_windows([w], ids, one, 'nll')[0]
        nat_sh = lb.run_windows([w], ids, engine.ShardPlan(S, H, K).windows(a.L), 'nll')[0]
        nat_sh_nll = sum(r[0] for r in nat_sh); nat_sh_n = sum(r[1] for r in nat_sh)
        pt_full, pn_full = engine.nll_full(model, ids, 'cpu'); pt_sh, pn_sh = engine.nll_shards(model, ids, engine.ShardPlan(S, H, K), 'cpu')
        rows.append({'chunk': c, 'n_scored': [n_full, nat_sh_n, pn_full, pn_sh],
                     'native_full': nat_full / n_full, 'torch_full': pt_full / pn_full, 'native_shard': nat_sh_nll / nat_sh_n, 'torch_shard': pt_sh / pn_sh})
        print(rows[-1], flush=True)
    w.close()
    rel = lambda x, y: abs(x - y) / y
    out = {'L': a.L, 'plan': [S, H, K], 'rows': rows,
           'C1_native_vs_torch_max_rel': max(rel(r['native_full'], r['torch_full']) for r in rows),
           'C2_native_vs_torch_shard_max_rel': max(rel(r['native_shard'], r['torch_shard']) for r in rows)}
    out['C1_pass_1pct'] = out['C1_native_vs_torch_max_rel'] <= 0.01; out['C2_pass_1pct'] = out['C2_native_vs_torch_shard_max_rel'] <= 0.01
    (ROOT / 'results').mkdir(exist_ok=True); (ROOT / 'results' / f'correct_{Path(a.worker.replace(":", "_")).name}.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
