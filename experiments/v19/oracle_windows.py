"""V19 oracle: SeedPlane window semantics recomputed from scratch per position (CPU FP32 Qwen2Engine).

Position i reads exactly the tokens of its ShardPlan window up to i, with original position ids, from an empty cache —
the same definition as engine.window_logits. Greedy tokens and logit dumps are written next to this script's outputs.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seedplane.engine import ShardPlan
from seedplane.qwen_engine import Qwen2Engine


def main():
    p = argparse.ArgumentParser(); p.add_argument('hf_dir'); p.add_argument('out_dir')
    p.add_argument('--prompt', default='9707,11,1879,0'); p.add_argument('-n', type=int, default=296)
    p.add_argument('--shard', type=int, default=64); p.add_argument('--halo', type=int, default=32); p.add_argument('--sinks', type=int, default=4)
    p.add_argument('--dump-at', default='3,63,64,65,128,200,299'); a = p.parse_args()
    torch.set_num_threads(6); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    seq = [int(t) for t in a.prompt.split(',')]; P = len(seq); total = P + a.n; dump = {int(x) for x in a.dump_at.split(',')}
    plan = ShardPlan(a.shard, a.halo, a.sinks); visible = {}
    for c0, c1, idx in plan.windows(total):
        for i in range(c0, c1): visible[i] = [int(j) for j in idx[:len(idx) - (c1 - 1 - i)]]
    e = Qwen2Engine(a.hf_dir, 'cpu', torch.float32)
    for i in range(total):
        idx = visible[i]; assert idx[-1] == i
        logits, _ = e.forward([seq[j] for j in idx], e.new_cache(len(idx)), positions=idx); z = logits[-1]
        if i in dump: z.numpy().astype(np.float32).tofile(out / f'logits_{i}.f32')
        if i + 1 >= P and len(seq) < total: seq.append(int(z.argmax()))
    json.dump({'plan': vars(plan), 'prompt': seq[:P], 'tokens': seq[P:], 'window_sizes': {str(i): len(visible[i]) for i in sorted(dump)}},
              open(out / 'oracle.json', 'w'), indent=1)
    print('oracle done', seq[P:P + 8])


if __name__ == '__main__': main()
