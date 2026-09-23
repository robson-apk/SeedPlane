"""Exploratory microbenchmark (1 thread): how to batch shards without losing the fast path."""
import sys, time, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT.parent / 'v8')); sys.path.insert(0, str(ROOT.parent / 'v6'))
import v8, v9
from v6 import MASK
torch.set_num_threads(1); m = v8._load(sys.argv[1]); L = 512; x = torch.randint(4, 1024, (L,)); x[::2] = MASK
wins = [(max(0, c0 - 16), min(L, c0 + 144)) for c0 in range(0, L, 128)]
def t(fn, n=15):
    fn(); ts = []
    for _ in range(n): a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    return round(1000 * float(np.median(ts)), 2)
with torch.inference_mode():
    sep = lambda: [m(x[q0:q1][None], torch.arange(q0, q1)[None]) for q0, q1 in wins]
    def padded():
        W = 160; xb = torch.full((4, W), MASK); pb = torch.zeros((4, W), dtype=torch.long); pad = torch.ones((4, W), dtype=torch.bool)
        for i, (q0, q1) in enumerate(wins): xb[i, :q1 - q0] = x[q0:q1]; pb[i, :q1 - q0] = torch.arange(q0, q1); pad[i, :q1 - q0] = False
        return v9.hidden(m, xb, pb, pad)
    def by_len():
        groups = {}
        for q0, q1 in wins: groups.setdefault(q1 - q0, []).append((q0, q1))
        return [v9.hidden(m, torch.stack([x[a:b] for a, b in g]), torch.stack([torch.arange(a, b) for a, b in g])) for g in groups.values()]
    hid_only = lambda: [v9.hidden(m, x[q0:q1][None], torch.arange(q0, q1)[None]) for q0, q1 in wins]
    print(json.dumps({'separate_full_logits_ms': t(sep), 'separate_hidden_only_ms': t(hid_only), 'batched_padded_mask_ms': t(padded), 'batched_by_equal_length_ms': t(by_len)}))
