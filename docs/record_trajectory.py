"""Record a REAL decoding trajectory (which token is committed at which step, and whether it is right) for the GIFs.
Both decoders use the shipped V6 checkpoint and the exact V8/V9 commit rule.

python docs/record_trajectory.py     (from the repo root; needs data/tinystories_word1024_cache.pt)
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'experiments/v9')); sys.path.insert(0, str(REPO / 'experiments/v8')); sys.path.insert(0, str(REPO / 'experiments/v6'))
import v9, v8
from v6 import MASK, SHARD

torch.set_num_threads(1)
L = 1024; HALO = 16
model = v8._load(REPO / 'checkpoints/v6_mdlm_d256_l6_seed1.pt')
cache = torch.load(REPO / 'data/tinystories_word1024_cache.pt', map_location='cpu'); vocab = cache['vocab']; val = cache['val_ids']
rng = np.random.default_rng(2026); s = int(rng.integers(0, len(val) - L - 1)); y = val[s:s + L].numpy()
m = rng.random(L) < 0.5; x0 = np.where(m, MASK, y)


def run(kind):
    x = x0.copy(); step = np.full(L, -1)
    with torch.inference_mode():
        for t in range(v9.K):
            pred = np.zeros(L, np.int64); conf = np.full(L, -1.0, np.float32); idx = np.nonzero(x == MASK)[0]
            if kind == 'trad':
                h = v9.hidden(model, torch.from_numpy(x)[None], torch.arange(L)[None])[0]
                p, c = v9.decide(v9.logits_at(model, h[idx])); pred[idx] = p; conf[idx] = c
            else:
                for c0 in range(0, L, SHARD):
                    q0, q1 = max(0, c0 - HALO), min(L, c0 + SHARD + HALO); loc = np.nonzero(x[c0:c0 + SHARD] == MASK)[0]
                    if not len(loc): continue
                    h = v9.hidden(model, torch.from_numpy(x[q0:q1])[None], torch.arange(q0, q1)[None])[0]
                    p, c = v9.decide(v9.logits_at(model, h[c0 - q0 + loc])); pred[c0 + loc] = p; conf[c0 + loc] = c
            before = x.copy(); v9.commit(x, pred, conf, t, L); step[(before == MASK) & (x != MASK)] = t
    return x, step


out = {'L': L, 'shard': SHARD, 'steps': v9.K, 'masked': m.tolist(), 'truth': [vocab[i] for i in y]}
for kind in ('trad', 'sp'):
    xf, st = run(kind); out[kind] = {'commit_step': st.tolist(), 'correct': (xf == y).tolist(), 'final': [vocab[i] for i in xf],
                                      'accuracy_masked': float((xf == y)[m].mean())}
    print(kind, 'accuracy on masked', round(out[kind]['accuracy_masked'], 4))
(REPO / 'docs' / 'trajectory.json').write_text(json.dumps(out))
