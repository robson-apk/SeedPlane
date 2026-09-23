import sys, numpy as np, torch, torch.nn.functional as F
from v7 import *
dev = device(); ck = torch.load(ROOT / 'v7_model_a.pt', map_location='cpu'); A = ModelA(**ck['cfg']).to(dev); A.load_state_dict(ck['state_dict']); A.eval()
y, m, meta = make_batch(np.random.default_rng(999), 32); yt = torch.from_numpy(y).to(dev); mt = torch.from_numpy(m).to(dev); qm = qmask_of(meta, 32, dev)
with torch.no_grad(): lg = A(yt.masked_fill(mt, MASK), torch.arange(L, device=dev).expand(32, -1)).float()
print('filler CE %.3f  query CE %.3f (chance 4.16)' % (F.cross_entropy(lg[mt & ~qm], yt[mt & ~qm]), F.cross_entropy(lg[qm], yt[qm])))
pred = lg.argmax(-1).cpu().numpy(); acc = {}
for b, qs in enumerate(meta):
    for vp, dp in qs: d = abs(vp // SHARD - dp // SHARD); acc.setdefault(min(d, 4), []).append(pred[b, vp] == y[b, vp])
print({k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(acc.items())})
