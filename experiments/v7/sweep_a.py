"""Short sweep (addendum 2 diagnosis): which optimizer config learns retrieval? Lc=256, 2000 steps, B=32."""
import sys, json, math, time, numpy as np, torch, torch.nn.functional as F
from v7 import *
dev = device(); lr, clip, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); Lc = 256; Bb = 32; steps = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
torch.manual_seed(seed); rng = np.random.default_rng(seed); model = ModelA(**cfg_for(False)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps)))
t0 = time.time()
for step in range(steps):
    y, m, meta = make_batch(rng, Bb, Lc); y = torch.from_numpy(y).to(dev); m = torch.from_numpy(m).to(dev); qm = qmask_of(meta, Bb, dev, Lc)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'xpu'):
        lg = model(y.masked_fill(m, MASK), torch.arange(Lc, device=dev).expand(Bb, -1)).float()
    loss = masked_loss(lg, y, m, qm); opt.zero_grad(set_to_none=True); loss.backward()
    if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step(); sched.step()
model.eval(); y, m, meta = make_batch(np.random.default_rng(999), 64, Lc); yt = torch.from_numpy(y).to(dev); mt = torch.from_numpy(m).to(dev); qm = qmask_of(meta, 64, dev, Lc)
with torch.no_grad(): lg = model(yt.masked_fill(mt, MASK), torch.arange(Lc, device=dev).expand(64, -1)).float()
acc = float((lg.argmax(-1)[qm] == yt[qm]).float().mean())
print(json.dumps({'lr': lr, 'clip': clip, 'seed': seed, 'filler_ce': float(F.cross_entropy(lg[mt & ~qm], yt[mt & ~qm])), 'query_ce': float(F.cross_entropy(lg[qm], yt[qm])), 'query_acc': acc, 's': round(time.time() - t0)}), flush=True)
