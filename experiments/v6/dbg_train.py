import sys, time, torch, torch.nn.functional as F
from v6 import MDLM, load_cache
dev = torch.device(sys.argv[1]); d, layers, bf16, clip, steps = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == '1', sys.argv[5] == '1', int(sys.argv[6])
tr, _ = load_cache('cache.pt'); torch.manual_seed(0)
m = MDLM(d=d, layers=layers, heads=max(4, d // 32), ff=4 * d).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
t0 = time.time(); out = []
for s in range(steps):
    st = torch.randint(0, len(tr) - 257, (32,)); y = torch.stack([tr[i:i + 256] for i in st.tolist()])
    msk = torch.rand(y.shape) < 0.15; x = y.masked_fill(msk, 1); x, y, msk = x.to(dev), y.to(dev), msk.to(dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
        z = m(x, torch.arange(256, device=dev).expand(32, -1))
    loss = F.cross_entropy(z.float()[msk], y[msk]); opt.zero_grad(); loss.backward()
    if clip: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if s % 500 == 0 or s == steps - 1: out.append('%d:%.2f' % (s, float(loss)))
print(f'dev={sys.argv[1]} d={d} L={layers} bf16={bf16} clip={clip} ->', ' '.join(out), '(%.0fs)' % (time.time() - t0), flush=True)
