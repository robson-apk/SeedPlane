"""Model sanity (not a pre-registered criterion): global-context masked loss by mask rate vs unigram entropy."""
import sys, json, torch, torch.nn.functional as F
from v6 import MDLM, load_cache, MASK, ROOT, RESULTS, device
dev = device(); ck = torch.load(ROOT / 'v6_model.pt', map_location='cpu')
model = MDLM(**ck['cfg']).to(dev); model.load_state_dict(ck['state_dict']); model.eval()
tr, val = load_cache(sys.argv[1])
p = torch.bincount(tr, minlength=1024).float(); p /= p.sum(); uni = float(-(p[p > 0] * p[p > 0].log()).sum())
g = torch.Generator().manual_seed(999); st = torch.randint(0, len(val) - 1025, (32,), generator=g)
y = torch.stack([val[s:s + 1024] for s in st.tolist()]).to(dev); out = {'unigram_entropy': uni, 'by_rate': {}}
with torch.no_grad():
    for r in [0.15, 0.3, 0.5, 0.7, 0.9, 1.0]:
        m = (torch.rand(y.shape, generator=g) < r).to(dev); x = y.masked_fill(m, MASK)
        z = torch.cat([model(x[i:i + 8], torch.arange(1024, device=dev).expand(8, -1)) for i in range(0, 32, 8)]).float()
        out['by_rate'][r] = {'loss': float(F.cross_entropy(z[m], y[m])), 'acc': float((z[m].argmax(-1) == y[m]).float().mean())}
print(json.dumps(out, indent=1)); (RESULTS / 'model_sanity.json').write_text(json.dumps(out, indent=1))
