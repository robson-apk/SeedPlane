"""Does the original SeedPlane toy checkpoint use context? Masked-token loss by mask rate vs unigram entropy."""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / 'seedplane'))
import clmp_parity_seed_v3 as v
m = v.ParityDenoiser('clmp'); m.load_state_dict(torch.load(REPO / 'checkpoints/clmp_parity_ctx1024.pt', map_location='cpu', weights_only=True)['state_dict']); m.eval()
p = torch.bincount(v.train_ids, minlength=1024).float(); p /= p.sum(); out = {'unigram_entropy': float(-(p[p > 0] * p[p > 0].log()).sum()), 'by_rate': {}}
g = torch.Generator().manual_seed(5); st = torch.randint(0, len(v.val_ids) - 161, (64,), generator=g)
y = torch.stack([v.val_ids[s:s + 160] for s in st.tolist()])  # one shard + halos, center head
with torch.no_grad():
    for r in [0.15, 0.5, 0.9]:
        msk = torch.rand(y.shape, generator=g) < r; x = y.masked_fill(msk, 1)
        z = m.logits_from_repr(m.hidden(x, torch.arange(160).expand(64, -1), 'none'))
        out['by_rate'][r] = {'loss': float(F.cross_entropy(z[msk], y[msk])), 'acc': float((z[msk].argmax(-1) == y[msk]).float().mean()), 'n': int(msk.sum())}
(Path(__file__).parent / 'results' / 'original_checkpoint_context_check.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
