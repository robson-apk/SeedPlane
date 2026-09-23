"""SeedPlane V6: iterative masked diffusion, global vs neighbor-halo shards vs isolated shards. See PROTOCOL.md.

python v6.py train   --cache <cache.pt> [--steps N] [--small]
python v6.py eval    --cache <cache.pt>
python v6.py analyze
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
V = 1024; MASK = 1; SHARD = 128; MAXPOS = 1024
RARE_FROM = 54  # ids 0-3 special, 4-53 = 50 most frequent words (vocab sorted by frequency)
SEEDS = (11, 23, 37); LS = (256, 512, 1024); RATES = (0.5, 0.9); KS = (1, 16)
DECODERS = {'global': None, 'halo16': 16, 'halo64': 64, 'isolated': 0}


def device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device('xpu')
    return torch.device('cpu')


class MDLM(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, ff=1024):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(MAXPOS, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.bias = nn.Parameter(torch.zeros(V))
        self.scale = 1.0  # V6 run 1 used 1/sqrt(d) with std-1 embeddings and collapsed to unigram; see PROTOCOL addendum
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x, pos):
        h = self.tr(self.emb(x) + self.pos(pos))
        return F.linear(self.norm(h), self.emb.weight) * self.scale + self.bias


def load_cache(path):
    c = torch.load(path, map_location='cpu')
    return c['train_ids'], c['val_ids']


# ---------------------------------------------------------------- train
def train(args):
    dev = device(); torch.manual_seed(args.seed)
    train_ids, _ = load_cache(args.cache)
    cfg = dict(d=64, layers=2, heads=4, ff=128) if args.small else dict(d=256, layers=6, heads=8, ff=1024)
    model = MDLM(**cfg).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warm = max(1, min(500, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    log = []; t0 = time.time()
    print(f'device={dev} params={nparams} cfg={cfg}', flush=True)
    for step in range(args.steps):
        # Effective batch 8 full sequences / 48 windows via 2 micro-batches (only ~3 GB VRAM available on the B580).
        opt.zero_grad(set_to_none=True); tot = 0.0
        H = [16, 64][(step // 2) % 2]; k = int(torch.randint(0, 8, (1,)))
        for _ in range(2):
            if step % 2 == 0:  # full sequences
                B, L = 4, 1024
                st = torch.randint(0, len(train_ids) - L - 1, (B,))
                y = torch.stack([train_ids[s:s + L] for s in st.tolist()])
                pos = torch.arange(L).expand(B, -1)
            else:  # local windows at absolute positions (same shard index/halo within batch)
                B = 24
                c0, c1 = k * SHARD, (k + 1) * SHARD; q0, q1 = max(0, c0 - H), min(MAXPOS, c1 + H)
                st = torch.randint(0, len(train_ids) - (q1 - q0) - 1, (B,))
                y = torch.stack([train_ids[s:s + q1 - q0] for s in st.tolist()])
                pos = torch.arange(q0, q1).expand(B, -1)
            hi = 0.3 if step < 3000 else 1.0  # mask-rate curriculum (PROTOCOL addendum 4)
            rate = 0.05 + (hi - 0.05) * torch.rand(B, 1)
            m = torch.rand(y.shape) < rate
            x = y.masked_fill(m, MASK)
            x, y, m, pos = x.to(dev), y.to(dev), m.to(dev), pos.to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'xpu'):
                logits = model(x, pos)
            loss = F.cross_entropy(logits.float()[m], y[m]) / 2
            loss.backward(); tot += float(loss)
        opt.step(); sched.step()  # no grad clipping: clip=1.0 kept the model on the unigram plateau (PROTOCOL addendum 3)
        loss = torch.tensor(tot)
        if step < 10 or step % 250 == 0 or step == args.steps - 1:
            log.append({'step': step, 'kind': 'full' if step % 2 == 0 else 'window', 'loss': float(loss), 'elapsed_s': time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
    RESULTS.mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg, 'steps': args.steps, 'seed': args.seed}, ROOT / 'v6_model.pt')
    (RESULTS / 'train_log.json').write_text(json.dumps({'params': nparams, 'cfg': cfg, 'steps': args.steps, 'device': str(dev), 'log': log}, indent=1))


# ---------------------------------------------------------------- eval
def shards(L):
    return [(c0, min(L, c0 + SHARD)) for c0 in range(0, L, SHARD)]


@torch.no_grad()
def view_logits(model, x, H, dev):
    """Logits for every position under the decoder's visibility. H=None -> global."""
    B, L = x.shape
    if H is None:
        return torch.cat([model(x[i:i + 8], torch.arange(L, device=dev).expand(len(x[i:i + 8]), -1)) for i in range(0, B, 8)])
    out = []
    for c0, c1 in shards(L):
        q0, q1 = max(0, c0 - H), min(L, c1 + H)
        z = model(x[:, q0:q1], torch.arange(q0, q1, device=dev).expand(B, -1))
        out.append(z[:, c0 - q0:c1 - q0])
    return torch.cat(out, 1)


@torch.no_grad()
def decode(model, x0, H, K, dev):
    x = x0.clone(); B, L = x.shape
    commit_step = torch.full((B, L), -1, dtype=torch.long, device=dev)
    for t in range(K):
        probs = view_logits(model, x, H, dev).float().softmax(-1)
        probs[..., MASK] = 0  # never predict the mask token itself
        conf, pred = probs.max(-1)
        masked = x == MASK
        conf = conf.masked_fill(~masked, -1.0)
        for c0, c1 in shards(L):  # per-shard quota, identical rule for all decoders
            mc = masked[:, c0:c1]; rem = mc.sum(1)
            n = torch.ceil(rem.float() / (K - t)).long()
            rank = conf[:, c0:c1].argsort(1, descending=True).argsort(1)
            take = (rank < n[:, None]) & mc
            seg = x[:, c0:c1]; seg[take] = pred[:, c0:c1][take]
            commit_step[:, c0:c1][take] = t
    assert not (x == MASK).any()
    return x, commit_step


def subsets(y, m0, L, dev):
    B = y.shape[0]; posn = torch.arange(L, device=dev)
    bnd = torch.zeros(L, dtype=torch.bool, device=dev)
    for b in range(SHARD, L, SHARD): bnd[b - 16:b + 16] = True
    boundary = m0 & bnd; interior = m0 & ~bnd
    vis = ~m0; lr = torch.zeros_like(m0)
    for c0, c1 in shards(L):
        w0, w1 = max(0, c0 - 16), min(L, c1 + 16)
        yc = y[:, c0:c1]
        eq = (yc[:, :, None] == y[:, None, :]) & vis[:, None, :]       # B,128,L
        inwin = ((posn >= w0) & (posn < w1))[None, None, :]
        lr[:, c0:c1] = m0[:, c0:c1] & (yc >= RARE_FROM) & (eq & ~inwin).any(-1) & ~(eq & inwin).any(-1)
    return {'all': m0, 'boundary': boundary, 'interior': interior, 'LR': lr}


def evaluate(args):
    dev = device()
    ck = torch.load(ROOT / 'v6_model.pt', map_location='cpu')
    model = MDLM(**ck['cfg']).to(dev); model.load_state_dict(ck['state_dict']); model.eval()
    _, val_ids = load_cache(args.cache)
    n_seq = args.n_seq; rows = []
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        st = torch.randint(0, len(val_ids) - 1025, (n_seq,), generator=g)
        full = torch.stack([val_ids[s:s + 1024] for s in st.tolist()])
        for r in RATES:
            mfull = torch.rand(full.shape, generator=g) < r
            for L in LS:
                y = full[:, :L].to(dev); m0 = mfull[:, :L].to(dev); x0 = y.masked_fill(m0, MASK)
                sub = subsets(y, m0, L, dev)
                for K in KS:
                    for name, H in DECODERS.items():
                        xf, cs = decode(model, x0, H, K, dev)
                        ok = (xf == y)
                        row = {'seed': seed, 'rate': r, 'L': L, 'S': L // SHARD, 'K': K, 'decoder': name}
                        for s, msk in sub.items():
                            row[s] = {'correct': (ok & msk).sum(1).tolist(), 'n': msk.sum(1).tolist()}
                        if H:  # exchange exercised: initially-masked halo tokens filled before the last step
                            hal = torch.zeros(L, dtype=torch.bool, device=dev)
                            for c0, c1 in shards(L):
                                if c0 > 0: hal[c0:min(L, c0 + H)] = True
                                if c1 < L: hal[max(0, c1 - H):c1] = True
                            hm = m0 & hal
                            row['exchange'] = {'halo_masked': int(hm.sum()), 'filled_before_last_step': int((hm & (cs < K - 1)).sum())}
                        rows.append(row)
                        print(seed, r, L, K, name, 'acc_all=%.4f' % (sum(row['all']['correct']) / max(1, sum(row['all']['n']))), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'eval_rows.json').write_text(json.dumps({'n_seq': n_seq, 'model_steps': ck['steps'], 'cfg': ck['cfg'], 'rows': rows}))


# ---------------------------------------------------------------- analyze
def analyze(_args):
    d = json.loads((RESULTS / 'eval_rows.json').read_text()); rows = d['rows']
    def get(seed, r, L, K, dec):
        return next(x for x in rows if x['seed'] == seed and x['rate'] == r and x['L'] == L and x['K'] == K and x['decoder'] == dec)
    def arr(row, s): return np.array(row[s]['correct'], float), np.array(row[s]['n'], float)
    def acc(c, n, ix=None):
        if ix is None: return c.sum() / max(1, n.sum())
        return c[ix].sum(1) / np.maximum(1, n[ix].sum(1))
    def diff(seed, r, L, K, a, b, s, boot=2000):
        (ca, na), (cb, nb) = arr(get(seed, r, L, K, a), s), arr(get(seed, r, L, K, b), s)
        ix = np.random.default_rng(seed).integers(len(ca), size=(boot, len(ca)))
        dd = acc(ca, na, ix) - acc(cb, nb, ix)
        return {'diff_pp': 100 * (acc(ca, na) - acc(cb, nb)), 'ci95_pp': (100 * np.quantile(dd, [.025, .975])).tolist(), 'n': int(na.sum())}
    def recovery(seed, r, L, K, s, boot=2000):
        g_, h_, i_ = (arr(get(seed, r, L, K, k), s) for k in ('global', 'halo16', 'isolated'))
        ix = np.random.default_rng(seed).integers(len(g_[0]), size=(boot, len(g_[0])))
        def R(ix_=None):
            G, Hh, I = acc(*g_, ix_), acc(*h_, ix_), acc(*i_, ix_)
            return (Hh - I) / np.where(np.abs(G - I) < 1e-12, np.nan, G - I)
        return {'R': float(R()), 'ci95': np.nanquantile(R(ix), [.025, .975]).tolist()}
    out = {'table': [], 'criteria': {}}
    for seed in SEEDS:
        for r in RATES:
            for L in LS:
                for K in KS:
                    t = {'seed': seed, 'rate': r, 'L': L, 'K': K}
                    for dec in DECODERS:
                        row = get(seed, r, L, K, dec)
                        t[dec] = {s: round(100 * acc(*arr(row, s)), 2) for s in ('all', 'boundary', 'interior', 'LR')}
                        if 'exchange' in row: t[dec]['exchange'] = row['exchange']
                    t['n'] = {s: int(sum(get(seed, r, L, K, 'global')[s]['n'])) for s in ('all', 'boundary', 'interior', 'LR')}
                    out['table'].append(t)
    r, L, K = 0.5, 1024, 16; c = {}
    c0 = {s: diff(s, r, L, K, 'global', 'isolated', 'LR') for s in SEEDS}
    c['C0'] = {'per_seed': c0, 'pass': all(v['diff_pp'] >= 2 and v['ci95_pp'][0] > 0 for v in c0.values())}
    h1a = {s: diff(s, r, L, K, 'global', 'halo16', 'all') for s in SEEDS}
    h1b = {s: {'S8': diff(s, r, 1024, K, 'global', 'halo16', 'boundary'), 'S2': diff(s, r, 256, K, 'global', 'halo16', 'boundary')} for s in SEEDS}
    h1c = {s: {'K16': diff(s, r, L, 16, 'global', 'halo16', 'all'), 'K1': diff(s, r, L, 1, 'global', 'halo16', 'all')} for s in SEEDS}
    c['H1a'] = {'per_seed': h1a, 'pass': all(v['ci95_pp'][1] <= 1 for v in h1a.values())}
    c['H1b'] = {'per_seed': h1b, 'pass': all(v['S8']['diff_pp'] - v['S2']['diff_pp'] <= 1 for v in h1b.values())}
    c['H1c'] = {'per_seed': h1c, 'pass': all(v['K16']['diff_pp'] - v['K1']['diff_pp'] <= 0.5 for v in h1c.values())}
    h2 = {s: recovery(s, r, L, K, 'LR') for s in SEEDS}
    c['H2'] = {'per_seed': h2, 'pass': all(v['R'] >= 0.5 for v in h2.values())}
    h1 = c['H1a']['pass'] and c['H1b']['pass'] and c['H1c']['pass']
    if not c['C0']['pass']: verdict = 'INCONCLUSIVE_LONG_CONTEXT_NOT_USED'
    elif h1 and c['H2']['pass']: verdict = 'CANDIDATE_CONTRIBUTION_NEEDS_SCALE_REPLICATION'
    elif h1: verdict = 'LOCAL_ONLY_BEHAVES_AS_LOCAL_ATTENTION'
    else: verdict = 'SEAM_ERROR_NOT_BOUNDED'
    out['criteria'] = c; out['verdict'] = verdict
    (RESULTS / 'analysis.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'verdict': verdict, 'criteria': c}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['train', 'eval', 'analyze'])
    ap.add_argument('--cache'); ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--small', action='store_true'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--n_seq', type=int, default=64)
    a = ap.parse_args(); {'train': train, 'eval': evaluate, 'analyze': analyze}[a.mode](a)
