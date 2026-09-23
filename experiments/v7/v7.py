"""SeedPlane V7: synthetic long-range key-value recall across shards. See PROTOCOL.md (criteria fixed before running).

python v7.py train_a [--steps N] [--small]
python v7.py train_b [--steps N] [--small]
python v7.py eval    [--n_seq N]
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'
V = 1024; PAD, MASK, DEF, QRY = 0, 1, 2, 3
K0, NK, V0, NV, F0 = 10, 64, 100, 64, 200
L = 1024; SHARD = 128; NS = L // SHARD; HALO = 16; NPAIR = 8; FILL_MASK = 0.15
import os
POS = os.environ.get('V7_POS', 'learned')
EVAL_SEEDS = (101, 202, 303); BUCKETS = {'d0': (0, 0), 'd1': (1, 1), 'd2-3': (2, 3), 'd4-7': (4, 7)}


def device():
    return torch.device('xpu') if hasattr(torch, 'xpu') and torch.xpu.is_available() else torch.device('cpu')


# ---------------------------------------------------------------- task
_rng0 = np.random.default_rng(0)
NF = V - F0
TRANS = _rng0.integers(0, NF, (NF, 4))                      # sparse Markov chain over filler tokens
TPROB = _rng0.dirichlet(np.ones(4) * 0.5, NF)


def make_batch(rng, B, Lc=L):
    """Returns tokens y (B,Lc), eval mask m (B,Lc) and query metadata list per sequence."""
    npair = max(4, NPAIR * Lc // L)
    y = np.empty((B, Lc), np.int64); m = np.zeros((B, Lc), bool); meta = []
    for b in range(B):
        f = np.empty(Lc, np.int64); f[0] = rng.integers(NF); u = rng.random(Lc)
        cum = np.cumsum(TPROB, 1)
        for i in range(1, Lc): f[i] = TRANS[f[i - 1], np.searchsorted(cum[f[i - 1]], u[i])]
        s = f + F0
        slots = rng.choice(Lc // 4, 3 * npair, replace=False) * 4
        keys = rng.choice(NK, npair, replace=False) + K0; vals = rng.integers(0, NV, npair) + V0
        q = []
        for j in range(npair):
            dp = slots[j]; s[dp:dp + 3] = (DEF, keys[j], vals[j])
            for qp in (slots[npair + 2 * j], slots[npair + 2 * j + 1]):
                s[qp:qp + 3] = (QRY, keys[j], vals[j]); q.append((qp + 2, dp + 2))
        y[b] = s
        fm = (rng.random(Lc) < FILL_MASK) & (s >= F0); m[b] = fm
        for vp, _ in q: m[b, vp] = True
        meta.append(q)
    return y, m, meta


# ---------------------------------------------------------------- models
class Enc(nn.Module):
    def __init__(self, d=192, layers=4, heads=6, ff=768):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        if POS == 'sin':  # fixed sinusoidal absolute positions (addendum 2)
            pe = torch.zeros(L, d); pp = torch.arange(L)[:, None].float(); div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pp * div); pe[:, 1::2] = torch.cos(pp * div); self.register_buffer('pe', pe * 0.5)
            self.pos = lambda idx: self.pe[idx]
        else:
            self.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(V)); self.d = d
        self.shl = nn.Linear(d, d, bias=False); self.shr = nn.Linear(d, d, bias=False)  # token shift (addendum 2)
        nn.init.normal_(self.emb.weight, std=0.02)
        if POS != 'sin': nn.init.normal_(self.pos.weight, std=0.02)

    def logits(self, h): return F.linear(self.norm(h), self.emb.weight) + self.bias

    def embed(self, x, pos):
        e = self.emb(x); z = torch.zeros_like(e[:, :1])
        return e + self.shl(torch.cat([z, e[:, :-1]], 1)) + self.shr(torch.cat([e[:, 1:], z], 1)) + self.pos(pos)


class ModelA(Enc):
    def forward(self, x, pos): return self.logits(self.tr(self.embed(x, pos)))


class ModelB(Enc):
    """Per-shard encoder; exchanges one latent vector per direction with immediate neighbors per round."""
    def __init__(self, **kw):
        super().__init__(**kw); d = self.d
        self.in_l = nn.Linear(d, d); self.in_r = nn.Linear(d, d)
        self.slot = nn.Parameter(torch.randn(4, d) * 0.02)  # [inL, inR, outL, outR] type embeddings

    def round(self, x, pos, mL, mR):
        tok = self.embed(x, pos)                                            # (N,S,d); shift stays inside the shard
        head = torch.stack([self.in_l(mL) + self.slot[0], self.in_r(mR) + self.slot[1]], 1)
        tail = self.slot[2:].expand(x.shape[0], -1, -1)
        h = self.tr(torch.cat([head, tok, tail], 1))
        return self.logits(h[:, 2:-2]), h[:, -2], h[:, -1]                 # logits, msg to left, msg to right

    def forward(self, y_masked, R, all_rounds=False, off=0):
        B = y_masked.shape[0]; d = self.d; NS = y_masked.shape[1] // SHARD; Lc = NS * SHARD
        x = y_masked.view(B, NS, SHARD).reshape(B * NS, SHARD)
        pos = (off + torch.arange(NS, device=x.device)[:, None] * SHARD + torch.arange(SHARD, device=x.device)).repeat(B, 1)
        mL = torch.zeros(B * NS, d, device=x.device); mR = torch.zeros_like(mL); outs = []
        for _ in range(R):
            lg, oL, oR = self.round(x, pos, mL, mR)
            oL = oL.view(B, NS, d); oR = oR.view(B, NS, d); z = torch.zeros(B, 1, d, device=x.device)
            mL = torch.cat([z, oR[:, :-1]], 1).reshape(B * NS, d)            # shard s receives right-msg of s-1
            mR = torch.cat([oL[:, 1:], z], 1).reshape(B * NS, d)             # shard s receives left-msg of s+1
            outs.append(lg.view(B, Lc, V))
        return outs if all_rounds else outs[-1]


def cfg_for(small): return dict(d=48, layers=2, heads=4, ff=96) if small else dict(d=192, layers=4, heads=6, ff=768)


def masked_loss(logits, y, m, qmask):
    lf = F.cross_entropy(logits[m & ~qmask], y[m & ~qmask]) if (m & ~qmask).any() else 0.0
    return lf + F.cross_entropy(logits[qmask], y[qmask])       # query values weighted as much as all filler


def qmask_of(meta, B, dev, Lc=L):
    q = torch.zeros(B, Lc, dtype=torch.bool)
    for b, qs in enumerate(meta):
        for vp, _ in qs: q[b, vp] = True
    return q.to(dev)


# ---------------------------------------------------------------- training
def train(args, which):
    dev = device(); torch.manual_seed(1); rng = np.random.default_rng(7 if which == 'a' else 8)
    model = (ModelA if which == 'a' else ModelB)(**cfg_for(args.small)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warm = max(1, min(500, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    log = []; t0 = time.time()
    for step in range(args.steps):
        Lc = 128 if step < 1500 else 256 if step < 3000 else 512 if step < 4500 else L  # length curriculum (addendum 1)
        B = 8 * L // Lc  # constant tokens per step (addendum 2)
        off = int(np.random.default_rng(step).integers(0, (L - Lc) // SHARD + 1)) * SHARD
        y, m, meta = make_batch(rng, B, Lc); y = torch.from_numpy(y).to(dev); m = torch.from_numpy(m).to(dev); qm = qmask_of(meta, B, dev, Lc)
        x = y.masked_fill(m, MASK)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'xpu'):
            if which == 'a' and (step % 2 == 0 or Lc < L):
                lg = model(x, off + torch.arange(Lc, device=dev).expand(B, -1)).float(); loss = masked_loss(lg, y, m, qm)
            elif which == 'a':  # window batch: every shard + halo, all shards of the batch at once
                losses = []
                for s in range(NS):
                    q0, q1 = max(0, s * SHARD - HALO), min(L, (s + 1) * SHARD + HALO)
                    lg = model(x[:, q0:q1], torch.arange(q0, q1, device=dev).expand(B, -1)).float()
                    mm, qq = m[:, q0:q1], qm[:, q0:q1]
                    if qq.any(): losses.append(masked_loss(lg, y[:, q0:q1], mm, qq))
                loss = sum(losses) / len(losses)
            else:
                loss = None
        if which == 'b':  # 2 micro-batches of 4 (only ~3 GB VRAM free on the B580); same effective batch of 8
            opt.zero_grad(set_to_none=True); tot = 0.0
            for h in (slice(0, B // 2), slice(B // 2, B)):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'xpu'):
                    outs = model(x[h], Lc // SHARD, all_rounds=True, off=off)
                    lh = sum(masked_loss(o.float(), y[h], m[h], qm[h]) for o in outs) / len(outs) / 2
                lh.backward(); tot += float(lh)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); loss = torch.tensor(tot)
        else:
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            log.append({'step': step, 'loss': float(loss), 's': time.time() - t0}); print(json.dumps(log[-1]), flush=True)
    RES.mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg_for(args.small), 'steps': args.steps}, ROOT / f'v7_model_{which}.pt')
    (RES / f'train_{which}.json').write_text(json.dumps({'steps': args.steps, 'cfg': cfg_for(args.small), 'log': log}, indent=1))


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def decode_a(model, x0, H, K, dev):
    x = x0.clone(); B = x.shape[0]
    for t in range(K):
        if H is None: lg = model(x, torch.arange(L, device=dev).expand(B, -1))
        else:
            parts = []
            for s in range(NS):
                c0, c1 = s * SHARD, (s + 1) * SHARD; q0, q1 = max(0, c0 - H), min(L, c1 + H)
                parts.append(model(x[:, q0:q1], torch.arange(q0, q1, device=dev).expand(B, -1))[:, c0 - q0:c1 - q0])
            lg = torch.cat(parts, 1)
        p = lg.float().softmax(-1); p[..., MASK] = 0; conf, pred = p.max(-1); masked = x == MASK; conf = conf.masked_fill(~masked, -1)
        for s in range(NS):
            c0, c1 = s * SHARD, (s + 1) * SHARD; mc = masked[:, c0:c1]; n = torch.ceil(mc.sum(1).float() / (K - t)).long()
            take = (conf[:, c0:c1].argsort(1, descending=True).argsort(1) < n[:, None]) & mc
            seg = x[:, c0:c1]; seg[take] = pred[:, c0:c1][take]
    return x


def evaluate(args):
    dev = device(); out = {'n_seq': args.n_seq, 'rows': []}
    ca = torch.load(ROOT / 'v7_model_a.pt', map_location='cpu'); A = ModelA(**ca['cfg']).to(dev); A.load_state_dict(ca['state_dict']); A.eval()
    cb = torch.load(ROOT / 'v7_model_b.pt', map_location='cpu'); Bm = ModelB(**cb['cfg']).to(dev); Bm.load_state_dict(cb['state_dict']); Bm.eval()
    for es in EVAL_SEEDS:
        rng = np.random.default_rng(es); preds = {}
        y, m, meta = make_batch(rng, args.n_seq); yt = torch.from_numpy(y).to(dev); x0 = yt.masked_fill(torch.from_numpy(m).to(dev), MASK)
        for i in range(0, args.n_seq, 16):
            xb = x0[i:i + 16]
            for name, H in (('global', None), ('isolated', 0), ('halo16', HALO)):
                preds.setdefault(name, []).append(decode_a(A, xb, H, 8, dev).cpu())
            with torch.no_grad():
                for R in (1, 2, 4, 8): preds.setdefault(f'msg_R{R}', []).append(Bm(xb, R).float().argmax(-1).cpu())
        preds = {k: torch.cat(v).numpy() for k, v in preds.items()}
        for b, qs in enumerate(meta):
            for vp, dp in qs:
                d = int(abs(vp // SHARD - dp // SHARD))
                out['rows'].append({'seed': es, 'seq': int(b), 'd': d, **{k: int(v[b, vp] == y[b, vp]) for k, v in preds.items()}})
        print('eval seed', es, 'done', flush=True)
    RES.mkdir(exist_ok=True); (RES / 'eval_rows.json').write_text(json.dumps(out)); analyze(out)


def analyze(out=None):
    out = out or json.loads((RES / 'eval_rows.json').read_text()); rows = out['rows']
    methods = [k for k in rows[0] if k not in ('seed', 'seq', 'd')]
    def acc(seed, meth, lo, hi, boot=None):
        r = [x for x in rows if x['seed'] == seed and lo <= x['d'] <= hi]
        seqs = sorted({x['seq'] for x in r}); c = np.array([[sum(x[meth] for x in r if x['seq'] == s), sum(1 for x in r if x['seq'] == s)] for s in seqs], float)
        if boot is None: return c[:, 0].sum() / max(1, c[:, 1].sum()), int(c[:, 1].sum())
        ix = np.random.default_rng(seed).integers(0, len(c), (boot, len(c))); return c[ix, 0].sum(1) / np.maximum(1, c[ix, 1].sum(1))
    table = {}
    for s in EVAL_SEEDS:
        for bk, (lo, hi) in BUCKETS.items():
            table[f'{s}_{bk}'] = {m_: round(acc(s, m_, lo, hi)[0], 4) for m_ in methods}; table[f'{s}_{bk}']['n'] = acc(s, 'global', lo, hi)[1]
    far = {s: {m_: acc(s, m_, 2, 7)[0] for m_ in methods} for s in EVAL_SEEDS}
    crit = {}
    crit['sanity_d0_global'] = {s: acc(s, 'global', 0, 0)[0] for s in EVAL_SEEDS}
    crit['C0'] = all(far[s]['global'] >= 0.90 and far[s]['isolated'] <= 0.116 for s in EVAL_SEEDS)
    crit['R_tok'] = {s: (far[s]['halo16'] - far[s]['isolated']) / max(1e-9, far[s]['global'] - far[s]['isolated']) for s in EVAL_SEEDS}
    crit['H1'] = all(v >= 0.5 for v in crit['R_tok'].values())
    crit['msg_R8_over_global'] = {s: far[s]['msg_R8'] / max(1e-9, far[s]['global']) for s in EVAL_SEEDS}
    crit['H2'] = all(v >= 0.9 for v in crit['msg_R8_over_global'].values())
    crit['msg_R2_d4plus'] = {s: acc(s, 'msg_R2', 4, 7)[0] for s in EVAL_SEEDS}
    crit['H2_mechanism'] = all(v <= 0.116 for v in crit['msg_R2_d4plus'].values())
    sanity = all(v >= 0.90 for v in crit['sanity_d0_global'].values())
    if not sanity: verdict = 'TRAINING_FAILED_SANITY'
    elif not crit['C0']: verdict = 'INCONCLUSIVE_C0'
    elif crit['H2'] and not crit['H2_mechanism']: verdict = 'H2_INVALID_LEAKAGE'
    else: verdict = f"H1_{'PASS' if crit['H1'] else 'FAIL'}__H2_{'PASS' if crit['H2'] else 'FAIL'}"
    res = {'verdict': verdict, 'criteria': crit, 'far_acc': far, 'table': table}
    (RES / 'analysis.json').write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['train_a', 'train_b', 'eval', 'analyze'])
    ap.add_argument('--steps', type=int, default=9000); ap.add_argument('--small', action='store_true'); ap.add_argument('--n_seq', type=int, default=256)
    a = ap.parse_args()
    {'train_a': lambda: train(a, 'a'), 'train_b': lambda: train(a, 'b'), 'eval': lambda: evaluate(a), 'analyze': lambda: analyze()}[a.mode]()
