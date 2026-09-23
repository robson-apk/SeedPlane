"""SeedPlane V11: vectorized GPU execution (part A) + long-context model up to 8,192 tokens (part B). See PROTOCOL.md.

python v11.py part_a  --ckpt <v6 ckpt> --cache <cache.pt>
python v11.py train   --cache <cache.pt> [--steps 10000] [--maxpos 8192]
python v11.py sanity  --cache <cache.pt>
python v11.py part_b  --cache <cache.pt>
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint

ROOT = Path(__file__).resolve().parent; RES = ROOT / 'results'
sys.path.insert(0, str(ROOT.parent / 'v8')); sys.path.insert(0, str(ROOT.parent / 'v6'))
from v6 import MASK, SHARD, load_cache
import v8

HALO, K, RATE, SEEDS, W = 16, 16, 0.5, (11, 23, 37), 128 + 2 * 16
dev = torch.device('xpu') if hasattr(torch, 'xpu') and torch.xpu.is_available() else torch.device('cpu')
def sync():
    if dev.type == 'xpu': torch.xpu.synchronize()


class LM(nn.Module):
    """Same architecture as the V6 MDLM, with a configurable position table."""
    def __init__(self, d=256, layers=6, heads=8, ff=1024, maxpos=1024, V=1024):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(maxpos, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.bias = nn.Parameter(torch.zeros(V)); self.scale = 1.0
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
    def hidden(self, x, pos, pad=None): return self.tr(self.emb(x) + self.pos(pos), src_key_padding_mask=pad)
    def logits(self, h): return F.linear(self.norm(h), self.emb.weight) * self.scale + self.bias
    def forward(self, x, pos): return self.logits(self.hidden(x, pos))


def load_lm(path, maxpos):
    ck = torch.load(path, map_location='cpu'); m = LM(maxpos=maxpos, **{k: v for k, v in ck['cfg'].items() if k in ('d', 'layers', 'heads', 'ff')})
    m.load_state_dict(ck['state_dict']); return m.to(dev).eval()


# ---------------------------------------------------------------- GPU-resident decoders
def commit_gpu(x, pred, conf, t):
    P, L = x.shape; ns = L // SHARD; masked = x == MASK
    cs = conf.masked_fill(~masked, -1.0).view(P, ns, SHARD); ms = masked.view(P, ns, SHARD)
    n = torch.ceil(ms.sum(-1).float() / (K - t)).long()
    rank = cs.argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
    take = ((rank < n[..., None]) & ms).view(P, L)
    return torch.where(take, pred, x)


def decide(z):
    p = z.float().softmax(-1); p[..., MASK] = 0; c, pr = p.max(-1); return pr, c


@torch.inference_mode()
def trad_gpu(model, x0, chunk=8):
    x = x0.clone(); P, L = x.shape; pos = torch.arange(L, device=dev)
    for t in range(K):
        prs, cfs = [], []
        for i in range(0, P, chunk):
            pr, cf = decide(model(x[i:i + chunk], pos.expand(len(x[i:i + chunk]), -1))); prs.append(pr); cfs.append(cf)
        x = commit_gpu(x, torch.cat(prs), torch.cat(cfs), t)
    return x


def window_index(L):
    c0 = torch.arange(0, L, SHARD, device=dev)[:, None]; idx = c0 - HALO + torch.arange(W, device=dev)[None]
    pad = (idx < 0) | (idx >= L); return idx.clamp(0, L - 1), pad


@torch.inference_mode()
def sp_gpu(model, x0, chunk=512):
    """All shards of all pages in ONE batch (per chunk of windows); cores are always window[16:144]."""
    x = x0.clone(); P, L = x.shape; ns = L // SHARD; idx, pad = window_index(L)
    padb = pad.expand(P, ns, W).reshape(P * ns, W); posb = idx.expand(P, ns, W).reshape(P * ns, W)
    for t in range(K):
        win = x[:, idx].reshape(P * ns, W); prs, cfs = [], []
        for i in range(0, P * ns, chunk):
            h = model.hidden(win[i:i + chunk], posb[i:i + chunk], padb[i:i + chunk])[:, HALO:HALO + SHARD]
            pr, cf = decide(model.logits(h)); prs.append(pr); cfs.append(cf)
        x = commit_gpu(x, torch.cat(prs).view(P, L), torch.cat(cfs).view(P, L), t)
    return x


@torch.inference_mode()
def sp_v10(model, x0np, L):
    """Port of the V10 endpoint path: batches per window length, per-shard Python loop, CPU commit every step."""
    x = x0np.copy(); P = len(x)
    for t in range(K):
        pred = np.zeros((P, L), np.int64); conf = np.full((P, L), -1.0, np.float32); groups = {}
        for p in range(P):
            for c0 in range(0, L, SHARD):
                q0, q1 = max(0, c0 - HALO), min(L, c0 + SHARD + HALO); groups.setdefault(q1 - q0, []).append((p, c0, q0, q1))
        for g in groups.values():
            for i in range(0, len(g), 64):
                sub = g[i:i + 64]
                xb = torch.from_numpy(np.stack([x[p, q0:q1] for p, _, q0, q1 in sub])).to(dev); pb = torch.from_numpy(np.stack([np.arange(q0, q1) for *_, q0, q1 in sub])).to(dev)
                h = model.hidden(xb, pb); cores = torch.stack([h[j, c0 - q0:c0 - q0 + SHARD] for j, (_, c0, q0, _) in enumerate(sub)])
                pr, cf = decide(model.logits(cores)); sync(); pr, cf = pr.cpu().numpy(), cf.cpu().numpy()
                for j, (p, c0, _, _) in enumerate(sub): pred[p, c0:c0 + SHARD] = pr[j]; conf[p, c0:c0 + SHARD] = cf[j]
        for p in range(P): v8.commit(x[p], pred[p], np.where(x[p] == MASK, conf[p], -1.0), t, L)
    return x


def timed(fn, *a):
    sync(); t0 = time.perf_counter(); out = fn(*a); sync(); return out, time.perf_counter() - t0


def pages(val, L, seed, n):
    g = np.random.default_rng(seed); out = []
    for s in g.integers(0, len(val) - L - 1, n):
        y = val[s:s + L].numpy(); m = g.random(L) < RATE; out.append((y, m))
    return out


# ---------------------------------------------------------------- part A
def part_a(a):
    torch.set_num_threads(1); model = load_lm(a.ckpt, 1024); _, val = load_cache(a.cache); L = 1024; rows = []
    for mode, n_pages, reps in (('latency', 1, 8), ('throughput', 32, 3)):
        w = torch.full((n_pages, L), MASK, device=dev); sp_gpu(model, w); trad_gpu(model, w); sp_v10(model, w.cpu().numpy(), L)  # warm
        for seed in SEEDS:
            batches = [pages(val, L, seed, 8)[i:i + 1] for i in range(8)] if mode == 'latency' else [pages(val, L, seed, 32)] * reps
            for bi, b in enumerate(batches):
                ys = np.stack([y for y, _ in b]); ms = np.stack([m for _, m in b]); x0 = np.where(ms, MASK, ys)
                r = {'mode': mode, 'seed': seed, 'batch': bi, 'n_masked': int(ms.sum())}; outs = {}
                for meth in np.random.default_rng(seed * 100 + bi).permutation(['sp_gpu', 'trad_gpu', 'sp_v10']):
                    if meth == 'sp_v10': xf, dt = timed(sp_v10, model, x0, L)
                    else: xf, dt = timed(sp_gpu if meth == 'sp_gpu' else trad_gpu, model, torch.from_numpy(x0).to(dev)); xf = xf.cpu().numpy()
                    r[f'{meth}_s'] = dt; r[f'{meth}_correct'] = int(((xf == ys) & ms).sum()); outs[meth] = xf
                r['agree_spgpu_spv10'] = float((outs['sp_gpu'][ms] == outs['sp_v10'][ms]).mean()); rows.append(r)
            print(f'part_a {mode} seed {seed} done', flush=True)
    RES.mkdir(exist_ok=True); (RES / 'part_a_rows.json').write_text(json.dumps(rows))
    out = {}
    for mode in ('latency', 'throughput'):
        for seed in SEEDS:
            r = [x for x in rows if x['mode'] == mode and x['seed'] == seed]; T = {k: np.array([x[f'{k}_s'] for x in r]) for k in ('sp_gpu', 'trad_gpu', 'sp_v10')}; n = sum(x['n_masked'] for x in r)
            ix = np.random.default_rng(seed).integers(0, len(r), (2000, len(r)))
            ci = lambda A, B: np.quantile(A[ix].sum(1) / B[ix].sum(1), [.025, .975]).tolist()
            out[f'{mode}_{seed}'] = {**{f'{k}_ms_median': 1000 * float(np.median(v)) for k, v in T.items()}, **{f'{k}_tok_s': n / float(v.sum()) for k, v in T.items()},
                                     'spgpu_over_v10': float(T['sp_gpu'].sum() / T['sp_v10'].sum()), 'spgpu_over_v10_ci': ci(T['sp_gpu'], T['sp_v10']),
                                     'spgpu_over_trad': float(T['sp_gpu'].sum() / T['trad_gpu'].sum()), 'spgpu_over_trad_ci': ci(T['sp_gpu'], T['trad_gpu']),
                                     **{f'{k}_acc': sum(x[f'{k}_correct'] for x in r) / n for k in T}, 'agree': float(np.mean([x['agree_spgpu_spv10'] for x in r]))}
    crit = {'A1': all(out[f'latency_{s}']['spgpu_over_v10_ci'][1] < 0.5 for s in SEEDS), 'A2': all(out[f'latency_{s}']['spgpu_over_trad_ci'][1] < 1 for s in SEEDS),
            'A3': all(out[f'{m}_{s}']['agree'] >= 0.99 for m in ('latency', 'throughput') for s in SEEDS)}
    (RES / 'part_a_analysis.json').write_text(json.dumps({'criteria': crit, 'per': out}, indent=1)); print(json.dumps({'criteria': crit, 'per': out}, indent=1))


# ---------------------------------------------------------------- part B: long-context model
def train(a):
    torch.manual_seed(1); train_ids, _ = load_cache(a.cache); MP = a.maxpos
    model = LM(maxpos=MP).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    lens = [l for l in (1024, 2048, 4096, 8192) if l <= MP]; log = []; t0 = time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True); hi = 0.3 if step < 3000 else 1.0; tot = 0.0
        if step % 2 == 0:
            L = lens[(step // 2) % len(lens)]; B = max(1, 8192 // L); micro = [(B, L, torch.arange(L))]
        else:
            H = [16, 64][(step // 2) % 2]; k = int(torch.randint(0, MP // SHARD, (1,))); c0 = k * SHARD; q0, q1 = max(0, c0 - H), min(MP, c0 + SHARD + H)
            micro = [(24, q1 - q0, torch.arange(q0, q1))] * 2
        for B, L, pos in micro:
            st = torch.randint(0, len(train_ids) - L - 1, (B,)); y = torch.stack([train_ids[s:s + L] for s in st.tolist()])
            m = torch.rand(y.shape) < (0.05 + (hi - 0.05) * torch.rand(B, 1)); x = y.masked_fill(m, MASK)
            x, y, m, p = x.to(dev), y.to(dev), m.to(dev), pos.expand(B, -1).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'xpu'):
                if L >= 4096:  # gradient checkpointing per layer (memory only; identical math) — PROTOCOL addendum 1
                    h = model.emb(x) + model.pos(p)
                    for layer in model.tr.layers: h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
                else:
                    h = model.hidden(x, p)
            loss = F.cross_entropy(model.logits(h.float()[m]), y[m]) / len(micro); loss.backward(); tot += float(loss)
        opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            log.append({'step': step, 'loss': tot, 's': time.time() - t0}); print(json.dumps(log[-1]), flush=True)
    torch.save({'state_dict': model.state_dict(), 'cfg': {'d': 256, 'layers': 6, 'heads': 8, 'ff': 1024, 'maxpos': MP}, 'steps': a.steps, 'seed': 1}, ROOT / 'v11_long_model.pt')
    RES.mkdir(exist_ok=True); (RES / 'train_long.json').write_text(json.dumps({'log': log, 'steps': a.steps, 'maxpos': MP}, indent=1))


def sanity(a):
    model = load_lm(ROOT / 'v11_long_model.pt', 8192); tr, val = load_cache(a.cache)
    p = torch.bincount(tr, minlength=1024).float(); p /= p.sum(); uni = float(-(p[p > 0] * p[p > 0].log()).sum()); out = {'unigram': uni}
    g = torch.Generator().manual_seed(999)
    with torch.inference_mode():
        for L in (1024, 8192):
            s = int(torch.randint(0, len(val) - L - 1, (1,), generator=g)); y = val[s:s + L][None].to(dev); m = (torch.rand(1, L, generator=g) < 0.15).to(dev)
            z = model(y.masked_fill(m, MASK), torch.arange(L, device=dev)[None]).float(); out[f'loss15_L{L}'] = float(F.cross_entropy(z[m], y[m]))
    out['gate_pass'] = out['loss15_L8192'] <= uni - 1.0; (RES / 'sanity_long.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out))


def part_b(a):
    torch.set_num_threads(1); model = load_lm(ROOT / 'v11_long_model.pt', 8192); _, val = load_cache(a.cache); rows = []
    for L in (1024, 2048, 4096, 8192):
        w = torch.full((1, L), MASK, device=dev); sp_gpu(model, w); trad_gpu(model, w, chunk=1)
        for seed in SEEDS:
            for bi, (y, m) in enumerate(pages(val, L, seed + L, 6)):
                x0 = torch.from_numpy(np.where(m, MASK, y)[None]).to(dev); r = {'L': L, 'seed': seed, 'page': bi, 'n_masked': int(m.sum())}
                for meth in np.random.default_rng(seed * 100 + bi).permutation(['sp_gpu', 'trad_gpu']):
                    xf, dt = timed(sp_gpu if meth == 'sp_gpu' else (lambda mm, xx: trad_gpu(mm, xx, 1)), model, x0)
                    r[f'{meth}_s'] = dt; r[f'{meth}_correct'] = int(((xf.cpu().numpy()[0] == y) & m).sum())
                rows.append(r)
            print(f'part_b L={L} seed {seed} done', flush=True)
    (RES / 'part_b_rows.json').write_text(json.dumps(rows)); out = {}
    for L in (1024, 2048, 4096, 8192):
        for seed in SEEDS:
            r = [x for x in rows if x['L'] == L and x['seed'] == seed]; ts = np.array([x['sp_gpu_s'] for x in r]); tt = np.array([x['trad_gpu_s'] for x in r])
            n = np.array([x['n_masked'] for x in r], float); cs = np.array([x['sp_gpu_correct'] for x in r], float); ct = np.array([x['trad_gpu_correct'] for x in r], float)
            ix = np.random.default_rng(seed).integers(0, len(r), (2000, len(r)))
            out[f'L{L}_{seed}'] = {'sp_ms_median': 1000 * float(np.median(ts)), 'trad_ms_median': 1000 * float(np.median(tt)), 'ratio': float(ts.sum() / tt.sum()),
                                   'ratio_ci': np.quantile(ts[ix].sum(1) / tt[ix].sum(1), [.025, .975]).tolist(), 'acc_sp': cs.sum() / n.sum(), 'acc_trad': ct.sum() / n.sum(),
                                   'acc_diff_pp': 100 * (cs.sum() - ct.sum()) / n.sum(), 'acc_diff_ci_pp': (100 * np.quantile((cs[ix].sum(1) - ct[ix].sum(1)) / n[ix].sum(1), [.025, .975])).tolist(),
                                   'sp_tok_s': n.sum() / ts.sum(), 'trad_tok_s': n.sum() / tt.sum()}
    crit = {'B1': all(out[f'L8192_{s}']['ratio_ci'][1] < 0.90 for s in SEEDS), 'B2': all(out[f'L8192_{s}']['acc_diff_ci_pp'][0] > -0.5 for s in SEEDS)}
    cross = [L for L in (1024, 2048, 4096, 8192) if all(out[f'L{L}_{s}']['ratio_ci'][1] < 1 for s in SEEDS)]; crit['crossover_L'] = cross[0] if cross else None
    (RES / 'part_b_analysis.json').write_text(json.dumps({'criteria': crit, 'per': out}, indent=1)); print(json.dumps({'criteria': crit, 'per': out}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['part_a', 'train', 'sanity', 'part_b'])
    ap.add_argument('--ckpt'); ap.add_argument('--cache'); ap.add_argument('--steps', type=int, default=10000); ap.add_argument('--maxpos', type=int, default=8192)
    a = ap.parse_args(); {'part_a': part_a, 'train': train, 'sanity': sanity, 'part_b': part_b}[a.mode](a)
