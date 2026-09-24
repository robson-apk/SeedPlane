"""Render README charts (light + dark SVG) from the committed experiment results.

python docs/make_charts.py      (from the repo root)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

REPO = Path(__file__).resolve().parents[1]; OUT = REPO / 'docs' / 'img'; OUT.mkdir(parents=True, exist_ok=True)
EXP = REPO / 'experiments'
import os
CHECK = Path(os.environ['CHART_CHECK']) if os.environ.get('CHART_CHECK') else None

# SeedPlane's README visual system: warm paper, ink, and stable colors for each method.
THEMES = {
    'light': dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', grid='#e4e3df', ref='#8a8984',
                  sp='#2a78d6', trad='#eb6834', v8='#1baf7a'),
    'dark': dict(surface='#1a1a19', text='#ffffff', text2='#c3c2b7', grid='#383835', ref='#8f8e88',
                 sp='#3987e5', trad='#d95926', v8='#199e70'),
}
NAMES = {'sp': 'SeedPlane (optimized)', 'trad': 'Traditional Transformer', 'v8': 'SeedPlane (V8)'}


def base(t, title, subtitle, w=8.4, h=4.9):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig.patch.set_facecolor(t['surface']); ax.set_facecolor(t['surface'])
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(t['grid'])
    ax.tick_params(colors=t['text2'], labelsize=9, length=0, pad=7)
    ax.grid(axis='y', color=t['grid'], linewidth=0.8); ax.set_axisbelow(True)
    fig.text(0.035, 0.95, title, fontsize=15, fontweight='bold', color=t['text'], va='top')
    fig.text(0.035, 0.885, subtitle, fontsize=9.5, color=t['text2'], va='top')
    fig.subplots_adjust(top=0.77, left=0.12, right=0.77, bottom=0.15)
    return fig, ax


def lines(ax, t, xs, series, fmt, min_gap_px=38, names=None):
    """Direct labels at the line ends, nudged apart so they never collide."""
    names = names or NAMES
    for key, ys in series.items():
        ax.plot(xs, ys, color=t[key], linewidth=2.6, marker='o', markersize=7, markeredgecolor=t['surface'], markeredgewidth=1.8, zorder=3)
    ax.figure.canvas.draw()
    ends = sorted(((ax.transData.transform((xs[-1], ys[-1]))[1], key, ys[-1]) for key, ys in series.items()), reverse=True)
    placed = []
    for y_px, key, v in ends:
        y_lab = y_px if not placed else min(y_px, placed[-1] - min_gap_px)
        placed.append(y_lab)
        ax.annotate(f'{names[key]}\n{fmt(v)}', (xs[-1], v), xytext=(10, y_lab - y_px), textcoords='offset pixels',
                    va='center', fontsize=8.5, color=t['text'], annotation_clip=False)


def save(fig, name, mode):
    svg = OUT / f'{name}-{mode}.svg'
    fig.savefig(svg, facecolor=fig.get_facecolor())
    # Matplotlib emits indentation spaces at the end of SVG path lines; keep generated diffs clean.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    if CHECK: fig.savefig(CHECK / f'{name}-{mode}.png', facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    have_v9 = (EXP / 'v9/results/rows.json').exists() and (EXP / 'v9/results/analysis.json').exists()
    have_mem = (EXP / 'v9/results/memory.json').exists()
    v8 = json.loads((EXP / 'v8/results/analysis.json').read_text()); v7 = json.loads((EXP / 'v7/results/analysis.json').read_text()); L = 1024
    if have_v9:
        v9 = json.loads((EXP / 'v9/results/analysis.json').read_text()); rows = json.loads((EXP / 'v9/results/rows.json').read_text())
        cores = sorted({r['c'] for r in rows})
        tps = {k: [np.median([r['n_masked'] / r[f'{m}_s'] for r in rows if r['L'] == L and r['c'] == c]) for c in cores]
               for k, m in (('sp', 'sp_opt'), ('v8', 'sp_v8'), ('trad', 'trad'))}
    if have_mem: mem = json.loads((EXP / 'v9/results/memory.json').read_text())
    for mode, t in THEMES.items():
      if have_v9:
        # 1. tokens per second
        fig, ax = base(t, 'Tokens per second vs CPU cores', 'L = 1024 · masked tokens filled per second (median) · Ryzen 5600X · same model')
        lines(ax, t, cores, tps, lambda v: f'{v:,.0f} tok/s'); ax.set_xticks(cores); ax.set_xlabel('CPU cores', color=t['text2'], fontsize=9)
        ax.set_ylim(0, max(max(v) for v in tps.values()) * 1.15); save(fig, 'tokens_per_second', mode)
        # 2. scaling vs ideal
        fig, ax = base(t, 'How well each extra core pays off', 'L = 1024 · speed-up over 1 core · dashed = perfect linear scaling')
        ax.plot(cores, cores, color=t['ref'], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate('ideal', (cores[-1], cores[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8.5, color=t['text2'])
        sc = {k: [v9['curves'][f'L{L}_{m}']['speedup_vs_1core'][str(c)] for c in cores] for k, m in (('sp', 'sp_opt'), ('trad', 'trad'))}
        lines(ax, t, cores, sc, lambda v: f'{v:.1f}×'); ax.set_xticks(cores); ax.set_xlabel('CPU cores', color=t['text2'], fontsize=9)
        ax.set_ylim(0, cores[-1] + 0.5); save(fig, 'scaling', mode)
      if have_mem:
        # 3. memory
        mc = sorted({r['c'] for r in mem['rows'] if r['L'] == L})
        ms = {k: [next(r['peak_mb'] for r in mem['rows'] if r['L'] == L and r['c'] == c and r['method'] == m) for c in mc] for k, m in (('sp', 'sp_opt'), ('trad', 'trad'))}
        fig, ax = base(t, 'Peak RAM vs CPU cores', 'L = 1024 · resident memory of the whole process tree (main + workers), peak during decoding')
        lines(ax, t, mc, ms, lambda v: f'{v:,.0f} MB'); ax.set_xticks(mc); ax.set_xlabel('CPU cores', color=t['text2'], fontsize=9)
        ax.set_ylim(0, max(max(v) for v in ms.values()) * 1.15); save(fig, 'memory', mode)
      if True:
        # 4. quality (V8, same model)
        seeds = ['11', '23', '37']; tr = [100 * v8['per'][f'L1024_s{s}']['acc_traditional'] for s in seeds]; sp = [100 * v8['per'][f'L1024_s{s}']['acc_seedplane'] for s in seeds]
        fig, ax = base(t, 'Same model, same text: accuracy', 'L = 1024 · masked-infill accuracy per evaluation seed (V8, pre-registered) · higher is better')
        y = np.arange(3)[::-1]
        for yi, a_, b_ in zip(y, tr, sp): ax.plot([a_, b_], [yi, yi], color=t['grid'], linewidth=2, zorder=1)
        ax.scatter(tr, y, s=70, color=t['trad'], edgecolor=t['surface'], linewidth=1.5, zorder=3, label=NAMES['trad'])
        ax.scatter(sp, y, s=70, color=t['sp'], edgecolor=t['surface'], linewidth=1.5, zorder=3, label='SeedPlane')
        for yi, a_, b_ in zip(y, tr, sp):
            ax.annotate(f'{a_:.1f}%', (a_, yi), xytext=(-8, 0), textcoords='offset points', ha='right', va='center', fontsize=8.5, color=t['text2'])
            ax.annotate(f'{b_:.1f}%  (+{b_ - a_:.1f} pp)', (b_, yi), xytext=(8, 0), textcoords='offset points', ha='left', va='center', fontsize=8.5, color=t['text'])
        ax.set_yticks(y, [f'seed {s}' for s in seeds]); ax.grid(axis='y', visible=False); ax.grid(axis='x', color=t['grid'], linewidth=0.8)
        ax.set_xlim(42.5, 50); ax.set_xlabel('accuracy on hidden tokens (%)', color=t['text2'], fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=t['text'], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.5)
        fig.subplots_adjust(top=0.74, right=0.95); save(fig, 'quality', mode)
        # 5. long-range recall (V7)
        fa = v7['far_acc']; keys = [('global', 'Traditional (global attention)', 'trad'), ('halo16', 'SeedPlane, token halo', 'sp'), ('msg_R8', 'SeedPlane, latent messages', 'sp')]
        vals = [100 * np.mean([fa[s][k] for s in fa]) for k, _, _ in keys]
        fig, ax = base(t, 'Where SeedPlane loses: far-away information', 'V7 synthetic task · recall when the answer sits ≥ 2 shards away · dashed = chance (1.6%)')
        bars = ax.barh(range(3), vals, 0.55, color=[t[c] for _, _, c in keys]); ax.set_yticks(range(3), [n for _, n, _ in keys]); ax.invert_yaxis()
        ax.axvline(100 / 64, color=t['ref'], linewidth=1.2, linestyle=(0, (4, 3))); ax.grid(axis='y', visible=False); ax.grid(axis='x', color=t['grid'], linewidth=0.8)
        for b, v in zip(bars, vals): ax.annotate(f'{v:.1f}%', (v, b.get_y() + b.get_height() / 2), xytext=(5, 0), textcoords='offset points', va='center', fontsize=9, color=t['text'])
        ax.set_xlim(0, 110); fig.subplots_adjust(left=0.33, right=0.95); save(fig, 'long_range', mode)
    if (EXP / 'v10/results/analysis.json').exists():
        v10 = json.loads((EXP / 'v10/results/analysis.json').read_text())['cells']
        sets = ['gpu', 'gpu+cpu4', 'gpu+cpu4+mac2', 'cpu4', 'cpu4+mac2']
        pretty = {'gpu': 'Arc B580', 'gpu+cpu4': 'B580 + 4 Ryzen cores', 'gpu+cpu4+mac2': 'B580 + 4 Ryzen + 2 Mac cores',
                  'cpu4': '4 Ryzen cores', 'cpu4+mac2': '4 Ryzen + 2 Mac cores'}
        for mode_name, key, title, sub in (('throughput', 'tok_s', 'Throughput: 32 pages at once', 'tokens per second (higher is better) · same model · devices synchronized every refinement step'),
                                           ('latency', 'ms_median', 'Latency: one page, split across devices', 'milliseconds per 1,024-token page (lower is better) · traditional runs on the single best device')):
            for mode, t in THEMES.items():
                sp = [np.mean([v10[f'{mode_name}|{s}|{sd}'][f'sp_{key}'] for sd in (11, 23, 37)]) for s in sets]
                tr = [np.mean([v10[f'{mode_name}|{s}|{sd}'][f'trad_{key}'] for sd in (11, 23, 37)]) if f'trad_{key}' in v10[f'{mode_name}|{s}|11'] else None for s in sets]
                fig, ax = base(t, title, sub, h=4.3); y = np.arange(len(sets)); hgt = 0.36
                ax.barh(y - hgt / 2 - 0.01, sp, hgt, color=t['sp'], label='SeedPlane')
                trv = [v if v is not None else 0 for v in tr]
                ax.barh(y + hgt / 2 + 0.01, trv, hgt, color=t['trad'], label='Traditional Transformer')
                fmt = (lambda v: f'{v:,.0f} tok/s') if key == 'tok_s' else (lambda v: f'{v:,.0f} ms')
                for yi, v in zip(y, sp): ax.annotate(fmt(v), (v, yi - hgt / 2), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=t['text'])
                for yi, v in zip(y, tr):
                    if v is not None: ax.annotate(fmt(v), (v, yi + hgt / 2), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=t['text'])
                    else: ax.annotate('n/a — cannot split one page', (0, yi + hgt / 2), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=t['text2'])
                ax.set_yticks(y, [pretty[s] for s in sets]); ax.invert_yaxis(); ax.grid(axis='y', visible=False); ax.grid(axis='x', color=t['grid'], linewidth=0.8)
                ax.set_xlim(0, max(max(sp), max(v for v in tr if v is not None)) * 1.3)
                ax.legend(frameon=False, fontsize=8.5, labelcolor=t['text'], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
                fig.subplots_adjust(left=0.3, right=0.95, top=0.74); save(fig, f'devices_{mode_name}', mode)
    if (EXP / 'v11/results/part_b_analysis.json').exists():
        pb = json.loads((EXP / 'v11/results/part_b_analysis.json').read_text())['per']; Ls = [1024, 2048, 4096, 8192]
        ser = {'sp': [np.mean([pb[f'L{L}_{s}']['sp_ms_median'] for s in (11, 23, 37)]) for L in Ls],
               'trad': [np.mean([pb[f'L{L}_{s}']['trad_ms_median'] for s in (11, 23, 37)]) for L in Ls]}
        for mode, t in THEMES.items():
            fig, ax = base(t, 'Longer text: the traditional Transformer hits the quadratic wall', 'One page on an Intel Arc B580 · milliseconds per page (lower is better) · timing only')
            lines(ax, t, list(range(len(Ls))), ser, lambda v: f'{v:,.0f} ms'); ax.set_xticks(range(len(Ls)), [f'{L:,}' for L in Ls])
            ax.set_xlabel('tokens per page', color=t['text2'], fontsize=9); ax.set_ylim(0, max(ser['trad']) * 1.12); save(fig, 'long_text_gpu', mode)
    v13 = EXP / 'v13/results/speed_run2.json'
    if v13.exists():
        d = json.loads(v13.read_text()); Ls = [4096, 8192, 16384, 32768]; nm = {'trad': 'llama.cpp (native)', 'sp': 'SeedPlane on llama.cpp'}
        ser = {'trad': [d['native'][f'gpu|{L}'] for L in Ls], 'sp': [d['sp'][f'gpu|span|{L}']['tok_s'] for L in Ls]}
        for mode, t in THEMES.items():
            fig, ax = base(t, 'Qwen2.5-0.5B on an Arc B580: prompt tokens per second',
                           'same GGUF, same llama.cpp kernels (Vulkan) · SeedPlane = span mode, halo 256 · quality cost: see next chart')
            lines(ax, t, list(range(len(Ls))), ser, lambda v: f'{v:,.0f} tok/s', names=nm); ax.set_xticks(range(len(Ls)), [f'{L:,}' for L in Ls])
            ax.set_xlabel('prompt length (tokens)', color=t['text2'], fontsize=9); ax.set_ylim(0, max(ser['sp']) * 1.15)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}')); save(fig, 'qwen_speed', mode)
        kv_full = {8192: 99, 16384: 195, 32768: 387}; Lm = sorted(kv_full)
        for mode, t in THEMES.items():
            fig, ax = base(t, 'KV-cache memory: constant instead of growing',
                           'Qwen2.5-0.5B, B580 · KV buffer reported by llama.cpp · both also hold a fixed ~300 MiB compute buffer')
            lines(ax, t, list(range(len(Lm))), {'trad': [kv_full[L] for L in Lm], 'sp': [12] * len(Lm)}, lambda v: f'{v:,.0f} MiB', names=nm)
            ax.set_xticks(range(len(Lm)), [f'{L:,}' for L in Lm]); ax.set_xlabel('prompt length (tokens)', color=t['text2'], fontsize=9)
            ax.set_ylim(0, 440); save(fig, 'qwen_kv_memory', mode)
    fr16, fr32 = EXP / 'v13c/results/frontier_1.json', EXP / 'v13c/results/frontier32k.json'
    if fr16.exists() and fr32.exists():
        native = {16384: 3891.0, 32768: 2096.0}
        pts = []
        for f, L, key in ((fr16, 16384, 'sp'), (fr32, 32768, 'v8')):
            for r in json.loads(f.read_text())['configs']:
                if r['mode'] == 'span': pts.append((L, key, r['H'], r['tok_s'] / native[L], 100 * (r['ppl_ratio_mean'] - 1)))
        for mode, t in THEMES.items():
            fig, ax = base(t, 'The trade-off, measured: speed vs quality',
                           'Qwen2.5-0.5B, B580, WikiText · perplexity change vs full attention (lower is better) · point labels = halo size', h=4.3)
            ax.axhline(0, color=t['ref'], linewidth=1.2, linestyle=(0, (4, 3))); ax.axvline(1, color=t['ref'], linewidth=1.2, linestyle=(0, (4, 3)))
            ax.annotate('same quality as the original model', (0.99, 0), xycoords=('axes fraction', 'data'), xytext=(0, -4), textcoords='offset points',
                        ha='right', va='top', fontsize=8, color=t['text2'])
            ax.annotate('llama.cpp speed', (1, 1), xycoords=('data', 'axes fraction'), xytext=(4, -4), textcoords='offset points', va='top', fontsize=8, color=t['text2'])
            for L, key in ((16384, 'sp'), (32768, 'v8')):
                P = sorted([p for p in pts if p[0] == L], key=lambda p: p[3])
                ax.plot([p[3] for p in P], [p[4] for p in P], color=t[key], linewidth=2, marker='o', markersize=7, markeredgecolor=t['surface'], markeredgewidth=1.5, zorder=3,
                        label=f'{L:,} tokens')
                for p in P: ax.annotate(f'{p[2]:,}', (p[3], p[4]), xytext=(6, 5), textcoords='offset points', fontsize=8, color=t['text'])
            ax.set_xlabel('speed-up over llama.cpp native (×)', color=t['text2'], fontsize=9); ax.set_ylabel('perplexity change (%)', color=t['text2'], fontsize=9)
            ax.set_xlim(0, 5.6); ax.grid(axis='x', color=t['grid'], linewidth=0.8)
            ax.legend(frameon=False, fontsize=8.5, labelcolor=t['text'], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
            fig.subplots_adjust(top=0.74, right=0.95, left=0.1); save(fig, 'qwen_frontier', mode)
    print('wrote', sorted(p.name for p in OUT.glob('*.svg')))


if __name__ == '__main__':
    main()
