"""Render README GIFs (light + dark) from a REAL decoding trajectory (docs/trajectory.json) and REAL measured timings (V8).

python docs/make_gifs.py     (from the repo root)
"""
import json, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]; OUT = REPO / 'docs' / 'img'; OUT.mkdir(parents=True, exist_ok=True)
TR = json.loads((REPO / 'docs/trajectory.json').read_text())
V8 = json.loads((REPO / 'experiments/v8/results/analysis.json').read_text())
T_TRAD = np.mean([V8['per'][f'L1024_s{s}']['time_traditional_ms_median'] for s in (11, 23, 37)])
T_SP = np.mean([V8['per'][f'L1024_s{s}']['time_seedplane_ms_median'] for s in (11, 23, 37)])
THEMES = {
    'light': dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', empty='#e9e8e4', given='#c9c8c3', halo='#f3d9b1', sp='#2a78d6', trad='#eb6834', v8='#1baf7a'),
    'dark': dict(surface='#1a1a19', text='#ffffff', text2='#c3c2b7', empty='#2b2b29', given='#4a4a47', halo='#5a4424', sp='#3987e5', trad='#d95926', v8='#199e70'),
}
L, SH, K = TR['L'], TR['shard'], TR['steps']; MASKED = np.array(TR['masked'])
FPS = 20; SLOW = 3.0                                   # GIF plays at 1/3 of real speed so the steps are visible
DUR_MS = T_TRAD * SLOW; HOLD = 40                      # frames held at the end


def grid_colors(t, kind, step_now):
    st = np.array(TR[kind]['commit_step']); col = np.empty(L, dtype=object)
    col[~MASKED] = t['given']; col[MASKED] = t['empty']; done = MASKED & (st >= 0) & (st < step_now); col[done] = t[kind]
    return col.reshape(32, 32)


def hero(mode):
    t = THEMES[mode]; fig = plt.figure(figsize=(8.4, 4.9), dpi=90); fig.patch.set_facecolor(t['surface'])
    fig.text(0.03, 0.955, 'Same model. Same text. Half the time.', fontsize=16, fontweight='bold', color=t['text'], va='top')
    fig.text(0.03, 0.885, f'Real decoding of one 1,024-token page · 16 refinement steps · 4 CPU cores · played at 1/{SLOW:.0f} speed',
             fontsize=9.5, color=t['text2'], va='top')
    axes, cells, clocks = {}, {}, {}
    for j, (kind, title, total) in enumerate((('trad', 'Traditional Transformer', T_TRAD), ('sp', 'SeedPlane', T_SP))):
        ax = fig.add_axes([0.03 + j * 0.49, 0.1, 0.45, 0.68]); ax.set_xlim(0, 32); ax.set_ylim(33.5, -0.5); ax.axis('off')
        ax.text(0, -0.9, title, fontsize=12, fontweight='bold', color=t['text'], va='bottom')
        rects = []
        for r in range(32):
            for c in range(32):
                rects.append(Rectangle((c + 0.08, r + 0.08), 0.84, 0.84, linewidth=0))
        for rc in rects: ax.add_patch(rc)
        if kind == 'sp':                                   # shard boundaries: 4 rows = 128 tokens = one core's shard
            for b in range(4, 32, 4): ax.plot([0, 32], [b, b], color=t['sp'], linewidth=1.4, solid_capstyle='butt')
            ax.text(32, -0.9, '8 shards · lines = core boundaries', fontsize=8.5, color=t['text2'], va='bottom', ha='right')
        cells[kind] = rects; axes[kind] = ax
        clocks[kind] = ax.text(0, 33.3, '', fontsize=10.5, color=t['text'], va='top', fontfamily='monospace')
    lx = 0.03
    for label, key in (('given word', 'given'), ('still hidden', 'empty'), ('filled (traditional)', 'trad'), ('filled (SeedPlane)', 'sp')):
        fig.patches.append(Rectangle((lx, 0.028), 0.014, 0.024, transform=fig.transFigure, facecolor=t[key], linewidth=0))
        txt = fig.text(lx + 0.02, 0.04, label, fontsize=8.5, color=t['text2'], va='center')
        lx += 0.02 + txt.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + 0.03
    n_frames = int(DUR_MS / 1000 * FPS) + HOLD

    def draw(f):
        ms_real = min(f / FPS * 1000, DUR_MS) / SLOW
        for kind, total in (('trad', T_TRAD), ('sp', T_SP)):
            step = min(K, int(ms_real / (total / K)))
            cols = grid_colors(t, kind, step).ravel()
            for rc, c in zip(cells[kind], cols): rc.set_facecolor(c)
            done = step >= K
            clocks[kind].set_text(f'{min(ms_real, total):6.0f} ms   step {step:2d}/{K}' + ('   ✓ done' if done else ''))
        return []
    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / FPS)
    anim.save(OUT / f'hero-race-{mode}.gif', writer=PillowWriter(fps=FPS)); plt.close(fig)


def how(mode):
    t = THEMES[mode]; words = TR['truth']; st = np.array(TR['sp']['commit_step']); fin = TR['sp']['final']
    fig = plt.figure(figsize=(8.4, 5.2), dpi=90); fig.patch.set_facecolor(t['surface'])
    fig.text(0.03, 0.96, 'Every core writes its own paragraph — at the same time', fontsize=15, fontweight='bold', color=t['text'], va='top')
    fig.text(0.03, 0.9, 'Four shards of the same page, real model output. Hidden words (▢) are filled in parallel by every core, '
             'step by step.', fontsize=9, color=t['text2'], va='top')
    shards = [0, 1, 2, 3]; N = 36                                       # first 36 words of each of four shards
    boxes = []
    for i, sh in enumerate(shards):
        y0 = 0.72 - i * 0.175; ax = fig.add_axes([0.03, y0, 0.94, 0.15]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0,rounding_size=0.04', facecolor=t['empty'], linewidth=0, alpha=0.55))
        ax.add_patch(Rectangle((0, 0), 0.012, 1, facecolor=t['sp'], linewidth=0))
        ax.text(0.025, 0.86, f'Core {i + 1} · tokens {sh * SH}–{sh * SH + SH - 1}', fontsize=8.5, color=t['text2'], va='top')
        boxes.append((ax, sh * SH))
    global REND; REND = fig.canvas.get_renderer()
    stepc = fig.text(0.97, 0.055, '', fontsize=10, color=t['text'], ha='right', fontfamily='monospace')
    fig.text(0.03, 0.02, '… = word outside the model’s 1,024-word vocabulary', fontsize=8, color=t['text2'])

    def line(ax, start, step_now):
        for artist in list(ax.texts)[1:]: artist.remove()
        x, y = 0.025, 0.58
        for k in range(start, start + N):
            w = words[k]; hidden = MASKED[k]; shown = (not hidden) or (0 <= st[k] < step_now)
            txt = fin[k] if hidden and shown else w
            txt = '…' if txt == '<unk>' else txt
            if not shown: txt, color, weight = '▢', t['text2'], 'normal'
            elif hidden: color, weight = t['sp'], 'bold'
            else: color, weight = t['text'], 'normal'
            art = ax.text(x, y, txt, fontsize=9.5, color=color, fontweight=weight, va='top', fontfamily='DejaVu Sans')
            wlen = art.get_window_extent(REND).width / ax.bbox.width
            if x + wlen > 0.985:
                x = 0.025; y -= 0.3
                if y < 0.05: art.remove(); break
                art.set_position((x, y))
            x += wlen + 0.009

    frames = [s for s in range(K + 1) for _ in range(6)] + [K] * 30
    def draw(f):
        s = frames[f]
        for ax, start in boxes: line(ax, start, s)
        stepc.set_text(f'refinement step {s:2d}/{K}')
        return []
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / 10)
    anim.save(OUT / f'how-it-works-{mode}.gif', writer=PillowWriter(fps=10)); plt.close(fig)


def long_race(mode):
    """8,192-token page on the Arc B580, real V11 timings (3,814 ms vs 323 ms). Fill order is illustrative (timing is real)."""
    t = THEMES[mode]; pb = json.loads((REPO / 'experiments/v11/results/part_b_analysis.json').read_text())['per']
    T = {k: float(np.mean([pb[f'L8192_{s}'][f'{k}_ms_median'] for s in (11, 23, 37)])) for k in ('trad', 'sp')}
    rows, cols = 64, 128; rng = np.random.default_rng(7); masked = rng.random(rows * cols) < 0.5
    order = {k: rng.permutation(np.flatnonzero(masked)) for k in ('trad', 'sp')}
    fig = plt.figure(figsize=(8.4, 4.6), dpi=90); fig.patch.set_facecolor(t['surface'])
    fig.text(0.03, 0.955, '8,192 tokens on one GPU: 11.8× faster', fontsize=16, fontweight='bold', color=t['text'], va='top')
    fig.text(0.03, 0.885, 'Intel Arc B580 · one page · real measured time, shown in real time · fill pattern illustrative', fontsize=9.5, color=t['text2'], va='top')
    ims, clocks = {}, {}
    for j, (k, title) in enumerate((('trad', 'Traditional Transformer'), ('sp', 'SeedPlane'))):
        ax = fig.add_axes([0.03 + j * 0.49, 0.12, 0.45, 0.63]); ax.axis('off')
        ax.text(0, 1.03, title, transform=ax.transAxes, fontsize=12, fontweight='bold', color=t['text'], va='bottom')
        ims[k] = ax.imshow(np.zeros((rows, cols, 3)), interpolation='nearest', aspect='auto')
        clocks[k] = ax.text(0, -0.08, '', transform=ax.transAxes, fontsize=11, color=t['text'], va='top', fontfamily='monospace')
    rgb = lambda h: np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    FPS2 = 25; total = T['trad'] / 1000; frames = int(total * FPS2) + 30
    def draw(f):
        now = min(f / FPS2, total) * 1000
        for k in ('trad', 'sp'):
            done = int(min(1.0, now / T[k]) * len(order[k])); img = np.tile(rgb(t['given']), (rows * cols, 1)); img[masked] = rgb(t['empty']); img[order[k][:done]] = rgb(t[k])
            ims[k].set_data(img.reshape(rows, cols, 3)); fin = now >= T[k]
            clocks[k].set_text(f'{min(now, T[k]):7,.0f} ms' + ('   ✓ done' if fin else ''))
        return []
    FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS2).save(OUT / f'long-race-{mode}.gif', writer=PillowWriter(fps=FPS2)); plt.close(fig)


def scheduler(mode):
    """How one prompt is split across B580 + CPU slots + Mac: REAL numbers from V13 run 2 (gpu+cpu4+mac, span, 16,384)."""
    d = json.loads((REPO / 'experiments/v13/results/speed_run2.json').read_text())['sp']
    rates = d['gpu+cpu4+mac|16384']['rates']; row = d['gpu+cpu4+mac|span|16384']; tpw = row['tokens_per_worker']
    tl = {e['worker']: e['end'] for e in row['timeline']}; L_ = 16384; H = 256
    t = THEMES[mode]
    lanes = [('127.0.0.1:54000', 'Arc B580 GPU', 'sp'), ('127.0.0.1:54002', 'CPU slot 1 (2 threads)', 'trad'),
             ('127.0.0.1:54002#1', 'CPU slot 2 (2 threads)', 'trad'), ('10.0.0.92:54110', 'Mac M4 over the LAN', 'v8')]
    spans, c0 = {}, 0
    for w, _, _ in sorted(lanes, key=lambda x: rates[x[0]]):          # slow devices take the first, small pieces
        if tpw[w]: spans[w] = (c0, c0 + tpw[w]); c0 += tpw[w]
    T_END = max(tl.values()); FPS_ = 12; P1, P2, P3, HOLD_ = 36, 48, 60, 48
    fig = plt.figure(figsize=(8.4, 4.9), dpi=90); fig.patch.set_facecolor(t['surface'])
    fig.text(0.03, 0.955, 'How one prompt is shared between a GPU, CPU cores and a Mac', fontsize=15, fontweight='bold', color=t['text'], va='top')
    fig.text(0.03, 0.885, 'Real run (V13): Qwen2.5-0.5B · one 16,384-token prompt · span mode · numbers measured, not simulated',
             fontsize=9.5, color=t['text2'], va='top')
    ax = fig.add_axes([0.27, 0.2, 0.68, 0.56]); phase_t = fig.text(0.03, 0.8, '', fontsize=12, fontweight='bold', color=t['text'], va='top')
    cap = fig.text(0.03, 0.1, '', fontsize=9.5, color=t['text'], va='top', wrap=True)

    def lane_labels():
        for i, (w, name, key) in enumerate(lanes):
            fig.text(0.03, 0.2 + 0.56 * (1 - (i + 0.5) / len(lanes)), name, fontsize=10, color=t['text'], va='center')
    lane_labels()

    def frame(f):
        ax.clear(); ax.set_facecolor(t['surface']); ax.set_ylim(len(lanes), 0); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(False)
        ax.tick_params(colors=t['text2'], labelsize=8.5, length=0)
        if f < P1:                                   # 1. pre-bench once
            phase_t.set_text('① Measure every device once, at startup')
            cap.set_text('Each device processes one window; the round trip (network included) is timed. Weights stay loaded and\n'
                         'warm, so later requests never pay for measuring again.')
            ax.set_xlim(0, 1); ax.set_xticks([])
            shown = int(f / (P1 / (len(lanes) + 1)))
            for i, (w, name, key) in enumerate(lanes[:shown]):
                ax.add_patch(Rectangle((0.0, i + 0.3), 0.012, 0.4, facecolor=t[key], linewidth=0))
                ax.text(0.03, i + 0.5, f'{rates[w]:,.0f} tok/s', fontsize=13, fontweight='bold', color=t['text'], va='center')
            if shown > len(lanes) - 1:
                ax.text(0.55, 1.0, 'the GPU is ~160× faster\nthan one CPU slot', fontsize=10, color=t['text2'], va='center')
        elif f < P1 + P2:                            # 2. cut the text so everyone finishes together
            k = min(1.0, (f - P1) / (P2 * 0.6))
            phase_t.set_text('② Cut the text so every device finishes at the same moment')
            cap.set_text('Bisection finds the common finish time T; each device gets exactly what it can do by T (+ one 256-token halo).\n'
                         'A CPU slot would need 2.4 s for its smallest piece while the rest finish in 0.8 s → 0 tokens this time, kept warm.')
            ax.set_xlim(0, L_); ax.set_xlabel('position in the prompt (tokens)', color=t['text2'], fontsize=9)
            ax.set_xticks([0, 4096, 8192, 12288, 16384], ['0', '4,096', '8,192', '12,288', '16,384'])
            for i, (w, name, key) in enumerate(lanes):
                if w in spans:
                    a, b = spans[w]; ax.add_patch(Rectangle((a, i + 0.2), (b - a) * k, 0.6, facecolor=t[key], linewidth=0))
                    ax.text(a + (b - a) * k + 150, i + 0.5, f'{b - a:,} tokens', fontsize=9, color=t['text'], va='center')
                else:
                    ax.text(150, i + 0.5, '0 tokens — stays loaded for the next request', fontsize=9, color=t['text2'], va='center')
        else:                                        # 3. run in parallel (real finish times, slowed down)
            g = min(1.0, (f - P1 - P2) / P3); now = g * T_END * 1.02
            phase_t.set_text('③ Run all pieces at the same time')
            ax.set_xlim(0, T_END * 1.25); ax.set_xlabel('time (seconds)', color=t['text2'], fontsize=9)
            for i, (w, name, key) in enumerate(lanes):
                if w not in spans:
                    ax.text(0.01, i + 0.5, 'idle this request (0 tokens)', fontsize=9, color=t['text2'], va='center'); continue
                end = tl[w]; ax.add_patch(Rectangle((0, i + 0.2), min(now, end), 0.6, facecolor=t[key], linewidth=0))
                if now >= end: ax.text(end + 0.01, i + 0.5, f'done {end:.2f} s', fontsize=9, color=t['text'], va='center')
            ax.axvline(now, color=t['text2'], linewidth=1)
            if g >= 1:
                cap.set_text(f'Done in {T_END:.2f} s → {row["tok_s"]:,.0f} tok/s, vs 19,413 tok/s on the GPU alone (+3%). An equal 4-way split would\n'
                             f'wait ≈ 38 s for a CPU slot (projected). The Mac finished {T_END - tl["10.0.0.92:54110"]:.2f} s early: network time is not in the plan yet.')
            else:
                cap.set_text('Each device reuses its halo in the KV cache; the coordinator costs 0.1–0.3% of the time.')
        return []
    FuncAnimation(fig, frame, frames=P1 + P2 + P3 + HOLD_, interval=1000 / FPS_).save(OUT / f'scheduler-{mode}.gif', writer=PillowWriter(fps=FPS_))
    plt.close(fig)


if __name__ == '__main__':
    import sys
    only = sys.argv[1:]
    for mode in THEMES:
        for fn in (hero, how, long_race, scheduler):
            if not only or fn.__name__ in only: fn(mode)
    print('T_trad %.0f ms, T_sp %.0f ms' % (T_TRAD, T_SP), sorted(p.name for p in OUT.glob('*.gif')))
