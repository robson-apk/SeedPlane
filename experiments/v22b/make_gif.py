"""Create a small README animation from the published V22b per-device ratios."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


OUT = Path(__file__).resolve().parents[2] / "docs" / "img" / "v22b-sampling.gif"
DEVICES = ["Intel Arc B580", "Apple M4", "AMD Radeon RX 570"]
MODES = ["T=1", "k40 + p0.9", "p=0.9", "p=0.99", "k=200"]
RATIOS = [
    [1.0040, 0.9743, 0.9832, 0.9760, 0.9727],
    [1.0270, 1.0328, 1.0134, 1.0131, 1.0156],
    [0.9932, 0.9777, 0.9940, 0.9924, 0.9829],
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fbfaf7")
    fig.suptitle("V22b · Sampling decode speed vs greedy", fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.89, "Median of three rounds · each GPU tested locally · 1.00× = greedy", ha="center", fontsize=9)
    bars = []
    for ax, device, ratios in zip(axes, DEVICES, RATIOS):
        ax.set_facecolor("#fbfaf7")
        ax.axvline(1.0, color="#202a35", lw=1.2)
        ax.axvline(0.95, color="#b54b37", lw=1, ls="--")
        ax.set_xlim(0.92, 1.07)
        ax.set_title(device, fontsize=10, fontweight="bold", pad=8)
        ax.set_yticks(range(len(MODES)), MODES, fontsize=8)
        ax.grid(axis="x", color="#e6e2dc", lw=0.7)
        ax.set_axisbelow(True)
        rects = ax.barh(range(len(MODES)), [r - 0.92 for r in ratios], left=0.92,
                        color=["#a8b4bf"] * len(MODES), height=0.62)
        for rect, ratio in zip(rects, ratios):
            ax.text(ratio + 0.001, rect.get_y() + rect.get_height() / 2,
                    f"{ratio:.3f}×", va="center", fontsize=7.5, color="#202a35")
        bars.append(list(rects))
    axes[0].invert_yaxis()
    fig.text(0.5, 0.04, "Dashed line: pre-registered 0.95× floor. This is not a multi-GPU speedup test.",
             ha="center", fontsize=8.5, color="#4c5965")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.16, wspace=0.28)

    def frame(i):
        for group in bars:
            for j, rect in enumerate(group):
                rect.set_color("#2a78d6" if j == i else "#a8b4bf")
        return [rect for group in bars for rect in group]

    animation = FuncAnimation(fig, frame, frames=len(MODES), interval=900, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    animation.save(OUT, writer=PillowWriter(fps=1.25))
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
