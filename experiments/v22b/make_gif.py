"""Create compact, editorial-style V22b README animations from published measurements."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parents[2] / "docs" / "img"
BG, TEXT, MUTED, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#e9e8e4"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = [BLUE, ORANGE, GREEN]


def canvas(title, subtitle):
    fig = plt.figure(figsize=(8.4, 4.9), dpi=90)
    fig.patch.set_facecolor(BG)
    fig.text(.035, .95, title, color=TEXT, fontsize=16, fontweight="bold", va="top")
    fig.text(.035, .885, subtitle, color=MUTED, fontsize=9.5, va="top")
    return fig


def card(ax, label, value, note, color):
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=.05",
                                facecolor="white", edgecolor=EMPTY, linewidth=1.1))
    ax.add_patch(Rectangle((0, 0), .025, 1, facecolor=color, linewidth=0))
    ax.text(.08, .76, label, color=TEXT, fontsize=10.5, fontweight="bold", va="center")
    ax.text(.08, .47, value, color=color, fontsize=17, fontweight="bold", va="center")
    ax.text(.08, .18, note, color=MUTED, fontsize=8.5, va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def sampling():
    fig = canvas("One local runtime. Three different computers.",
                 "Same Qwen model family · each device tested on its own · median greedy decode speed")
    cards = []
    for i, (name, speed, note) in enumerate([
        ("Arc B580", "264 tok/s", "Windows · Vulkan"),
        ("Radeon RX 570", "116.6 tok/s", "Linux · RADV/Vulkan"),
        ("Apple M4", "64.9 tok/s", "macOS · MoltenVK"),
    ]):
        ax = fig.add_axes([.035 + i * .32, .35, .29, .32])
        card(ax, name, speed, note, COLORS[i]); cards.append(ax)
    dots = []
    for i, ax in enumerate(cards):
        dots.append(fig.text(.18 + i * .32, .73, "●", color=EMPTY, fontsize=19, ha="center"))
    footer = fig.text(.035, .17, "Next: connect all three and test them as one request pool.",
                      fontsize=11, color=TEXT, fontweight="bold")
    fig.text(.035, .08, "Solo-device measurements do not show combined speed. The three-device test is still ahead.",
             fontsize=8.5, color=MUTED)

    def draw(frame):
        for i, dot in enumerate(dots):
            dot.set_color(COLORS[i] if frame == i else EMPTY)
        footer.set_text("Next: connect all three and test them as one request pool." if frame < 3 else
                        "Three devices tested separately; combined performance still to be measured.")
        return dots + [footer]

    anim = FuncAnimation(fig, draw, frames=4, interval=1100, blit=False, repeat=True)
    anim.save(OUT / "v22b-sampling.gif", writer=PillowWriter(fps=1))
    plt.close(fig)


def fleet():
    fig = canvas("More local requests, finished sooner.",
                 "An early two-device test · 24 separate requests · 128 tokens each · same output tokens")
    ax1 = fig.add_axes([.04, .43, .4, .29]); card(ax1, "B580 alone", "11.2 s", "p99 from burst arrival", BLUE)
    ax2 = fig.add_axes([.56, .43, .4, .29]); card(ax2, "B580 + RX 570", "8.1 s", "p99 · 1.38× aggregate throughput", GREEN)
    fig.text(.5, .56, "→", color=TEXT, fontsize=22, ha="center", va="center")
    left, right = [], []
    for i in range(24):
        x = .05 + (i % 12) * .031
        y = .34 - (i // 12) * .045
        p = Rectangle((x, y), .024, .027, transform=fig.transFigure, facecolor=EMPTY, linewidth=0)
        fig.patches.append(p); left.append(p)
        q = Rectangle((.56 + (i % 12) * .031, y), .024, .027, transform=fig.transFigure,
                      facecolor=EMPTY, linewidth=0)
        fig.patches.append(q); right.append(q)
    status = fig.text(.04, .17, "Requests are independent: the pool handles more at once; it does not split one answer across GPUs.",
                      fontsize=8.4, color=MUTED)
    fig.text(.04, .08, "B580 + RX 570 was tested. Adding the M4—and measuring network overhead—is the next experiment.",
             fontsize=8.5, color=TEXT)

    def draw(frame):
        # Illustrate work completing on two workers while keeping the 24-request batch visible.
        for i, rect in enumerate(left):
            rect.set_facecolor(BLUE if i < min(24, frame * 4) else EMPTY)
        for i, rect in enumerate(right):
            rect.set_facecolor((BLUE if i < 17 else ORANGE) if i < min(24, frame * 6) else EMPTY)
        status.set_text("Independent requests share the work; slower workers can increase an individual request's wait." if frame >= 4 else
                        "Each small tile represents one independent local request.")
        return left + right + [status]

    anim = FuncAnimation(fig, draw, frames=7, interval=850, blit=False, repeat=True)
    anim.save(OUT / "v22b-fleet.gif", writer=PillowWriter(fps=1.25))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sampling()
    fleet()
    print(f"Wrote V22b animations to {OUT}")


if __name__ == "__main__":
    main()
