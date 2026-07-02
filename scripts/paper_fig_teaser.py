#!/usr/bin/env python
"""Paper Figure 1 (teaser): the delegation mechanism + the evidence ladder.

Left: predicted metric size = (dimensionless SSI ratio from the frozen generator)
x (scene scale of the conditioning pointmap) -- the anchor, not the model,
carries the units. The swappable anchors are annotated with the measured median
isotropic size error of the SAME frozen model (Table 1).
Right: the recipe ladder (Table 2) with the oracle iso-correction floor.

  python scripts/paper_fig_teaser.py --out /mnt/source/paper-template/figures/teaser.pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#1a1a1a"
MUTED = "#6b6b6b"
BLUE = "#0072B2"      # Okabe-Ito blue  — model / ours
ORANGE = "#E69F00"    # Okabe-Ito orange — anchor
GRID = "#d9d9d9"
IMG = "/mnt/source/datasets_sam3d/OmniNOCS/real_test/scene_1/0032_color.png"


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, tc=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.3)


def arrow(ax, p, q, color=INK, lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=color, lw=lw,
                                 shrinkA=3, shrinkB=3, mutation_scale=11))


def mechanism_panel(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 7)
    ax.set_aspect("auto")
    ax.axis("off")

    # input image (extent pre-squeezed in x to counter the panel's ~1.7x
    # horizontal stretch so the photo displays undistorted)
    img = mpimg.imread(IMG)
    h, w = img.shape[:2]
    crop = img[0:int(h * 0.95), int(w * 0.12):int(w * 0.88)]
    ax.imshow(crop, extent=(0.10, 1.25, 2.80, 4.60), aspect="auto", zorder=2)
    ax.add_patch(plt.Rectangle((0.10, 2.80), 1.15, 1.80, fill=False,
                               ec=MUTED, lw=0.8, zorder=3))
    ax.text(0.675, 2.42, "RGB image\n+ mask", ha="center", va="top",
            fontsize=7.5, color=MUTED, linespacing=1.2)

    # top path: frozen generator -> dimensionless ratio
    box(ax, 2.0, 5.0, 2.3, 1.35, "frozen generator\n(shape + layout)",
        "#eef3fa", BLUE)
    arrow(ax, (1.30, 4.30), (1.95, 5.40), color=MUTED)
    arrow(ax, (4.40, 5.68), (6.55, 5.68), color=BLUE)
    ax.text(5.48, 6.18, r"dimensionless ratio $\tilde{\mathbf{s}}$",
            ha="center", fontsize=8.2, color=BLUE)

    # bottom path: anchor -> scene scale
    box(ax, 2.0, 1.80, 2.3, 1.35, "depth anchor:\npointmap $\\mathbf{P}$",
        "#fdf3e3", ORANGE)
    arrow(ax, (1.30, 3.15), (1.95, 2.55), color=MUTED)
    arrow(ax, (4.40, 2.48), (6.55, 2.48), color=ORANGE)
    ax.text(5.48, 2.95, r"metric scene scale $s_{\mathrm{scene}}$",
            ha="center", fontsize=8.2, color=ORANGE)
    ax.text(5.48, 2.02, "the units live here", ha="center", fontsize=7.2,
            color=MUTED)

    # product node and output
    ax.add_patch(plt.Circle((7.0, 4.08), 0.22, fc="white", ec=INK, lw=1.2))
    ax.text(7.0, 4.08, r"$\times$", ha="center", va="center", fontsize=10,
            color=INK)
    arrow(ax, (6.68, 5.60), (6.90, 4.40), color=BLUE)
    arrow(ax, (6.68, 2.56), (6.90, 3.76), color=ORANGE)
    arrow(ax, (7.25, 4.08), (7.85, 4.08), color=INK)
    ax.text(8.85, 4.52, "metric size", ha="center", fontsize=8.6, color=INK,
            fontweight="bold")
    ax.text(8.85, 3.82,
            r"$\mathbf{d}=\mathbf{c}\odot\tilde{\mathbf{s}}\cdot s_{\mathrm{scene}}$",
            ha="center", fontsize=8.6, color=INK)

    # swappable-anchor strip (fixed columns)
    ax.plot([2.0, 9.9], [1.30, 1.30], color=GRID, lw=0.8)
    ax.text(2.0, 0.92, "swap the anchor $\\Rightarrow$ swap the units.  "
                       "Same frozen model, median size error:",
            fontsize=7.4, color=MUTED, va="center")
    cols = [(2.0, "MoGe (affine)", "75.2%", MUTED),
            (4.9, "MoGe-2 (metric)", "29.2%", ORANGE),
            (7.6, "GT depth", "6.7%", INK)]
    for x, name, err, c in cols:
        ax.text(x, 0.40, name, fontsize=7.6, color=MUTED, va="center")
        ax.text(x, -0.10, err, fontsize=9.5, color=c, va="center",
                fontweight="bold")


def ladder_panel(ax):
    labels = ["affine anchor (MoGe)", "+ metric anchor (MoGe-2)",
              "+ layout SFT (full recipe)"]
    vals = [22.3, 20.2, 17.6]
    colors = ["#9dbfdd", "#5b9ec9", BLUE]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.58, color=colors, edgecolor="none")
    for yi, v, lab in zip(y, vals, labels):
        ax.text(0.4, yi + 0.44, lab, va="center", ha="left", fontsize=8,
                color=INK)
        ax.text(v + 0.4, yi, f"{v:.1f}%", va="center", fontsize=8.8,
                color=INK, fontweight="bold")
    ax.axvline(15, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.text(14.5, -0.72, "oracle floor $\\approx$15%", fontsize=7.4,
            color=MUTED, ha="right")
    ax.set_yticks([])
    ax.set_ylim(-0.95, 2.95)
    ax.set_xlim(0, 26.5)
    ax.set_xlabel("mean per-axis dimension error (%)  $\\downarrow$",
                  fontsize=8.2, color=INK)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/source/paper-template/figures/teaser.pdf")
    args = ap.parse_args()

    fig = plt.figure(figsize=(9.6, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0],
                          left=0.005, right=0.99, top=0.99, bottom=0.15,
                          wspace=0.10)
    mechanism_panel(fig.add_subplot(gs[0]))
    ladder_panel(fig.add_subplot(gs[1]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=170)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
