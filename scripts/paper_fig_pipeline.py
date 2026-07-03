#!/usr/bin/env python
"""Paper method/pipeline figure, styled after sample_pipeline_figure.png:
grey stage panels with title chips, real artifacts as flow nodes, and
frozen/trained markers on every learnable block.

Flow: RGB+mask -> [metric anchor: MoGe-2 pointmap, frozen, swappable]
              -> [geometry MoT: shape branch frozen -> canonical shape c;
                  layout branch trained -> SSI ratio + translation;
                  MetricScaleHead trained -> scale token]
              -> [SLAT: cross-attn trained, rest frozen; MetricScaleDecoder]
              -> metric W,H,D + 3D box + metric-scaled mesh.

  python scripts/paper_fig_pipeline.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

sys.path.insert(0, "/mnt/source/sam-3d-objects/scripts")
import paper_fig_qualitative as pq  # box-overlay renderer (frames verified)

INK = "#16203a"
MUTED = "#5b6677"
BLUE = "#0072B2"
ORANGE = "#E69F00"
PANEL = "#ececf0"
CHIP = "#3a3f4b"
FROZEN_BG, FROZEN_FG = "#e3ecf7", "#2b5c8a"
TRAIN_BG, TRAIN_FG = "#fdeeda", "#b45309"

REAL = "/mnt/source/datasets_sam3d/OmniNOCS/real_test"
PM_DIR = "/mnt/source/sam-3d-objects/artifacts/metric_scale/moge2_pointmaps"
FRAME = "scene_1/0000"
OUT = "/mnt/source/paper-template/figures/pipeline.pdf"


def load_assets():
    fp = os.path.join(REAL, FRAME)
    rgb = cv2.imread(fp + "_color.png")[:, :, ::-1]
    mask_im = cv2.imread(fp + "_mask.png")[:, :, 2]
    # mask overlay: dim background, tint object regions
    inst = (mask_im != 255)
    over = rgb.astype(np.float32) * 0.45
    over[inst] = rgb[inst].astype(np.float32) * 0.55 + np.array([230, 159, 0]) * 0.45
    over = over.clip(0, 255).astype(np.uint8)
    # MoGe-2 pointmap -> turbo depth colouring
    pm = np.load(os.path.join(PM_DIR, "nocs_real275__test__scene_1__0000.npy")).astype(np.float32)
    z = pm[..., 2]
    zv = z[np.isfinite(z)]
    zn = (z - np.percentile(zv, 2)) / max(np.percentile(zv, 98) - np.percentile(zv, 2), 1e-6)
    depth_vis = plt.get_cmap("turbo")(1.0 - zn.clip(0, 1))[..., :3]
    depth_vis[~np.isfinite(z)] = 1.0
    # canonical shape node: NOCS coord map (normalized-coordinate colours), masked
    coord = cv2.imread(fp + "_coord.png")[:, :, ::-1].astype(np.float32) / 255.0
    coord_vis = np.ones_like(coord)
    coord_vis[inst] = coord[inst]
    return rgb, over, depth_vis, coord_vis


def output_crop():
    recs = [json.loads(l) for l in open(
        "/mnt/source/sam-3d-objects/artifacts/nocs_map_step05_joint_mot_full/pred_boxes.jsonl")]
    best = {}
    for r in sorted(recs, key=lambda r: r["trans_err"]):
        out = pq.render_instance(r, pad=55)
        if out is None:
            continue
        crop, cls, _ = out
        if cls not in best:
            best[cls] = crop
        if len(best) >= 2 and "bowl" in best:
            break
    return best.get("bowl", next(iter(best.values())))


def panel(ax, x, y, w, h, title, tw=None):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=PANEL, ec="none", zorder=1,
                               joinstyle="round"))
    tw = tw or (0.145 * len(title) + 0.25)
    ax.add_patch(plt.Rectangle((x + w / 2 - tw / 2, y + h - 0.16), tw, 0.34,
                               fc=CHIP, ec="none", zorder=3))
    ax.text(x + w / 2, y + h + 0.01, title, ha="center", va="center", zorder=4,
            fontsize=9, color="white", fontweight="bold")


def chip(ax, x, y, text, trained, fs=6.8):
    bg, fg = (TRAIN_BG, TRAIN_FG) if trained else (FROZEN_BG, FROZEN_FG)
    mark = "▲ trained" if trained else "❄ frozen"
    ax.text(x, y, f"{text}", ha="center", va="center", fontsize=fs, color=INK,
            zorder=5, bbox=dict(boxstyle="round,pad=0.32", fc=bg, ec=fg, lw=1.0))
    ax.text(x, y - 0.30, mark, ha="center", va="center", fontsize=5.6,
            color=fg, zorder=5, fontweight="bold")


def img_node(ax, img, x, y, w, h, label=None, ec=MUTED):
    ax.imshow(img, extent=(x, x + w, y, y + h), zorder=2, aspect="auto")
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=ec, lw=0.9, zorder=3))
    if label:
        ax.text(x + w / 2, y - 0.16, label, ha="center", va="top", fontsize=6.6,
                color=MUTED, zorder=4)


def arrow(ax, p, q, color=INK, lw=1.4, z=4):
    ax.annotate("", xy=q, xytext=p, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=2, shrinkB=2, mutation_scale=11))


def main():
    rgb, over, depth_vis, coord_vis = load_assets()
    outcrop = output_crop()

    fig, ax = plt.subplots(figsize=(10.2, 4.1))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0.15, 6.15)
    ax.axis("off")

    # ---- Panel: Input ----
    panel(ax, 0.15, 0.5, 2.1, 5.1, "Input")
    img_node(ax, rgb, 0.38, 3.45, 1.64, 1.55, "RGB image")
    img_node(ax, over, 0.38, 1.15, 1.64, 1.55, "instance mask")

    # ---- Panel: Metric anchor (bottom path) ----
    panel(ax, 2.75, 0.5, 3.5, 2.25, "Metric anchor")
    img_node(ax, depth_vis, 2.95, 0.95, 1.7, 1.35, None)
    chip(ax, 5.45, 2.0, "MoGe-2", trained=False)
    ax.text(5.45, 1.30, "metric pointmap $\\mathbf{P}$\n$s_{\\mathrm{scene}}$: units live here",
            ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.35)
    ax.text(4.5, 0.66, "swappable: MoGe-v1 (affine) / MoGe-2 / sensor",
            ha="center", fontsize=5.8, color=ORANGE, style="italic")

    # ---- Panel: Geometry model (MoT) ----
    panel(ax, 2.75, 3.1, 5.0, 2.5, "Geometry model (MoT)")
    chip(ax, 3.75, 4.85, "shape branch", trained=False)
    img_node(ax, coord_vis, 4.9, 4.42, 1.3, 0.95, None)
    ax.text(5.55, 4.28, "canonical shape, extents $\\mathbf{c}$", ha="center",
            va="top", fontsize=6.2, color=MUTED, zorder=5)
    chip(ax, 3.75, 3.65, "layout branch", trained=True)
    ax.text(5.9, 3.85, "SSI ratio $\\tilde{\\mathbf{s}}$, translation\n(dimensionless)",
            ha="center", va="center", fontsize=6.6, color=BLUE, linespacing=1.3)
    chip(ax, 7.0, 4.85, "MetricScaleHead", trained=True)

    # ---- Panel: SLAT refinement ----
    panel(ax, 8.15, 1.7, 3.0, 3.9, "SLAT refinement")
    chip(ax, 9.65, 4.75, "cross-attention", trained=True)
    chip(ax, 9.65, 3.85, "self-attn + FFN", trained=False)
    ax.text(9.65, 3.15, "scale token injected\ninto conditioning", ha="center",
            va="center", fontsize=6.4, color=MUTED, linespacing=1.3)
    chip(ax, 9.65, 2.35, "MetricScaleDecoder", trained=True, fs=6.4)

    # ---- Panel: Output ----
    panel(ax, 11.7, 0.5, 2.55, 5.1, "Metric output")
    ch, cw = outcrop.shape[:2]
    bw, bh = 2.15, 2.0
    # fit crop into box preserving aspect (fig data-units: x stretched ~1.55x vs y)
    disp_ar = (cw / ch) / 1.55
    if disp_ar > bw / bh:
        w2, h2 = bw, bw / disp_ar
    else:
        w2, h2 = bh * disp_ar, bh
    img_node(ax, outcrop, 13.0 - w2 / 2, 3.0 + (bh - h2) / 2, w2, h2,
             "metric 3D box (vs GT)", ec=ORANGE)
    ax.text(13.0, 2.35, "$\\mathbf{d}=[W,H,D]$ metres", ha="center", fontsize=7.2,
            color=INK, fontweight="bold")
    ax.text(13.0, 1.85, "mesh $\\times$ metric scale\n+ pose $(R, t)$", ha="center",
            va="center", fontsize=6.6, color=MUTED, linespacing=1.35)
    ax.text(13.0, 1.05, "$\\mathbf{d}=\\mathbf{c}\\odot\\tilde{\\mathbf{s}}\\cdot s_{\\mathrm{scene}}$",
            ha="center", fontsize=7.4, color=INK)

    # ---- Arrows ----
    arrow(ax, (2.28, 4.2), (2.72, 4.3), color=MUTED)          # input -> MoT
    arrow(ax, (2.28, 1.9), (2.72, 1.75), color=MUTED)         # input -> anchor
    arrow(ax, (3.3, 2.78), (3.3, 3.07), color=ORANGE)          # anchor -> MoT (conditioning)
    ax.text(3.42, 2.9, "conditions", ha="left", fontsize=5.8, color=ORANGE, zorder=6)
    arrow(ax, (6.3, 1.6), (12.4, 1.6), color=ORANGE, lw=1.6)  # anchor -> output (units)
    ax.text(9.3, 1.32, "$s_{\\mathrm{scene}}$ lifts SSI $\\to$ metric (delegation)",
            ha="center", fontsize=6.4, color=ORANGE)
    arrow(ax, (7.82, 4.85), (8.85, 4.78), color=BLUE)         # head -> SLAT (token)
    ax.text(8.3, 5.06, "scale token", ha="center", fontsize=5.8, color=BLUE)
    ax.annotate("", xy=(12.3, 1.95), xytext=(7.5, 3.35), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2,
                                connectionstyle="arc3,rad=0.32",
                                shrinkA=2, shrinkB=2, mutation_scale=11))
    ax.text(8.0, 1.05, "pose $(R,t)$", ha="center", fontsize=6.0, color=BLUE)
    arrow(ax, (11.18, 2.35), (12.0, 2.35), color=INK)         # decoder -> W,H,D

    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.02)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=165)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
