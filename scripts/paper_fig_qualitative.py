#!/usr/bin/env python
"""Paper figure: predicted vs GT 3D box overlays on NOCS-Real275 (CPU-only).

Reads the mAP eval's cached predictions (pred_boxes.jsonl from
scripts/nocs_map_step05_predpose.py) and recomputes GT boxes with the official
NOCS alignment, then draws projected wireframes: GT (white, dashed feel via
thin double line) and prediction (orange) on the RGB frame, cropped around the
object. Assembles a grid figure for the paper.

  python scripts/paper_fig_qualitative.py \
      --pred artifacts/nocs_map_step05_live_v2/pred_boxes.jsonl \
      --out  /mnt/source/paper-template/figures/qualitative.pdf
"""
import argparse
import json
import os
import sys
import types

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
sys.modules.setdefault("ICP", types.ModuleType("ICP"))
NOCS_REPO = "/mnt/source/NOCS_CVPR2019"
sys.path.insert(0, NOCS_REPO)
import utils as nocs_utils  # noqa: E402

sys.path.insert(0, "/mnt/source/sam-3d-objects/scripts")
import nocs_map_step0_oracle as step0  # noqa: E402

INTRINSICS = step0.INTRINSICS
REAL_TEST = step0.REAL_TEST
SYNSET_NAMES = step0.SYNSET_NAMES

# GT = white (achromatic, safe against any overlay hue); prediction = orange
# (Okabe-Ito #E69F00) — high contrast on indoor imagery, CVD-safe against white.
COL_GT = (255, 255, 255)
COL_PRED = (230, 159, 0)


def project(pts3, K, flip):
    """[3,N] points -> [N,2] pixels. flip=True for the pipeline's [-X,-Y,Z]
    frame (cached pred_RT, stored un-rotated); flip=False for plain OpenCV
    (gt_RT from nocs_utils.align). Verified per-instance against mask centroids."""
    if flip:
        x_cv, y_cv, z = -pts3[0], -pts3[1], pts3[2]
    else:
        x_cv, y_cv, z = pts3[0], pts3[1], pts3[2]
    u = K[0, 0] * x_cv / z + K[0, 2]
    v = K[1, 1] * y_cv / z + K[1, 2]
    return np.stack([u, v], axis=1)


def box_edges(corners8):
    """Pairs of corner indices that differ in exactly one axis of the canonical box."""
    c = corners8.T  # [8,3]
    signs = np.sign(c - c.mean(0))
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if (signs[i] != signs[j]).sum() == 1:
                edges.append((i, j))
    return edges


def draw_box(img, RT, scales, K, color, thickness=2, flip=False):
    corners = nocs_utils.get_3d_bbox(np.asarray(scales, dtype=np.float64))  # [3,8]
    cam = nocs_utils.transform_coordinates_3d(corners, np.asarray(RT))
    if (cam[2] <= 0.05).any():
        return None
    px = project(cam, K, flip)
    for i, j in box_edges(corners):
        p, q = tuple(np.round(px[i]).astype(int)), tuple(np.round(px[j]).astype(int))
        cv2.line(img, p, q, color, thickness, cv2.LINE_AA)
    return px


def gt_for_frame(fp):
    parsed = step0.load_gt_instances(fp)
    if parsed is None:
        return None
    gt_mask, gt_coord, gt_class_ids = parsed
    depth16 = step0.load_depth(fp)
    if depth16 is None:
        return None
    gt_RTs, gt_scales, _, _ = nocs_utils.align(
        gt_class_ids, gt_mask, gt_coord, depth16, INTRINSICS, SYNSET_NAMES, fp, None)
    return gt_RTs, gt_scales, gt_class_ids, gt_mask


def render_instance(rec, pad=70):
    frame_rel, i = rec["key"].split("#")
    i = int(i)
    fp = os.path.join(REAL_TEST, frame_rel)
    gt = gt_for_frame(fp)
    if gt is None:
        return None
    gt_RTs, gt_scales, gt_class_ids, gt_mask = gt
    if i >= len(gt_class_ids):
        return None
    img = cv2.imread(fp + "_color.png")[:, :, ::-1].copy()
    px_gt = draw_box(img, gt_RTs[i], gt_scales[i], INTRINSICS, COL_GT, 2, flip=False)
    px_pr = draw_box(img, np.asarray(rec["pred_RT"]), rec["pred_scales"],
                     INTRINSICS, COL_PRED, 2, flip=True)
    if px_gt is None or px_pr is None:
        return None
    allpx = np.concatenate([px_gt, px_pr], axis=0)
    x0, y0 = np.floor(allpx.min(0)).astype(int) - pad
    x1, y1 = np.ceil(allpx.max(0)).astype(int) + pad
    H, W = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    cls = SYNSET_NAMES[gt_class_ids[i]]
    return crop, cls, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="artifacts/nocs_map_step05_live_v2/pred_boxes.jsonl")
    ap.add_argument("--out", default="/mnt/source/paper-template/figures/qualitative.pdf")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="explicit instance keys; default = auto-pick per class")
    ap.add_argument("--n-good", type=int, default=6)
    ap.add_argument("--n-fail", type=int, default=2)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.pred) if l.strip()]
    print(f"{len(recs)} cached predictions")

    if args.keys:
        chosen = [r for r in recs if r["key"] in set(args.keys)]
    else:
        # auto-pick: best trans_err per class for the "good" panels, then the
        # worst laptop / worst overall as failure panels.
        byclass = {}
        rendered = {}
        for r in sorted(recs, key=lambda r: r["trans_err"]):
            out = render_instance(r)
            if out is None:
                continue
            rendered[r["key"]] = out
            cls = out[1]
            byclass.setdefault(cls, []).append(r)
        chosen, seen = [], set()
        for cls in ["bottle", "bowl", "camera", "can", "mug", "laptop"]:
            if cls in byclass and len(chosen) < args.n_good:
                r = byclass[cls][0]
                chosen.append(r); seen.add(r["key"])
        fails = sorted([r for r in recs if r["key"] in rendered and r["key"] not in seen],
                       key=lambda r: -r["trans_err"])
        chosen += fails[:args.n_fail]
        rendered_cache = rendered
    if not chosen:
        sys.exit("nothing to draw")

    n = len(chosen)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, chosen):
        out = rendered_cache.get(r["key"]) if not args.keys else render_instance(r)
        if out is None:
            continue
        crop, cls, _ = out
        ax.imshow(crop)
        ax.set_title(f"{cls}  ({r['trans_err']*100:.1f} cm)", fontsize=9)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    fig.savefig(args.out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out} ({n} panels)")


if __name__ == "__main__":
    main()
