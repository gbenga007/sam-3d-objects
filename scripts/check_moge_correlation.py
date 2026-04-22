"""
MoGe vs GT metric-scale correlation check on OmniNOCS-NOCS275.

Validates the load-bearing assumption in the Phase 1 architecture: that MoGe's
object-masked pointmap extent correlates with ground-truth metric object size.

If this correlation is weak, MetricScaleHead has no useful metric anchor and
the architecture needs revisiting before any fine-tuning work.

Inputs:
    - OmniNOCS metadata at /mnt/dest/OmniNOCS/omninocs_release_nocs_real275/
    - NOCS-Real275 source RGB at /mnt/dest/OmniNOCS/real_test/
    - MoGe-2 checkpoint at /mnt/source/MoGe/checkpoints/model.pt

Outputs (in --out-dir):
    - pairs.csv: per-object (gt_w, gt_h, gt_d, gt_vol_cbrt, gt_diag, moge_scale_mean,
      moge_diag, moge_vol_cbrt, category)
    - scatter.png: log-log scatter plot of MoGe vol^(1/3) vs GT vol^(1/3), colored by category
    - summary.md: Pearson + Spearman correlations + per-category breakdown

Env needed:
    micromamba activate sam3d-objects
    export PYTHONPATH=/mnt/source/sam-3d-objects:/mnt/source/MoGe:$PYTHONPATH
    export LIDRA_SKIP_INIT=1
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from moge.model.v2 import MoGeModel


OMNINOCS_ROOT = Path("/mnt/dest/OmniNOCS/omninocs_release_nocs_real275")
NOCS_RGB_ROOT = Path("/mnt/dest/OmniNOCS/real_test")  # XXXX_color.png lives under scene_N/
MOGE_CKPT = Path("/mnt/source/MoGe/checkpoints/model.pt")


def load_rgb(image_name: str) -> np.ndarray:
    """image_name like 'nocs_real275/test/scene_1/0066' -> load scene_1/0066_color.png."""
    parts = Path(image_name).parts
    scene = parts[-2]
    frame = parts[-1]
    path = NOCS_RGB_ROOT / scene / f"{frame}_color.png"
    return np.array(Image.open(path).convert("RGB"))


def load_instances(image_name: str) -> np.ndarray:
    """Load the OmniNOCS 16-bit instance map."""
    parts = Path(image_name).parts
    # image_name is 'nocs_real275/test/scene_X/NNNN'
    path = OMNINOCS_ROOT / f"{image_name}_instances.png"
    arr = np.array(Image.open(path))
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr


def run_moge(model: MoGeModel, rgb_uint8: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Run MoGe-2 on an HxWx3 uint8 RGB image, return HxWx3 metric pointmap (numpy)."""
    rgb = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    rgb = rgb.to(device)
    with torch.no_grad():
        out = model.infer(rgb)  # dict with 'points', 'depth', 'mask', ...
    points = out["points"].cpu().numpy()  # [H, W, 3] metric
    valid = out["mask"].cpu().numpy() if "mask" in out else np.ones(points.shape[:2], dtype=bool)
    return points, valid.astype(bool)


def object_scale_stats(points_hw3: np.ndarray, valid: np.ndarray, obj_mask: np.ndarray) -> dict | None:
    """
    Given a metric pointmap and a per-object binary mask, compute scale statistics.
    Returns None if too few valid points.
    """
    combined = valid & obj_mask
    if combined.sum() < 50:
        return None
    pts = points_hw3[combined]  # [N, 3]
    # Robust percentile-based extent (trim outliers from depth noise)
    p_lo, p_hi = np.percentile(pts, [2, 98], axis=0)
    extent = p_hi - p_lo  # [3]
    diag = float(np.linalg.norm(extent))
    vol_cbrt = float(np.cbrt(np.prod(extent))) if np.all(extent > 0) else 0.0
    scale_mean = float(extent.mean())
    return {
        "moge_extent_x": float(extent[0]),
        "moge_extent_y": float(extent[1]),
        "moge_extent_z": float(extent[2]),
        "moge_scale_mean": scale_mean,
        "moge_diag": diag,
        "moge_vol_cbrt": vol_cbrt,
        "n_points": int(combined.sum()),
    }


def gt_stats(size_whd: list[float]) -> dict:
    w, h, d = size_whd
    return {
        "gt_w": w, "gt_h": h, "gt_d": d,
        "gt_diag": float(np.linalg.norm([w, h, d])),
        "gt_vol_cbrt": float(np.cbrt(w * h * d)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=200,
                    help="Number of frames to sample (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metadata", type=str,
                    default=str(OMNINOCS_ROOT / "nocs_real275_train_metadata.json"))
    ap.add_argument("--out-dir", type=str,
                    default="/mnt/dest/OmniNOCS/correlation_check")
    ap.add_argument("--paint-objects", action="store_true",
                    help="Fill all segmented object pixels with neutral gray (0.5,0.5,0.5) "
                         "before MoGe inference. Intended to test whether transparency is the "
                         "cause of poor correlation for bottle/cup categories.")
    ap.add_argument("--categories", type=str, default="",
                    help="Comma-separated category filter, e.g. 'bottle,cup'. Empty = all.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metadata) as f:
        meta = json.load(f)

    category_filter = set(c.strip() for c in args.categories.split(",") if c.strip())

    rng = random.Random(args.seed)
    if args.n_frames and args.n_frames < len(meta):
        meta = rng.sample(meta, args.n_frames)

    print(f"Loading MoGe-2 from {MOGE_CKPT}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(str(MOGE_CKPT)).to(device).eval()
    # MoGe is typically FP32 for accuracy; switch to FP16 if speed matters
    print(f"MoGe loaded on {device}")

    rows = []
    skipped_noimg = skipped_nomoge = skipped_toosmall = 0

    for frame in tqdm(meta, desc="frames"):
        image_name = frame["image_name"]
        try:
            rgb = load_rgb(image_name)
            instances = load_instances(image_name)
        except FileNotFoundError:
            skipped_noimg += 1
            continue

        # Optionally paint all object pixels with neutral gray before MoGe.
        # Rationale: transparent/translucent objects (bottles, cups) confuse depth
        # networks. Painting them opaque gives MoGe a real surface to estimate.
        moge_input = rgb.copy()
        if args.paint_objects:
            all_obj_mask = instances > 0
            moge_input[all_obj_mask] = 128  # neutral gray in uint8

        try:
            points, valid = run_moge(model, moge_input, device=device)
        except Exception as e:
            skipped_nomoge += 1
            continue

        for obj in frame["objects"]:
            if category_filter and obj["category"] not in category_filter:
                continue
            obj_id = obj["object_id"]
            obj_mask = (instances == obj_id)
            if obj_mask.sum() < 100:
                skipped_toosmall += 1
                continue
            stats = object_scale_stats(points, valid, obj_mask)
            if stats is None:
                skipped_toosmall += 1
                continue
            row = {
                "image_name": image_name,
                "object_id": obj_id,
                "category": obj["category"],
                **gt_stats(obj["size"]),
                **stats,
            }
            rows.append(row)

    if not rows:
        print("ERROR: no rows collected. Check paths / metadata / RGB layout.")
        return

    csv_path = out_dir / "pairs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pairs to {csv_path}")
    print(f"Skipped: no_image={skipped_noimg}, moge_fail={skipped_nomoge}, too_small={skipped_toosmall}")

    # Correlations (log-log)
    gt = np.array([r["gt_vol_cbrt"] for r in rows])
    moge_v = np.array([r["moge_vol_cbrt"] for r in rows])
    moge_d = np.array([r["moge_diag"] for r in rows])
    moge_m = np.array([r["moge_scale_mean"] for r in rows])
    gt_d = np.array([r["gt_diag"] for r in rows])

    # Filter positives for log
    mask = (gt > 0) & (moge_v > 0) & (moge_d > 0) & (moge_m > 0)
    lg = np.log(gt[mask]); lv = np.log(moge_v[mask]); ld = np.log(moge_d[mask]); lm = np.log(moge_m[mask])
    lg_d = np.log(gt_d[mask])

    summary_lines = [
        "# MoGe vs GT scale correlation — NOCS275",
        "",
        f"N samples (object instances): {len(rows)}",
        f"Valid (>0) for log: {mask.sum()}",
        "",
        "## Log-log Pearson correlations",
        f"- log(MoGe_vol_cbrt) vs log(GT_vol_cbrt): r = {pearsonr(lv, lg).statistic:.3f}",
        f"- log(MoGe_diag)     vs log(GT_vol_cbrt): r = {pearsonr(ld, lg).statistic:.3f}",
        f"- log(MoGe_scale_mean) vs log(GT_vol_cbrt): r = {pearsonr(lm, lg).statistic:.3f}",
        f"- log(MoGe_diag)     vs log(GT_diag):     r = {pearsonr(ld, lg_d).statistic:.3f}",
        "",
        "## Spearman (rank) correlations",
        f"- MoGe_vol_cbrt vs GT_vol_cbrt: rho = {spearmanr(moge_v[mask], gt[mask]).statistic:.3f}",
        f"- MoGe_diag     vs GT_diag:     rho = {spearmanr(moge_d[mask], gt_d[mask]).statistic:.3f}",
        "",
        "## Per-category Pearson on log(vol_cbrt)",
    ]
    by_cat = defaultdict(list)
    for r in rows:
        if r["gt_vol_cbrt"] > 0 and r["moge_vol_cbrt"] > 0:
            by_cat[r["category"]].append((r["gt_vol_cbrt"], r["moge_vol_cbrt"]))
    for cat, pairs in sorted(by_cat.items()):
        g = np.log([p[0] for p in pairs]); v = np.log([p[1] for p in pairs])
        if len(g) >= 3:
            r_val = pearsonr(v, g).statistic
        else:
            r_val = float("nan")
        mean_gt = np.mean([p[0] for p in pairs])
        mean_moge = np.mean([p[1] for p in pairs])
        summary_lines.append(f"- {cat:<10} n={len(pairs):<4} r={r_val:+.3f}  mean_GT={mean_gt:.3f}m  mean_MoGe={mean_moge:.3f}m  ratio={mean_moge/mean_gt:.2f}")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"Wrote summary to {summary_path}")
    print("\n".join(summary_lines))

    # Scatter plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    cats = sorted({r["category"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(cats):
        pts = [(r["gt_vol_cbrt"], r["moge_vol_cbrt"]) for r in rows if r["category"] == cat and r["gt_vol_cbrt"] > 0 and r["moge_vol_cbrt"] > 0]
        if pts:
            g = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts])
            ax[0].scatter(g, m, s=8, alpha=0.5, label=cat, color=cmap(i))
    lims = [min(gt.min(), moge_v.min()) * 0.5, max(gt.max(), moge_v.max()) * 2]
    ax[0].plot(lims, lims, "k--", lw=0.7, label="y=x")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("GT volume^(1/3)  [m]")
    ax[0].set_ylabel("MoGe volume^(1/3)  [m]")
    ax[0].set_title("Log-log: MoGe vs GT characteristic length")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(True, which="both", ls=":", alpha=0.4)

    for i, cat in enumerate(cats):
        pts = [(r["gt_diag"], r["moge_diag"]) for r in rows if r["category"] == cat and r["gt_diag"] > 0 and r["moge_diag"] > 0]
        if pts:
            g = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts])
            ax[1].scatter(g, m, s=8, alpha=0.5, label=cat, color=cmap(i))
    lims = [min(gt_d.min(), moge_d.min()) * 0.5, max(gt_d.max(), moge_d.max()) * 2]
    ax[1].plot(lims, lims, "k--", lw=0.7, label="y=x")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("GT 3D diagonal  [m]")
    ax[1].set_ylabel("MoGe 3D diagonal  [m]")
    ax[1].set_title("Log-log: MoGe vs GT 3D diagonal")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(f"MoGe-2 vs NOCS-Real275 GT scale  (N={len(rows)})")
    fig.tight_layout()
    scatter_path = out_dir / "scatter.png"
    fig.savefig(scatter_path, dpi=120)
    print(f"Wrote scatter plot to {scatter_path}")


if __name__ == "__main__":
    main()
