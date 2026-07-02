#!/usr/bin/env python
"""
MoGe-2 depth-bridge oracle.

Tests whether PER-INSTANCE metric depth (MoGe-2) recovers object metric size better
than the category prior the trained MetricScaleHead has maxed out at. See
[[project_translation_scale_oracle]]: the head sits at the category-prior ceiling
(M1 ~ NOCS 13.5% / Objectron 22% / ARKit 43.6%) and the residual is within-category
per-instance variance. If MoGe-2 metric depth -> object bbox beats that ceiling
(especially ARKit), per-instance metric depth is the missing signal -> pivot the
scale anchor to MoGe-2.

Method per held-out object (RGBA mask comes baked in alpha from the real loader):
  full RGB -> MoGe-2 metric points [H,W,3] -> apply object mask -> axis-aligned 3D bbox.
  pred_iso = max(W,H,D) of masked points (camera AABB max extent).
  gt_iso   = max(GT metric_dims) (object's largest metric dimension).
  rel = |pred_iso - gt_iso| / gt_iso.

CAVEAT (report honestly): depth-bridge measures only the VISIBLE surface, so it
under-measures occluded extent -- a structural disadvantage vs the generative model
which hallucinates full shape. Camera-AABB max-extent also inflates for rotated
objects. Approximate probe, not a perfect oracle.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/mnt/source/MoGe")  # MoGe-2 (model.v2) lives in the repo, not pip pkg
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bbox_from_depth import MoGe2Estimator  # noqa: E402
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset  # noqa: E402


META = {
    "nocs_real275": "omninocs_release_nocs_real275/nocs_real275_train_metadata.json",
    "objectron": "omninocs_release_objectron/objectron_train_metadata.json",
    "arkitscenes": "omninocs_release_ARKitScenes/ARKitScenes_train_metadata.json",
}


def build_fovx_map(src):
    """image_name -> horizontal FOV (radians) from GT intrinsics.
    fov_x = 2*atan(cx/fx); scale-invariant so downscale (ARKit 7.5x) is irrelevant."""
    frames = json.load(open(f"{OMNI}/{META[src]}"))
    m = {}
    for fr in frames:
        K = fr.get("intrinsics")
        if K and K.get("fx"):
            m[fr["image_name"]] = 2.0 * math.atan(float(K["cx"]) / float(K["fx"]))
    return m


def infer_points_fovx(est, rgb_u8, fov_x):
    """MoGe-2 metric points using a supplied horizontal FOV (radians) instead of
    the model's self-estimate. Mirrors MoGe2Estimator.infer_points preprocessing."""
    img_t = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().to(est.device) / 255.0
    out = est.model.infer(img_t.unsqueeze(0), apply_mask=True,
                          force_projection=True, fov_x=float(fov_x))
    pts = out["points"].squeeze(0)
    if pts.ndim == 3 and pts.shape[0] == 3 and pts.shape[-1] != 3:
        pts = pts.permute(1, 2, 0)
    pts_np = pts.cpu().float().numpy()
    pts_np[~np.isfinite(pts_np)] = np.nan
    return pts_np


def robust_iso_extent(points, mask, pct):
    """Isotropic extent (max per-axis range) of masked metric points, with
    percentile clipping to reject silhouette depth-bleed outliers.
    pct=0 -> raw min/max (fragile); pct=2 -> 2nd..98th percentile per axis."""
    pts = points[mask]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 100:
        return None, 0
    lo = np.percentile(pts, pct, axis=0)
    hi = np.percentile(pts, 100 - pct, axis=0)
    return float((hi - lo).max()), len(pts)

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
RGB_ROOTS = {
    "nocs_real275": f"{OMNI}/real_test",
    "objectron": OMNI,
    "arkitscenes": OMNI,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--pool-per-source", type=int, default=1500)
    ap.add_argument("--pct", type=float, default=2.0,
                    help="percentile clip per axis (0=raw min/max)")
    ap.add_argument("--gt-intrinsics", action="store_true",
                    help="feed GT fov_x to MoGe-2 instead of its self-estimate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="+",
                    default=["nocs_real275", "objectron", "arkitscenes"])
    ap.add_argument("--output",
                    default="artifacts/metric_scale/metrics/moge2_depth_bridge_oracle.jsonl")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    est = MoGe2Estimator(device="cuda")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    summary = {}
    for src in args.sources:
        ds = OmniNOCSObjectDataset(
            omninocs_root=OMNI, sources=[src], rgb_roots=RGB_ROOTS,
            split="train", max_records_per_source=args.pool_per_source,
            skip_missing_rgb=True)
        fovx_map = build_fovx_map(src) if args.gt_intrinsics else {}
        idxs = rng.permutation(len(ds))
        rels, n_no_fov = [], 0
        for i in idxs:
            if len(rels) >= args.per_source:
                break
            try:
                item = ds[int(i)]
                rgba = item["image"]
                rgb = np.ascontiguousarray(rgba[..., :3])
                mask = rgba[..., 3] > 127
                if mask.sum() < 200:
                    continue
                if args.gt_intrinsics:
                    fovx = fovx_map.get(ds.records[int(i)]["image_name"])
                    if fovx is None:
                        n_no_fov += 1
                        continue
                    pts = infer_points_fovx(est, rgb, fovx)
                else:
                    pts = est.infer_points(rgb, np.eye(3))  # MoGe-2 self-estimates FOV
                pred_iso, npts = robust_iso_extent(pts, mask, args.pct)
                if pred_iso is None:
                    continue
                gt = np.asarray(item["metric_dims"], float)
                gt_iso = float(gt.max())
                if gt_iso < 1e-4:
                    continue
                rel = abs(pred_iso - gt_iso) / gt_iso
                rels.append(rel)
                fout.write(json.dumps(dict(
                    source=src, uid=item["uid"], category=item["category"],
                    gt_iso=gt_iso, pred_iso=pred_iso, rel=rel, n_pts=npts)) + "\n")
            except Exception as e:
                print(f"  [skip] {src} idx={int(i)}: {e}")
                continue
        rels = np.array(rels)
        med = float(np.median(rels)) if len(rels) else float("nan")
        mean = float(np.mean(rels)) if len(rels) else float("nan")
        summary[src] = (len(rels), med, mean)
        print(f"[{src}] n={len(rels)}  median={med*100:.1f}%  mean={mean*100:.1f}%", flush=True)
    fout.close()

    print("\n=== MoGe-2 depth-bridge isotropic-size error (per source) ===")
    print(f"{'source':<14}{'N':>5}{'median%':>10}{'mean%':>9}")
    for src, (n, med, mean) in summary.items():
        print(f"{src:<14}{n:>5}{med*100:>9.1f}%{mean*100:>8.1f}%")
    print(f"\nwrote {out_path}")
    print("\nCompare to: trained model (NOCS ~7 / Objectron ~18 / ARKit ~34%) and")
    print("category-prior ceiling M1 (NOCS 13.5 / Objectron 22 / ARKit 43.6%).")


if __name__ == "__main__":
    main()
