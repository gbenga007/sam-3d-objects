#!/usr/bin/env python
"""
MoGe-2 vs GT pointmap_scale screen (cheap, no SS/SLAT — MoGe-2 inference only).

WHY: [[project_pointmap_scale_anchor]] verified `out["scale"] = ssi_ratio · pointmap_scale`,
where `pointmap_scale` is the scene scale the pipeline computes from the INPUT pointmap via
`ObjectCentricSSI(use_scene_scale=True)` (the configured normalizer in pipeline.yaml). The
`ssi_ratio` is unit-free, so the metric error from swapping in a MoGe-2 pointmap reduces to
ONE ratio:  MoGe2_pointmap_scale / GT_pointmap_scale.  This screen measures that ratio per
NOCS object WITHOUT running the heavy pipeline, to predict — before committing GPU-hours —
whether MoGe-2-as-pointmap lands near the GT oracle (~5%) and whether any residual bias is
systematic (→ a slimmed head can correct it) or random (→ it can't).

FAITHFUL: instantiates the SAME normalizer the pipeline uses and calls it on each pointmap.
The scale stat (per-pixel max|x|,|y|,|z| about the object-median shift, median over frame) is
invariant to axis sign/permutation, so GT (camera coords) vs MoGe-2 (its own convention) is a
valid comparison.

Per NOCS object:
  GT_scale   = normalizer(GT_metric_pointmap from Kinect depth + REAL275 K, mask).scale
  MoGe2_scale= normalizer(MoGe-2 metric points [self-intrinsics], mask).scale
  ratio      = MoGe2_scale / GT_scale     (1.0 = MoGe-2 anchor as good as GT)

Reports: median/mean ratio, per-category ratio, ratio-vs-object-size trend (systematic bias?),
and how tightly GT_scale itself tracks object metric size (sanity on the anchor).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
import imageio.v3 as iio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/source/MoGe")          # MoGe-2 (model.v2), not pip moge (v1)
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from eval_bbox_from_depth import MoGe2Estimator  # noqa: E402
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset  # noqa: E402
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import ObjectCentricSSI  # noqa: E402

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
NOCS_RGB_ROOT = f"{OMNI}/real_test"
RGB_ROOTS = {"nocs_real275": NOCS_RGB_ROOT}
NOCS_K = np.array(
    [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def depth_path_for(image_name):
    rel = image_name.split("/test/")[-1]
    return Path(NOCS_RGB_ROOT) / f"{rel}_depth.png"


def gt_camera_pointmap(depth_m, K):
    """GT metric pointmap in camera coords [3,H,W] (X,Y,Z). Convention-robust for the
    scale stat (per-pixel max|.| is invariant to axis sign/permutation)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_m
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    return np.stack([X, Y, Z], axis=0).astype(np.float32)  # [3,H,W]


def scale_of(normalizer, pm_3hw_np, mask_hw):
    pm = torch.from_numpy(np.ascontiguousarray(pm_3hw_np)).float()
    mask = torch.from_numpy(mask_hw.astype(np.float32))[None]  # [1,H,W]
    out = normalizer.normalize(pm, mask)
    s = out.scale.detach().cpu().float().numpy().reshape(-1)
    return float(s[0])  # ObjectCentricSSI scale is isotropic (expand_as shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--pool-per-source", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-mask-pixels", type=int, default=200)
    ap.add_argument("--output",
                    default="artifacts/metric_scale/metrics/moge2_vs_gt_pointmap_scale_screen.jsonl")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    est = MoGe2Estimator(device="cuda")
    normalizer = ObjectCentricSSI(allow_scale_and_shift_override=True, use_scene_scale=True)

    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI, sources=["nocs_real275"], rgb_roots=RGB_ROOTS,
        split="train", max_records_per_source=args.pool_per_source, skip_missing_rgb=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    rows = []
    for i in rng.permutation(len(ds)):
        if len(rows) >= args.per_source:
            break
        try:
            item = ds[int(i)]
            if item["mask_pixels"] < args.min_mask_pixels:
                continue
            dpath = depth_path_for(item["image_name"])
            if not dpath.exists():
                continue
            depth = iio.imread(dpath).astype(np.float32) / 1000.0  # mm -> m
            depth[depth <= 0] = np.nan

            rgba = item["image"]
            rgb = np.ascontiguousarray(rgba[..., :3])
            mask = rgba[..., 3] > 127
            gt_dims = np.asarray(item["metric_dims"], dtype=np.float64)
            gt_iso = float(gt_dims.max())

            # align K to image res if depth differs (NOCS native should already match)
            H, W = mask.shape
            K = NOCS_K.copy()
            if depth.shape != (H, W):
                import cv2
                sy, sx = H / depth.shape[0], W / depth.shape[1]
                K[0, :] *= sx
                K[1, :] *= sy
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

            gt_pm = gt_camera_pointmap(depth, K)
            gt_scale = scale_of(normalizer, gt_pm, mask)

            # MoGe-2 metric points (self-estimated intrinsics — the deploy setting)
            pts = est.infer_points(rgb, np.eye(3))  # [H,W,3], nan for invalid
            pts = np.ascontiguousarray(pts)
            pts[~np.isfinite(pts)] = np.nan
            moge2_pm = np.transpose(pts, (2, 0, 1))  # [3,H,W]
            moge2_scale = scale_of(normalizer, moge2_pm, mask)

            if not (np.isfinite(gt_scale) and np.isfinite(moge2_scale)) or gt_scale <= 0:
                continue
            ratio = moge2_scale / gt_scale

            rec = dict(uid=item["uid"], category=item["category"],
                       image_name=item["image_name"], gt_iso=gt_iso,
                       gt_scale=gt_scale, moge2_scale=moge2_scale, ratio=ratio)
            rows.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"[{len(rows):>3}] {item['category']:<8} gt_scale={gt_scale:.3f} "
                  f"moge2={moge2_scale:.3f} ratio={ratio:.3f}", flush=True)
        except Exception as e:
            print(f"  [skip] idx={int(i)}: {e}", flush=True)
            continue
    fout.close()

    if not rows:
        print("no rows")
        return

    ratios = np.array([r["ratio"] for r in rows])
    gt_iso = np.array([r["gt_iso"] for r in rows])
    gt_scale = np.array([r["gt_scale"] for r in rows])

    print(f"\n=== MoGe-2 / GT pointmap_scale ratio  (n={len(rows)}) ===")
    print(f"median={np.median(ratios):.3f}  mean={np.mean(ratios):.3f}  std={np.std(ratios):.3f}")
    print(f"log-ratio std (systematicity; small => globally calibratable)="
          f"{np.std(np.log(ratios)):.3f}")
    # implied metric error if we used MoGe-2 pointmap with NO correction:
    impl = np.abs(ratios - 1.0)
    print(f"implied |ratio-1| median={np.median(impl)*100:.1f}%  mean={np.mean(impl)*100:.1f}%")
    # after a single global scalar correction (divide by median ratio):
    corr = np.abs(ratios / np.median(ratios) - 1.0)
    print(f"  after global-scalar correction: median={np.median(corr)*100:.1f}%  "
          f"mean={np.mean(corr)*100:.1f}%")

    cats = sorted(set(r["category"] for r in rows))
    print("\nper-category median ratio:")
    for c in cats:
        rs = np.array([r["ratio"] for r in rows if r["category"] == c])
        print(f"  {c:<10} n={len(rs):>3}  median={np.median(rs):.3f}  mean={np.mean(rs):.3f}")

    # does the ratio depend on object size? (systematic small-object bias check)
    if len(rows) >= 8:
        lr = np.log(ratios)
        ls = np.log(gt_iso)
        r_pear = float(np.corrcoef(ls, lr)[0, 1])
        print(f"\nlog(ratio) vs log(gt_iso) Pearson r={r_pear:.3f}  "
              f"(neg => over-measures SMALL objects, as depth-bridge found)")
        r_anchor = float(np.corrcoef(np.log(gt_scale), ls)[0, 1])
        print(f"log(gt_scale) vs log(gt_iso) Pearson r={r_anchor:.3f}  "
              f"(how well the scene anchor tracks object size)")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
