#!/usr/bin/env python
"""
Step 0 (2026-06-16): GT-pose + size mAP oracle on NOCS-Real275, via the OFFICIAL NOCS eval.

Purpose ([[scene-scale-and-full-scene-reconstruction]] benchmark plan): get a real, comparable
mAP @ 3D IoU 0.25/0.5 number on NOCS-Real275 (the NOCSformer/CubeRCNN table), upper-bounded by
GT pose so it isolates SIZE quality. Answers "is this a size paper or a pose paper" before any
retrain.

Reuses the official NOCS_CVPR2019 code (/mnt/source/NOCS_CVPR2019):
  - dataset.NOCSDataset  -> load real_test (color/coord/depth/mask/meta)
  - utils.align          -> GT 6DoF poses (gt_RTs) + gt_scales from coord+depth (Umeyama)
  - utils.compute_degree_cm_mAP -> the official mAP @ 3D IoU (symmetry handled)
Only `tensorflow` (imported at utils top, unused by the eval fns) is missing -> stubbed.

Box parameterization (official): metric box corners = RT @ get_3d_bbox(scales). RT is a
similarity (sR|t); metric extent = (similarity scale)*scales. So scaling `scales` isotropically
scales the metric box isotropically -> the noise probe perturbs ONLY iso size, GT pose perfect.

Modes:
  sanity : pred = GT exactly            -> mAP@25/50 must be ~100 (validates the harness).
  noise  : pred pose = GT, pred size = GT * lognormal(sigma)  -> mAP-vs-iso-size-error curve
           (ZERO GPU; the ceiling on mAP given size error alone, with perfect pose).
  (real pipeline predictions = a follow-up flag, not built here.)

Run:
  CONDA_PREFIX=/opt/conda/envs/sam3d /opt/conda/envs/sam3d/bin/python \
      scripts/nocs_map_step0_oracle.py --max-images 60
"""
import argparse
import glob
import os
import sys
import types

import cv2
import numpy as np

# --- make the official NOCS code importable in the sam3d env ---
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))  # utils imports tf; eval fns don't use it
sys.modules.setdefault("ICP", types.ModuleType("ICP"))  # utils imports ICP->open3d (libgomp); align uses Umeyama, not ICP

NOCS_REPO = "/mnt/source/NOCS_CVPR2019"
sys.path.insert(0, NOCS_REPO)
import utils as nocs_utils  # noqa: E402

SYNSET_NAMES = ["BG", "bottle", "bowl", "camera", "can", "laptop", "mug"]
REAL_TEST = "/mnt/source/datasets_sam3d/OmniNOCS/real_test"
INTRINSICS = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])


def load_depth(frame_path):
    """NOCS-Real depth: 16-bit, possibly encoded across G/B channels (BGR in cv2)."""
    d = cv2.imread(frame_path + "_depth.png", -1)
    if d is None:
        return None
    if d.ndim == 3:
        return (np.uint16(d[:, :, 1] * 256) + np.uint16(d[:, :, 2])).astype(np.uint16)
    return d


def load_gt_instances(frame_path):
    """Replicates NOCSDataset.process_data mask/coord/class parsing (no CAD models).
    Returns gt_mask [H,W,N], gt_coord [H,W,N,3], gt_class_ids [N] or None."""
    mask_im = cv2.imread(frame_path + "_mask.png")
    coord_im = cv2.imread(frame_path + "_coord.png")
    if mask_im is None or coord_im is None:
        return None
    mask_im = mask_im[:, :, 2]                                  # instance ids (R channel)
    coord_map = coord_im[:, :, :3][:, :, (2, 1, 0)].astype(np.float32) / 255.0
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]                 # NOCS flips z
    inst_dict = {}
    with open(frame_path + "_meta.txt") as f:
        for line in f:
            w = line.split()
            if len(w) >= 2:
                inst_dict[int(w[0])] = int(w[1])
    inst_ids = sorted(int(x) for x in np.unique(mask_im))
    inst_ids = [i for i in inst_ids if i != 255]               # drop background
    masks, coords, class_ids = [], [], []
    for iid in inst_ids:
        cls = inst_dict.get(iid, 0)
        if cls < 1 or cls > 6:                                  # keep the 6 NOCS classes
            continue
        m = (mask_im == iid)
        if m.sum() < 50:
            continue
        masks.append(m.astype(np.uint8))
        coords.append(coord_map * m[..., None])
        class_ids.append(cls)
    if not class_ids:
        return None
    return (np.stack(masks, axis=2), np.stack(coords, axis=2),
            np.asarray(class_ids, dtype=np.int_))


def gather_gt(max_images, seed=0):
    """Per frame: parse GT, align coord+depth -> gt_RTs + gt_scales (align's own bbox_scales)."""
    frames = sorted(glob.glob(os.path.join(REAL_TEST, "*", "*_color.png")))
    frame_paths = [f[: -len("_color.png")] for f in frames]
    rng = np.random.RandomState(seed)
    if max_images and max_images < len(frame_paths):
        idx = sorted(rng.choice(len(frame_paths), size=max_images, replace=False).tolist())
        frame_paths = [frame_paths[i] for i in idx]
    gts = []
    for k, fp in enumerate(frame_paths):
        parsed = load_gt_instances(fp)
        if parsed is None:
            continue
        gt_mask, gt_coord, gt_class_ids = parsed
        depth = load_depth(fp)
        if depth is None:
            continue
        gt_bbox = nocs_utils.extract_bboxes(gt_mask)
        # align returns (RTs, bbox_scales): metric box = RT @ get_3d_bbox(bbox_scales). Use both
        # -> fully self-consistent (no CAD models); sanity pred=GT must give IoU=1.
        gt_RTs, gt_scales, _, _ = nocs_utils.align(
            gt_class_ids, gt_mask, gt_coord, depth, INTRINSICS, SYNSET_NAMES, fp, None,
        )
        gts.append({
            "gt_class_ids": np.asarray(gt_class_ids),
            "gt_RTs": np.asarray(gt_RTs),
            "gt_scales": np.asarray(gt_scales),
            "gt_bboxes": np.asarray(gt_bbox),
            "gt_handle_visibility": np.ones_like(gt_class_ids),
        })
        if (k + 1) % 10 == 0:
            print(f"[step0] aligned {k+1}/{len(frame_paths)} frames "
                  f"({sum(len(g['gt_class_ids']) for g in gts)} instances)", flush=True)
    return gts


def make_results(gts, iso_sigma, seed=0):
    """Build final_results; pred pose = GT, pred size = GT * lognormal(iso_sigma) (per instance)."""
    rng = np.random.RandomState(seed)
    out = []
    for g in gts:
        n = len(g["gt_class_ids"])
        if iso_sigma > 0:
            factor = np.exp(rng.normal(0.0, iso_sigma, size=(n, 1)))  # per-instance isotropic
        else:
            factor = np.ones((n, 1))
        out.append({
            "gt_class_ids": g["gt_class_ids"],
            "gt_RTs": g["gt_RTs"],
            "gt_scales": g["gt_scales"],
            "gt_handle_visibility": g["gt_handle_visibility"],
            "gt_bboxes": g["gt_bboxes"],
            "pred_class_ids": g["gt_class_ids"].copy(),
            "pred_RTs": g["gt_RTs"].copy(),                       # GT pose
            "pred_scales": g["gt_scales"] * factor,               # GT size * iso noise
            "pred_scores": np.ones(n, dtype=np.float32),
            "pred_bboxes": g["gt_bboxes"].copy(),
        })
    return out


def run_map(results, log_dir, tag):
    # degree/shift thresholds must include 5,10,15 (the fn hardcodes printing those pose APs);
    # iou list must include 0.25 & 0.5 (it indexes them). Matches detect_eval's call.
    iou_aps, _ = nocs_utils.compute_degree_cm_mAP(
        results, SYNSET_NAMES, log_dir,
        degree_thresholds=[5, 10, 15], shift_thresholds=[5, 10, 15],
        iou_3d_thresholds=[0.25, 0.5], iou_pose_thres=0.1, use_matches_for_pose=False,
    )
    map25, map50 = iou_aps[-1, 0] * 100, iou_aps[-1, 1] * 100
    per_cls25 = {SYNSET_NAMES[c]: round(iou_aps[c, 0] * 100, 1) for c in range(1, 7)}
    print(f"[step0] {tag:>16}:  mAP@25 = {map25:5.1f}   mAP@50 = {map50:5.1f}   "
          f"per-class@25 {per_cls25}", flush=True)
    return map25, map50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=60, help="subsample real_test (align is slow)")
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.07, 0.15, 0.30],
                    help="lognormal iso size-error stds for the noise probe")
    ap.add_argument("--out-dir", default="artifacts/nocs_map_step0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join("/mnt/source/sam-3d-objects", args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[step0] parsing NOCS-Real275 + aligning GT poses ...", flush=True)
    gts = gather_gt(args.max_images, seed=args.seed)
    n_inst = sum(len(g["gt_class_ids"]) for g in gts)
    print(f"[step0] GT ready: {len(gts)} images, {n_inst} instances", flush=True)

    print("\n[step0] === reference (transferred SOTA, OmniNOCS Tab.5): "
          "NOCSformer 43.5/10.6 ; CubeRCNN 14.9/4.1 ; supervised NOCS 79.6/72.4 ===", flush=True)
    print("[step0] === results: mAP @ IoU 0.25 / 0.50 (GT pose; size = GT * iso-error) ===\n", flush=True)

    run_map(make_results(gts, 0.0), out_dir, "SANITY pred=GT")  # must be ~100
    for s in args.sigmas:
        run_map(make_results(gts, s, seed=args.seed), out_dir, f"iso-err sigma={s}")
    print("\n[step0] DONE  (sanity must read ~100; noise curve = mAP ceiling vs iso size error, "
          "perfect pose)", flush=True)


if __name__ == "__main__":
    main()
