#!/usr/bin/env python
"""mAP@50 error decomposition (CPU-only, from cached predictions).

Re-scores cached full-set predictions (pred_boxes.jsonl from
nocs_map_step05_predpose.py) with GT components swapped in one at a time:

  FULL PRED            -> reproduces the published number (sanity)
  GT translation       -> how many points translation error costs
  GT rotation          -> how many points rotation error costs
  GT rot+trans         -> size-only residual (step-0 predicts ~100)
  GT size              -> how many points size error costs

  python scripts/map50_decomposition.py \
      --pred artifacts/nocs_map_step05_joint_mot_full/pred_boxes.jsonl
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/mnt/source/sam-3d-objects/scripts")
import nocs_map_step0_oracle as step0  # noqa: E402  (stubs tf/ICP, imports nocs utils)

nocs_utils = step0.nocs_utils
Z180 = np.diag([-1.0, -1.0, 1.0, 1.0])


def assemble(pred_path):
    recs = [json.loads(l) for l in open(pred_path) if l.strip()]
    byframe = {}
    for r in recs:
        f, i = r["key"].split("#")
        byframe.setdefault(f, {})[int(i)] = r
    results = []
    for frame_rel, preds in sorted(byframe.items()):
        fp = os.path.join(step0.REAL_TEST, frame_rel)
        parsed = step0.load_gt_instances(fp)
        if parsed is None:
            continue
        gt_mask, gt_coord, gt_class_ids = parsed
        d16 = step0.load_depth(fp)
        if d16 is None:
            continue
        gt_RTs, gt_scales, _, _ = nocs_utils.align(
            gt_class_ids, gt_mask, gt_coord, d16, step0.INTRINSICS,
            step0.SYNSET_NAMES, fp, None)
        gt_bbox = nocs_utils.extract_bboxes(gt_mask)
        n = len(gt_class_ids)
        if set(preds.keys()) != set(range(n)):
            continue  # incomplete frame in cache
        pred_RTs = np.stack([Z180 @ np.array(preds[i]["pred_RT"]) for i in range(n)])
        pred_scales = np.stack([np.array(preds[i]["pred_scales"]) for i in range(n)])
        results.append({
            "gt_class_ids": np.asarray(gt_class_ids), "gt_RTs": np.asarray(gt_RTs),
            "gt_scales": np.asarray(gt_scales),
            "gt_handle_visibility": np.ones_like(gt_class_ids),
            "gt_bboxes": np.asarray(gt_bbox),
            "pred_class_ids": np.asarray(gt_class_ids), "pred_RTs": pred_RTs,
            "pred_scales": pred_scales,
            "pred_scores": np.ones(n, dtype=np.float32),
            "pred_bboxes": np.asarray(gt_bbox),
        })
    return results


def variant(results, use_gt_t=False, use_gt_r=False, use_gt_s=False):
    out = []
    for r in results:
        rr = dict(r)
        pred_RTs = r["pred_RTs"].copy()
        pred_scales = r["pred_scales"].copy()
        for i in range(len(r["gt_class_ids"])):
            gt_RT = r["gt_RTs"][i]
            iso = np.cbrt(np.linalg.det(gt_RT[:3, :3]))
            if use_gt_t:
                pred_RTs[i][:3, 3] = gt_RT[:3, 3]
            if use_gt_r:
                pred_RTs[i][:3, :3] = gt_RT[:3, :3] / iso   # pure rotation
            if use_gt_s:
                pred_scales[i] = r["gt_scales"][i] * iso     # metric dims
        rr["pred_RTs"], rr["pred_scales"] = pred_RTs, pred_scales
        out.append(rr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",
                    default="artifacts/nocs_map_step05_joint_mot_full/pred_boxes.jsonl")
    args = ap.parse_args()
    results = assemble(args.pred)
    n_inst = sum(len(r["gt_class_ids"]) for r in results)
    print(f"assembled {len(results)} frames / {n_inst} instances from {args.pred}\n")
    rows = [
        ("FULL PRED", {}),
        ("GT translation", dict(use_gt_t=True)),
        ("GT rotation", dict(use_gt_r=True)),
        ("GT rot+trans", dict(use_gt_t=True, use_gt_r=True)),
        ("GT size", dict(use_gt_s=True)),
    ]
    summary = []
    for tag, kw in rows:
        with tempfile.TemporaryDirectory() as td:
            m25, m50 = step0.run_map(variant(results, **kw), td, tag)
        summary.append((tag, m25, m50))
    print("\n=== DECOMPOSITION SUMMARY ===")
    for tag, m25, m50 in summary:
        print(f"  {tag:<16} mAP@25 {m25:5.1f}   mAP@50 {m50:5.1f}")


if __name__ == "__main__":
    main()
