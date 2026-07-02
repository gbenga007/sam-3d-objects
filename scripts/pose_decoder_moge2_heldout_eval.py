#!/usr/bin/env python
"""
Pose-decoder + MoGe-2 pointmap eval on the mixed heldout set (calibration baseline).

The decisive test of the "pointmap IS the metric anchor" thesis at deployment
conditions: run the FROZEN stock pipeline (no MetricScaleHead) with the precomputed
MoGe-2 metric pointmaps on the SAME heldout instances the trained heads are evaluated
on, and read metric dims off the pose decoder:

    pred_dims = canonical_mesh_extents * out["scale"]     # per-axis W,H,D candidate
    pred_iso  = max(pred_dims) vs max(GT dims)            # isotropic

This is baseline #2 from planning/MOGE2_METRIC_HEAD_PLAN_2026-06-10.md ("calibration
constant"): raw numbers here; a global / per-source / per-category constant can be
fit post-hoc from the JSONL (fit on train-split instances or leave-one-out, not the
same instances being scored).

Per-axis caveat: pred axes live in the predicted-mesh canonical frame, GT [W,H,D] in
the OmniNOCS canonical frame. We record BOTH direct-order and sorted-axes errors;
the analysis can choose (prior mesh-bbox evals used direct order successfully).

Instances: the heldout entries of feature_caches/moge2_mixed_10k.pt (uid-exact same
set the head evals use; mask_pixels filter already applied there). Stock 25/25 flow
steps (~1-3 min/instance) — use --per-source to subsample (seeded, stratified).

Resumable: skips uids already present in the output JSONL (append mode).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/opt/conda/envs/sam3d"))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import sam3d_objects  # noqa: F401,E402  (registers targets; do not remove)
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset  # noqa: E402

from gt_pointmap_metric_oracle import (  # noqa: E402  (same dir)
    load_pipeline,
    run_inference,
    pred_dims_from_output,
    canonical_extents,
)

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
RGB_ROOTS = {
    "nocs_real275": f"{OMNI}/real_test",
    "objectron": OMNI,
    "arkitscenes": OMNI,
}


def sanitize(image_name: str) -> str:
    return image_name.replace("/", "__").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "checkpoints/hf/pipeline.yaml"))
    ap.add_argument("--feature-cache",
                    default="artifacts/metric_scale/feature_caches/moge2_mixed_10k.pt")
    ap.add_argument("--pointmap-dir",
                    default="artifacts/metric_scale/moge2_pointmaps")
    ap.add_argument("--per-source", type=int, default=50,
                    help="instances per source (seeded stratified subsample); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inference-seed", type=int, default=42)
    ap.add_argument("--output",
                    default="artifacts/metric_scale/metrics/pose_decoder_moge2_heldout.jsonl")
    args = ap.parse_args()

    # Heldout instance list (uid, source, image_name, gt dims) from the feature cache.
    cache = torch.load(args.feature_cache, weights_only=False)
    heldout = cache["heldout"]
    del cache
    by_source: dict[str, list[dict]] = {}
    for e in heldout:
        by_source.setdefault(e["source"], []).append(
            dict(uid=e["uid"], source=e["source"], image_name=e["image_name"],
                 category=e["category"], gt_dims=e["metric_dims"].tolist())
        )
    rng = np.random.default_rng(args.seed)
    chosen: list[dict] = []
    for src, entries in sorted(by_source.items()):
        order = rng.permutation(len(entries))
        take = len(entries) if args.per_source <= 0 else min(args.per_source, len(entries))
        chosen.extend(entries[i] for i in order[:take])
    print(f"heldout instances chosen: {len(chosen)} "
          f"({ {s: min(len(v), args.per_source or len(v)) for s, v in sorted(by_source.items())} })",
          flush=True)

    # uid -> dataset index (same dataset args as the cache run => same records).
    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI, sources=["nocs_real275", "objectron", "arkitscenes"],
        rgb_roots=RGB_ROOTS, split="train", min_mask_pixels=500,
        max_records_per_source=3334, skip_missing_rgb=True,
    )
    uid_to_idx = {rec["uid"]: i for i, rec in enumerate(ds.records)}

    pm_dir = Path(args.pointmap_dir)
    manifest = json.load(open(pm_dir / "manifest.json"))["frames"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_uids = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done_uids.add(json.loads(line)["uid"])
            except Exception:
                pass
        print(f"resume: {len(done_uids)} uids already in {out_path}", flush=True)
    fout = open(out_path, "a")

    pipeline = load_pipeline(args.config)

    rels = {"iso": {}, "axis": {}, "axis_sorted": {}}
    n_done = n_fail = 0
    for k, inst in enumerate(chosen):
        if inst["uid"] in done_uids:
            continue
        try:
            item = ds[uid_to_idx[inst["uid"]]]
            rgba = item["image"]
            rgb = np.ascontiguousarray(rgba[..., :3])
            mask = rgba[..., 3] > 127
            gt_dims = np.asarray(inst["gt_dims"], dtype=np.float64)

            pm = np.load(pm_dir / manifest[inst["image_name"]]).astype(np.float32)
            out = run_inference(pipeline, rgb, mask, args.inference_seed,
                                torch.from_numpy(pm))
            pdims = pred_dims_from_output(out)
            if pdims is None:
                n_fail += 1
                continue

            gt_iso = float(np.max(gt_dims))
            rel_iso = abs(float(np.max(pdims)) - gt_iso) / max(gt_iso, 1e-4)
            rel_axis = (np.abs(pdims - gt_dims) / np.clip(gt_dims, 1e-4, None)).tolist()
            rel_axis_sorted = (
                np.abs(np.sort(pdims) - np.sort(gt_dims))
                / np.clip(np.sort(gt_dims), 1e-4, None)
            ).tolist()

            cext = canonical_extents(out)
            sc = out["scale"][0].detach().cpu().float().numpy().reshape(-1)
            rec = dict(inst, pred_dims=pdims.tolist(), gt_iso=gt_iso,
                       rel_iso=rel_iso, rel_axis=rel_axis,
                       rel_axis_sorted=rel_axis_sorted,
                       canon_ext=(cext.tolist() if cext is not None else None),
                       scale=sc.tolist())
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            src = inst["source"]
            rels["iso"].setdefault(src, []).append(rel_iso)
            rels["axis"].setdefault(src, []).append(float(np.mean(rel_axis)))
            rels["axis_sorted"].setdefault(src, []).append(float(np.mean(rel_axis_sorted)))
            n_done += 1
            print(f"[{n_done}/{len(chosen)}] {src:<13} {inst['category']:<10} "
                  f"iso={rel_iso:.3f} axis={np.mean(rel_axis):.3f}", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"  [skip] {inst['uid']}: {e}", flush=True)
            continue

    fout.close()
    print(f"\nDONE n={n_done} fail={n_fail}")
    for metric, per_src in rels.items():
        for src, vals in sorted(per_src.items()):
            v = np.array(vals)
            print(f"{metric:<12} {src:<13} n={len(v):>3} "
                  f"median={np.median(v)*100:6.1f}% mean={np.mean(v)*100:6.1f}%")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
