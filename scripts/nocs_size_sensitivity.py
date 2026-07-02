#!/usr/bin/env python
"""Size-sensitivity demo (2026-06-16): how much does box SIZE matter for mAP @ 3D IoU,
holding pose = GT? Answers "would a unit cube give the same mAP?" (No: mAP -> 0.)

Reuses nocs_map_step0_oracle (GT parse/align/eval). GT pose for every box; only the SIZE
hypothesis varies. No GPU."""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/mnt/source/sam-3d-objects/scripts")
import nocs_map_step0_oracle as step0  # noqa: E402


def results_with_sizes(gts, size_fn, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for g in gts:
        n = len(g["gt_class_ids"])
        out.append({
            "gt_class_ids": g["gt_class_ids"], "gt_RTs": g["gt_RTs"], "gt_scales": g["gt_scales"],
            "gt_handle_visibility": g["gt_handle_visibility"], "gt_bboxes": g["gt_bboxes"],
            "pred_class_ids": g["gt_class_ids"].copy(), "pred_RTs": g["gt_RTs"].copy(),  # GT pose
            "pred_scales": size_fn(g["gt_scales"], n, rng),
            "pred_scores": np.ones(n, dtype=np.float32), "pred_bboxes": g["gt_bboxes"].copy(),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--out-dir", default="/mnt/source/sam-3d-objects/artifacts/nocs_size_sens")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print("[size-sens] aligning GT (pose=GT for all hypotheses) ...", flush=True)
    gts = step0.gather_gt(args.max_images, seed=0)
    n_inst = sum(len(g["gt_class_ids"]) for g in gts)
    print(f"[size-sens] {len(gts)} images, {n_inst} instances\n", flush=True)
    print("[size-sens] mAP @ IoU 0.25 / 0.50, GT pose, varying SIZE hypothesis:\n", flush=True)

    # gt_scales = NOCS coord extent; align bakes metric scale into RT, so isotropic multipliers
    # on gt_scales scale the metric box isotropically. A "unit cube" in metric terms = make the
    # NOCS-coord box a cube of the right NORM but unit aspect (replace shape, keep volume-ish) OR
    # simply blow it up. Cleanest "unit cube" = each pred box a 1m cube regardless of GT: do that
    # in metric space by overriding scales to a value whose metric size ~1m. Since metric ext =
    # (RT scale)*gt_scales, and gt_scales~O(1) coord extent, set pred ext via a fixed multiple is
    # not 1m absolute -> instead demonstrate with isotropic factors + an aspect-destroying cube.
    cases = [
        ("GT size", lambda s, n, r: s.copy()),
        ("iso +7%", lambda s, n, r: s * 1.07),
        ("iso +15%", lambda s, n, r: s * 1.15),
        ("iso +30%", lambda s, n, r: s * 1.30),
        ("iso +50%", lambda s, n, r: s * 1.50),
        ("iso 2x (k=2)", lambda s, n, r: s * 2.0),
        ("iso 3x (k=3)", lambda s, n, r: s * 3.0),
        # aspect destroyed: replace each box with an axis-aligned CUBE of the same volume (right
        # overall scale, wrong proportions) -> isolates PROPORTION error from overall scale
        ("equal-vol CUBE (proportions destroyed)",
         lambda s, n, r: np.cbrt(np.prod(s, axis=1, keepdims=True)) * np.ones_like(s)),
        # true "unit-ish cube": every box the SAME large cube (ignores GT size entirely) ~ the
        # user's "output meshes in a unit cube" -> use 5x the median coord-extent as the cube side
        ("fixed BIG cube (ignores GT, ~unit-cube analogue)",
         lambda s, n, r: np.tile(5.0 * np.median(np.concatenate([g['gt_scales'] for g in gts])),
                                 (n, 3))),
    ]
    for name, fn in cases:
        step0.run_map(results_with_sizes(gts, fn), args.out_dir, name[:22])
    print("\n[size-sens] DONE", flush=True)


if __name__ == "__main__":
    main()
