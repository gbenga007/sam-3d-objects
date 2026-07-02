"""
Diagnostic: how accurate is the ICP-grounded scale (after layout_post_optimization)
against OmniNOCS ground-truth metric dimensions?

The previous check_ss_scale_accuracy.py measured the raw SS scale token before any
geometric refinement. That measurement was of the right quantity mathematically but the
question is whether the full pipeline — which fits the generated Gaussian to the MoGe
pointmap via height-ratio pre-alignment + ICP + render-compare — produces an accurate
metric scale estimate.

ICP source: generated Gaussian positions pre-aligned by height ratio to the MoGe pointmap.
ICP target: MoGe 3D points under the object SAM mask (depth-edge-cleaned).

Both live in metric MoGe space (metres).  The final revised_scale is the metric scale of
the canonical mesh and directly predicts object size in metres.

Three-way comparison:
  1. ICP-grounded scale: output["scale"].mean() after layout_post_optimization
  2. MetricScaleDecoder: output["metric_dimensions"].max() (if checkpoint loaded)
  3. Ground truth: max(W, H, D) from OmniNOCS

Evaluation split: same scene-heldout split used by the 3.09% MAPE baseline.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_pipeline(config_path: str, device: str, metric_checkpoint: str = None):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)
    config.compile_model = False
    config.device = device
    pipeline = instantiate(config)
    for model in pipeline.models.values():
        if model is None:
            continue
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    if metric_checkpoint:
        print(f"Loading metric scale checkpoint from {metric_checkpoint}")
        pipeline.load_metric_scale_checkpoint(metric_checkpoint)
        print("  MetricScaleDecoder loaded.")
    else:
        print("No metric checkpoint provided — MetricScaleDecoder predictions will be absent.")

    return pipeline


def build_split(dataset, overfit_n, heldout_n, seed, split_group):
    import random

    rng = random.Random(seed)
    records = dataset.records

    if split_group == "scene":
        from collections import defaultdict
        by_scene = defaultdict(list)
        for i, r in enumerate(records):
            scene = r.get("scene_id") or r.get("image_name", "").split("/")[2]
            by_scene[scene].append(i)
        scenes = sorted(by_scene.keys())
        rng.shuffle(scenes)
        heldout_scenes = set()
        heldout_indices = []
        for sc in reversed(scenes):
            if len(heldout_indices) >= heldout_n:
                break
            heldout_scenes.add(sc)
            heldout_indices.extend(by_scene[sc])
        train_indices = [i for i in range(len(records)) if i not in set(heldout_indices)]
        rng.shuffle(train_indices)
        train_indices = train_indices[:overfit_n]
        heldout_indices = heldout_indices[:heldout_n]
    else:
        all_idx = list(range(len(records)))
        rng.shuffle(all_idx)
        train_indices = all_idx[:overfit_n]
        heldout_indices = all_idx[overfit_n: overfit_n + heldout_n]

    return train_indices, heldout_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument("--metric-checkpoint", default=None,
                        help="Path to MetricScaleDecoder checkpoint (.pt) for 3-way comparison")
    parser.add_argument("--annotations-root",
                        default="/mnt/dest/OmniNOCS/omninocs_release_nocs_real275")
    parser.add_argument("--rgb-root", default="/mnt/dest/OmniNOCS/real_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-group", default="scene",
                        choices=["record", "image", "scene"])
    parser.add_argument("--overfit-samples", type=int, default=12882)
    parser.add_argument("--heldout-samples", type=int, default=3228)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Max samples to evaluate (subset of heldout split)")
    parser.add_argument("--eval-train", action="store_true",
                        help="Evaluate on train split instead of heldout")
    parser.add_argument("--stage1-steps", type=int, default=None,
                        help="SS inference steps (default: model config, currently 2)")
    parser.add_argument("--stage2-steps", type=int, default=None,
                        help="SLAT inference steps (default: model config, currently 12)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="/tmp/icp_scale_accuracy.json")
    args = parser.parse_args()

    from sam3d_objects.data.dataset.metric import OmniNOCSReal275Dataset

    print("Loading dataset...")
    dataset = OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split=args.split,
        min_mask_pixels=500,
    )
    print(f"Dataset size: {len(dataset)}")

    train_idx, heldout_idx = build_split(
        dataset, args.overfit_samples, args.heldout_samples,
        args.seed, args.split_group
    )
    eval_indices = train_idx if args.eval_train else heldout_idx
    print(f"Using {'train' if args.eval_train else 'heldout'} split: {len(eval_indices)} samples")

    if args.n_samples and args.n_samples < len(eval_indices):
        rng = np.random.default_rng(args.seed + 1)
        eval_indices = rng.choice(eval_indices, size=args.n_samples, replace=False).tolist()
    print(f"Evaluating on {len(eval_indices)} samples")

    print(f"Loading pipeline from {args.config}...")
    pipeline = load_pipeline(args.config, args.device, args.metric_checkpoint)

    results = []
    errors = 0

    for rank, idx in enumerate(eval_indices):
        item = dataset[idx]
        cat = item.get("category", "?")
        img_name = item.get("image_name", "?")
        W = float(item["metric_dims"][0])
        H = float(item["metric_dims"][1])
        D = float(item["metric_dims"][2])
        s_gt_max  = max(W, H, D)
        s_gt_cbrt = (W * H * D) ** (1.0 / 3.0)
        s_gt_mean = (W + H + D) / 3.0

        print(f"[{rank+1}/{len(eval_indices)}] {cat} | {img_name}", end=" ... ", flush=True)

        try:
            with torch.no_grad():
                output = pipeline.run(
                    image=item["image"],
                    with_layout_postprocess=True,
                    with_mesh_postprocess=False,
                    with_texture_baking=False,
                    stage1_inference_steps=args.stage1_steps,
                    stage2_inference_steps=args.stage2_steps,
                    decode_formats=["gaussian"],
                )

            # Detect whether layout_post_optimization actually ran.
            # If it ran, output["iou"] is set by run_post_optimization_GS.
            post_opt_ran = "iou" in output

            # ICP-grounded scale: revised_scale after layout_post_optimization.
            # Shape [1, 3], all channels equal after refine_scale. Units: metres.
            # Falls back to raw pose_decoder scale if post-opt failed/skipped.
            icp_scale_tensor = output.get("scale")
            if icp_scale_tensor is None:
                errors += 1
                print("no scale in output")
                continue
            icp_scale = float(icp_scale_tensor.mean())

            # MetricScaleDecoder prediction (if checkpoint loaded).
            # output["metric_dimensions"] is [batch, 3] in metres — use max dim.
            msd_scale = None
            msd_max = None
            if "metric_dimensions" in output:
                md = output["metric_dimensions"]
                msd_scale = float(md.max())
                msd_cbrt  = float((md.prod()) ** (1.0 / 3.0))
                msd_mean  = float(md.mean())
                msd_max   = msd_scale

            # ICP errors
            icp_pct_max  = abs(icp_scale - s_gt_max)  / max(s_gt_max,  1e-6) * 100
            icp_pct_cbrt = abs(icp_scale - s_gt_cbrt) / max(s_gt_cbrt, 1e-6) * 100
            icp_pct_mean = abs(icp_scale - s_gt_mean) / max(s_gt_mean, 1e-6) * 100

            row = {
                "category": cat,
                "image_name": img_name,
                "post_opt_ran": post_opt_ran,
                "icp_scale_m": icp_scale,
                "W_real": W, "H_real": H, "D_real": D,
                "s_gt_max": s_gt_max,
                "s_gt_cbrt": s_gt_cbrt,
                "s_gt_mean": s_gt_mean,
                "icp_pct_max": icp_pct_max,
                "icp_pct_cbrt": icp_pct_cbrt,
                "icp_pct_mean": icp_pct_mean,
            }

            msd_pct_max = None
            if msd_scale is not None:
                msd_pct_max  = abs(msd_scale - s_gt_max)  / max(s_gt_max,  1e-6) * 100
                msd_pct_cbrt = abs(msd_scale - s_gt_cbrt) / max(s_gt_cbrt, 1e-6) * 100
                msd_pct_mean = abs(msd_scale - s_gt_mean) / max(s_gt_mean, 1e-6) * 100
                row.update({
                    "msd_scale_max_m": msd_scale,
                    "msd_pct_max": msd_pct_max,
                    "msd_pct_cbrt": msd_pct_cbrt,
                    "msd_pct_mean": msd_pct_mean,
                })

            results.append(row)

            msg = (f"{'ICP' if post_opt_ran else 'raw'}  "
                   f"scale={icp_scale:.4f}m  gt_max={s_gt_max:.4f}m  "
                   f"MAPE={icp_pct_max:.1f}%")
            if msd_pct_max is not None:
                msg += f"  MAPE_msd={msd_pct_max:.1f}%"
            print(msg)

        except Exception as e:
            errors += 1
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    if not results:
        print("No results.")
        return

    def arr(key):
        return np.array([r[key] for r in results if key in r])

    icp_pct = arr("icp_pct_max")
    s_max   = arr("s_gt_max")
    icp_s   = arr("icp_scale_m")

    def corr(a, b):
        if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n" + "=" * 70)
    print("ICP-GROUNDED SCALE — ACCURACY vs OmniNOCS GROUND TRUTH")
    print("=" * 70)
    n_post_opt = sum(1 for r in results if r["post_opt_ran"])
    print(f"\nN={len(results)}  errors={errors}  post_opt_ran={n_post_opt}/{len(results)}")
    if n_post_opt < len(results):
        print(f"  WARNING: {len(results)-n_post_opt} samples fell back to raw pose_decoder scale (post-opt failed/skipped).")

    print("\n── ICP scale (metres) ──")
    print(f"  mean={icp_s.mean():.4f}  std={icp_s.std():.4f}  "
          f"min={icp_s.min():.4f}  max={icp_s.max():.4f}")

    print("\n── MAPE vs max(W,H,D)  ← primary comparison ──")
    print(f"  mean={icp_pct.mean():.2f}%  median={np.median(icp_pct):.2f}%  "
          f"std={icp_pct.std():.2f}%")
    print(f"  corr(icp_scale, max(WHD)) = {corr(icp_s, s_max):.3f}")

    print("\n── MAPE vs cbrt(W*H*D) ──")
    p = arr("icp_pct_cbrt")
    print(f"  mean={p.mean():.2f}%  median={np.median(p):.2f}%")

    print("\n── MAPE vs mean(W,H,D) ──")
    p = arr("icp_pct_mean")
    print(f"  mean={p.mean():.2f}%  median={np.median(p):.2f}%")

    print("\n── Per-category breakdown (MAPE vs max(W,H,D)) ──")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["icp_pct_max"])
    for cat in sorted(by_cat):
        v = np.array(by_cat[cat])
        print(f"  {cat:12s}: mean={v.mean():.2f}%  median={np.median(v):.2f}%  n={len(v)}")

    print("\n── Systematic bias (ICP) ──")
    ratio = icp_s / np.clip(s_max, 1e-6, None)
    print(f"  icp_scale / max(WHD): mean={ratio.mean():.3f}  std={ratio.std():.3f}")

    # MetricScaleDecoder comparison (if available)
    msd_pct = arr("msd_pct_max")
    if len(msd_pct) > 0:
        print(f"\n── MetricScaleDecoder MAPE vs max(W,H,D) ── (N={len(msd_pct)})")
        msd_s = arr("msd_scale_max_m")
        print(f"  mean={msd_pct.mean():.2f}%  median={np.median(msd_pct):.2f}%")
        print(f"  corr(msd_scale, max(WHD)) = {corr(msd_s, s_max[:len(msd_s)]):.3f}")

    print("\n── Baselines ──")
    print("  Raw SS scale token (1-step):                 MAPE ~324,812% (check_ss_scale_accuracy)")
    print("  Category mean-size baseline:                 10.90% MAPE")
    print("  Frozen-SLAT MetricScaleDecoder (3228 eval):   3.09% MAPE")
    mape = icp_pct.mean()
    print(f"\n  ICP-grounded scale (this run, N={len(results)}): {mape:.2f}% MAPE")
    if mape < 3.09:
        print("  ✓ ICP scale BEATS the trained MetricScaleDecoder baseline (3.09%).")
        print("    The depth pointmap is doing most of the work — consider whether")
        print("    MetricScaleDecoder is still needed.")
    elif mape < 10.90:
        print("  ~ ICP scale is better than mean-size but worse than MetricScaleDecoder.")
        print("    MetricScaleDecoder adds genuine value on top of geometry-based scale.")
    else:
        print("  ✗ ICP scale is no better than category mean-size baseline.")
        print("    Depth-based alignment is unreliable for metric scale on this data.")
    print("=" * 70)

    summary = {
        "n_results": len(results),
        "n_errors": errors,
        "icp_mape_max_mean": float(icp_pct.mean()),
        "icp_mape_max_median": float(np.median(icp_pct)),
        "icp_corr_max": corr(icp_s, s_max),
        "results": results,
    }
    if len(msd_pct) > 0:
        summary["msd_mape_max_mean"] = float(msd_pct.mean())
        summary["msd_mape_max_median"] = float(np.median(msd_pct))

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
