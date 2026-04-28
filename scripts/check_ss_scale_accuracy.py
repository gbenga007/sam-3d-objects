"""
Diagnostic: how accurate is the SS generator's explicit scale output against
OmniNOCS ground-truth metric dimensions?

The SS generator (MM-DiT, SparseStructureFlowTdfyWrapper) predicts a dedicated
scale token [B, 1, 3] trained with explicit supervision. This script checks
whether that signal is already metrically accurate — if so, our trained
MetricScaleDecoder may be adding little on top of what Stage 1 already knows.

Comparison target: max(W_real, H_real, D_real) from OmniNOCS.
This is the right target because max(canonical_bbox) ≈ 1.0 (confirmed in
planning/CANONICAL_MESH_BBOX_DIAGNOSTIC_2026-04-27.md), so the SS scale
factor converts canonical → metric via: metric_verts = canonical_verts * s.

Evaluation split: same scene-heldout 3228-sample eval split used by the
3.09% MAPE frozen-SLAT baseline, so numbers are directly comparable.

Conversion: metric_s = exp(raw_log_ssi_scale) * pointmap_scale
where raw_log_ssi_scale is the raw SS generator output (log-space SSI coords)
and pointmap_scale is the MoGe scene scale (mean abs value of shifted pointmap).
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


def load_pipeline(config_path: str, device: str):
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
    return pipeline


def get_ss_metric_scale(pipeline, image, device):
    """
    Run MoGe + SS and return the SS generator's predicted metric scale.

    Returns:
        ss_scale_metric: float — scalar object size in metres
            (= exp(raw_ssi_scale) * pointmap_scale, averaged over the 3 channels)
        pointmap_scale: float — MoGe scene scale
        is_mm_dit: bool
    """
    if not pipeline.is_mm_dit():
        return None, None, False

    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(image)
        ss_input = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor,
            pointmap=pointmap_dict["pointmap"],
        )
        pointmap_scale = ss_input.get("pointmap_scale")  # scalar tensor

        with torch.no_grad():
            ss_out = pipeline.sample_sparse_structure(
                ss_input, inference_steps=1, use_distillation=False
            )

    if "scale" not in ss_out:
        return None, None, True

    # Raw model output: log-scale in SSI-normalised coords, shape [B, 1, 3] or [B, 3]
    raw = ss_out["scale"]
    if raw.ndim == 3:
        raw = raw.squeeze(1)          # [B, 3]
    raw = raw.float()

    # SSI → metric: multiply by pointmap_scale
    ssi_scale = torch.exp(raw)         # [B, 3], SSI space
    if pointmap_scale is not None:
        ps = pointmap_scale.float().to(ssi_scale.device)
        # pointmap_scale may be scalar or [3]; broadcast safely
        metric_vec = ssi_scale * ps    # [B, 3]
    else:
        metric_vec = ssi_scale

    ss_scale_m = metric_vec.mean().item()   # scalar in metres
    ps_val = float(pointmap_scale.mean()) if pointmap_scale is not None else float("nan")
    return ss_scale_m, ps_val, True


def build_split(dataset, overfit_n, heldout_n, seed, split_group):
    """Replicate the scene-heldout split from finetune_metric_scale.py."""
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
    parser.add_argument("--annotations-root",
                        default="/mnt/dest/OmniNOCS/omninocs_release_nocs_real275")
    parser.add_argument("--rgb-root", default="/mnt/dest/OmniNOCS/real_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-group", default="scene",
                        choices=["record", "image", "scene"])
    parser.add_argument("--overfit-samples", type=int, default=12882)
    parser.add_argument("--heldout-samples", type=int, default=3228)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Max samples to evaluate (subset of heldout split)")
    parser.add_argument("--eval-train", action="store_true",
                        help="Evaluate on train split instead of heldout")
    parser.add_argument("--stage1-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="/tmp/ss_scale_accuracy.json")
    args = parser.parse_args()

    from sam3d_objects.data.dataset.metric import OmniNOCSReal275Dataset

    print(f"Loading dataset ...")
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

    # Subsample if requested
    if args.n_samples and args.n_samples < len(eval_indices):
        rng = np.random.default_rng(args.seed + 1)
        eval_indices = rng.choice(eval_indices, size=args.n_samples, replace=False).tolist()
    print(f"Evaluating on {len(eval_indices)} samples")

    print(f"Loading pipeline from {args.config} ...")
    pipeline = load_pipeline(args.config, args.device)

    if not pipeline.is_mm_dit():
        print("ERROR: SS generator is NOT MM-DiT — no explicit scale output available.")
        return

    print("SS generator is MM-DiT — explicit scale output confirmed.")

    results = []
    errors = 0

    for rank, idx in enumerate(eval_indices):
        item = dataset[idx]
        cat = item.get("category", "?")
        img_name = item.get("image_name", "?")
        W, H, D = float(item["metric_dims"][0]), float(item["metric_dims"][1]), float(item["metric_dims"][2])

        print(f"[{rank+1}/{len(eval_indices)}] {cat} | {img_name}", end=" ... ", flush=True)
        try:
            ss_scale, ps, _ = get_ss_metric_scale(pipeline, item["image"], args.device)
            if ss_scale is None:
                errors += 1
                print("no scale key")
                continue

            s_gt_max  = max(W, H, D)
            s_gt_cbrt = (W * H * D) ** (1.0 / 3.0)
            s_gt_mean = (W + H + D) / 3.0

            abs_pct_max  = abs(ss_scale - s_gt_max)  / max(s_gt_max,  1e-6) * 100
            abs_pct_cbrt = abs(ss_scale - s_gt_cbrt) / max(s_gt_cbrt, 1e-6) * 100
            abs_pct_mean = abs(ss_scale - s_gt_mean) / max(s_gt_mean, 1e-6) * 100

            results.append({
                "category": cat,
                "image_name": img_name,
                "ss_scale_m": ss_scale,
                "pointmap_scale": ps,
                "W_real": W, "H_real": H, "D_real": D,
                "s_gt_max": s_gt_max,
                "s_gt_cbrt": s_gt_cbrt,
                "s_gt_mean": s_gt_mean,
                "abs_pct_max": abs_pct_max,
                "abs_pct_cbrt": abs_pct_cbrt,
                "abs_pct_mean": abs_pct_mean,
            })
            print(
                f"ss={ss_scale:.4f}m  max(WHD)={s_gt_max:.4f}m  "
                f"MAPE_max={abs_pct_max:.1f}%"
            )
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")

    if not results:
        print("No results.")
        return

    # ── Summary ─────────────────────────────────────────────────────────────
    def arr(key): return np.array([r[key] for r in results])

    ss    = arr("ss_scale_m")
    s_max = arr("s_gt_max")
    s_cbrt= arr("s_gt_cbrt")
    s_mean= arr("s_gt_mean")
    pct_max  = arr("abs_pct_max")
    pct_cbrt = arr("abs_pct_cbrt")
    pct_mean = arr("abs_pct_mean")

    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n" + "=" * 65)
    print("SS GENERATOR EXPLICIT SCALE — ACCURACY vs OMNINOCS GROUND TRUTH")
    print("=" * 65)
    print(f"\nN={len(results)}  errors={errors}")

    print("\n── Predicted scalar scale (metres) ──")
    print(f"  mean={ss.mean():.4f}  std={ss.std():.4f}  "
          f"min={ss.min():.4f}  max={ss.max():.4f}")

    print("\n── MAPE vs max(W,H,D)  ← primary comparison ──")
    print(f"  mean={pct_max.mean():.2f}%  median={np.median(pct_max):.2f}%  "
          f"std={pct_max.std():.2f}%")
    print(f"  corr(ss_scale, max(WHD)) = {corr(ss, s_max):.3f}")

    print("\n── MAPE vs cbrt(W*H*D) ──")
    print(f"  mean={pct_cbrt.mean():.2f}%  median={np.median(pct_cbrt):.2f}%")

    print("\n── MAPE vs mean(W,H,D) ──")
    print(f"  mean={pct_mean.mean():.2f}%  median={np.median(pct_mean):.2f}%")

    print("\n── Per-category breakdown (MAPE vs max(W,H,D)) ──")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["abs_pct_max"])
    for cat in sorted(by_cat):
        vals = np.array(by_cat[cat])
        print(f"  {cat:12s}: mean={vals.mean():.2f}%  median={np.median(vals):.2f}%  n={len(vals)}")

    print("\n── Systematic bias ──")
    ratio = ss / np.clip(s_max, 1e-6, None)
    print(f"  ss_scale / max(WHD): mean={ratio.mean():.3f}  std={ratio.std():.3f}")
    print(f"  (1.0 = no bias; >1 = model overestimates; <1 = underestimates)")

    print("\n── Comparison baseline ──")
    print("  Frozen-SLAT MetricScaleDecoder (scene-heldout, 3228 samples): 3.09% MAPE")
    print("  Category mean-size baseline: 10.90% MAPE")
    mape = pct_max.mean()
    if mape < 3.09:
        print(f"\n  ✓ SS scale ({mape:.2f}%) BEATS the trained decoder baseline (3.09%).")
        print("    MetricScaleDecoder may be adding little — investigate further.")
    elif mape < 10.90:
        print(f"\n  ~ SS scale ({mape:.2f}%) is better than mean-size but worse than trained decoder.")
        print("    The trained decoder is worth keeping; SS scale is a useful input feature.")
    else:
        print(f"\n  ✗ SS scale ({mape:.2f}%) is no better than category mean-size baseline.")
        print("    The explicit scale token is unreliable for absolute metric prediction.")
    print("=" * 65)

    with open(args.output, "w") as f:
        json.dump({"results": results, "n_errors": errors,
                   "mape_max_mean": float(pct_max.mean()),
                   "mape_max_median": float(np.median(pct_max)),
                   "corr_ss_max": corr(ss, s_max)}, f, indent=2)
    print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
