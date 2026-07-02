#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Evaluate a trained metric-scale checkpoint on the NOCS Real275 test split.

For each example:
  1. Runs the full SAM3D pipeline (SS + SLAT) with the metric scale token injected.
  2. Predicts [W, H, D] in metres from the trained MetricScaleHead + MetricScaleDecoder.
  3. Decodes a mesh from the SLAT latent.
  4. Scales the mesh so max(bbox) == max(pred_W, pred_H, pred_D) and saves it.
  5. Records predicted vs GT dimensions and per-axis errors.

Output directory layout:
    <output_dir>/
        results.csv              — per-sample: uid, category, pred, gt, errors
        results_summary.json     — per-category aggregates + overall
        meshes/<uid>.glb         — metric-scaled geometry (no texture, fast)

Usage:
    python scripts/eval_metric_scale.py \
        --checkpoint artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v2_best.pt \
        --config checkpoints/hf/pipeline.yaml \
        --annotations-root /mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_nocs_real275 \
        --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \
        --split test \
        --output-dir artifacts/metric_scale/eval_slat_conditioned_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from sam3d_objects.data.dataset.metric.nocs import OmniNOCSReal275Dataset
from sam3d_objects.model.backbone.metric_scale_decoder import MetricScaleDecoder
from sam3d_objects.model.backbone.scale_head import (
    MetricScaleHead,
    _ScaleAugmentedEmbedderProxy,
    extract_ss_scale_features,
)
from sam3d_objects.training.finetune_metric_scale import (
    freeze_pipeline,
    load_metric_checkpoint,
    make_train_eval_subsets,
)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_pipeline(config_path: str, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)
    config.compile_model = False
    config.device = device
    pipeline = instantiate(config)
    freeze_pipeline(pipeline)
    return pipeline


# ---------------------------------------------------------------------------
# Per-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_single(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    image: np.ndarray,
    stage1_steps: int | None,
    stage2_steps: int | None,
    inject_scale_token: bool,
    decode_mesh: bool,
) -> tuple[np.ndarray, object | None]:
    """
    Run the metric-scale pipeline on a single RGBA image.

    Returns:
        pred_dims_m  — np.ndarray [3], [W, H, D] in metres
        mesh_result  — MeshExtractResult or None
    """
    with pipeline.device:
        pointmap = pipeline.compute_pointmap(image)
        ss_input = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor, pointmap=pointmap["pointmap"]
        )
        slat_input = pipeline.preprocess_image(image, pipeline.slat_preprocessor)

        ss_out = pipeline.sample_sparse_structure(
            ss_input, inference_steps=stage1_steps, use_distillation=False
        )
        ss_scale_feats = extract_ss_scale_features(ss_out)
        if ss_scale_feats is not None:
            ss_scale_feats = ss_scale_feats.detach().to(pipeline.device)

        scale_token = scale_head(
            ss_out["shape"].detach(),
            ss_scale_feats,
            ss_input.get("pointmap_scale"),
            ss_input.get("pointmap_shift"),
        )

        # Inject scale token into SLAT conditioning (detached — inference only)
        orig_bb_emb = orig_ext_emb = slat_bb = None
        if inject_scale_token and not hasattr(pipeline, "_get_slat_backbone"):
            print(
                "Warning: --inject-scale-token is set but the pipeline has no "
                "_get_slat_backbone method. Scale token will NOT be injected into "
                "SLAT conditioning. Use InferencePipelinePointMap."
            )
        if inject_scale_token and hasattr(pipeline, "_get_slat_backbone"):
            tok = F.layer_norm(scale_token, [scale_token.shape[-1]]).detach()
            slat_bb = pipeline._get_slat_backbone()
            if slat_bb is not None:
                orig_bb_emb = slat_bb.condition_embedder
                slat_bb.condition_embedder = _ScaleAugmentedEmbedderProxy(orig_bb_emb, tok)
            cond_embs = getattr(pipeline, "condition_embedders", {})
            orig_ext_emb = cond_embs.get("slat_condition_embedder")
            if orig_ext_emb is not None:
                pipeline.condition_embedders["slat_condition_embedder"] = (
                    _ScaleAugmentedEmbedderProxy(orig_ext_emb, tok)
                )

        try:
            slat = pipeline.sample_slat(
                slat_input,
                ss_out["coords"],
                inference_steps=stage2_steps,
                use_distillation=False,
                with_grad=False,
            )
        finally:
            if slat_bb is not None and orig_bb_emb is not None:
                slat_bb.condition_embedder = orig_bb_emb
            if orig_ext_emb is not None:
                pipeline.condition_embedders["slat_condition_embedder"] = orig_ext_emb

        log_dims = scale_decoder(slat.feats.detach(), scale_token, slat.coords[:, 0])
        pred_dims = torch.exp(log_dims[0]).cpu().numpy()  # [W, H, D] metres

        mesh_raw = None
        if decode_mesh:
            decoded = pipeline.decode_slat(slat, formats=["mesh"])
            raw = decoded.get("mesh")
            # decode_slat may return a list [MeshExtractResult] or a single
            # MeshExtractResult depending on the decoder wrapper.
            if isinstance(raw, (list, tuple)):
                mesh_raw = raw[0] if raw else None
            else:
                mesh_raw = raw

    return pred_dims, mesh_raw


# ---------------------------------------------------------------------------
# Mesh scaling
# ---------------------------------------------------------------------------

def mesh_to_metric_trimesh(
    mesh_result,
    pred_dims: np.ndarray,
    per_axis: bool = False,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    Scale a MeshExtractResult to metric dimensions.

    per_axis=False (default): isotropic — one scale = max(pred_dims)/max(mesh_extents).
    per_axis=True: rank-matched per-axis — longest mesh axis → largest pred dim, etc.

    Always returns (scaled_trimesh, scale_xyz) where scale_xyz is [sx, sy, sz].
    For isotropic mode all three values are equal.
    """
    verts = mesh_result.vertices.float().cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    extents = tm.bounding_box.extents  # [ex, ey, ez] in normalized units
    if extents.max() < 1e-8:
        return tm, np.ones(3)

    if per_axis:
        mesh_rank = np.argsort(extents)[::-1]
        pred_rank = np.argsort(pred_dims)[::-1]
        scale_xyz = np.ones(3)
        for rank in range(3):
            ax = mesh_rank[rank]
            scale_xyz[ax] = pred_dims[pred_rank[rank]] / extents[ax]
    else:
        s = float(pred_dims.max()) / float(extents.max())
        scale_xyz = np.full(3, s)

    tm.vertices = tm.vertices * scale_xyz
    return tm, scale_xyz


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    rel = np.abs(pred - gt) / np.clip(gt, 1e-6, None)
    abs_cm = np.abs(pred - gt) * 100.0
    return {
        "mean_abs_pct": float(rel.mean() * 100),
        "median_abs_pct": float(np.median(rel) * 100),
        "axis_abs_pct_W": float(rel[0] * 100),
        "axis_abs_pct_H": float(rel[1] * 100),
        "axis_abs_pct_D": float(rel[2] * 100),
        "mean_abs_cm": float(abs_cm.mean()),
        "median_abs_cm": float(np.median(abs_cm)),
        "axis_abs_cm_W": float(abs_cm[0]),
        "axis_abs_cm_H": float(abs_cm[1]),
        "axis_abs_cm_D": float(abs_cm[2]),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results_table(results: list[dict]):
    cols = [
        ("uid", 32), ("cat", 8),
        ("pW_cm", 7), ("pH_cm", 7), ("pD_cm", 7),
        ("gW_cm", 7), ("gH_cm", 7), ("gD_cm", 7),
        ("err%", 6), ("err_cm", 7), ("scale", 14),
    ]
    header = "  ".join(h.ljust(w) for h, w in cols)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        p = r["pred_dims"] * 100
        g = r["gt_dims"] * 100
        sx, sy, sz = r["scale_xyz"]
        if np.allclose(sx, sy) and np.allclose(sy, sz):
            scale_str = f"{sx:.3f}"
        else:
            scale_str = f"{sx:.2f}/{sy:.2f}/{sz:.2f}"
        row = [
            r["uid"][:32],
            r["category"][:8],
            f"{p[0]:.1f}", f"{p[1]:.1f}", f"{p[2]:.1f}",
            f"{g[0]:.1f}", f"{g[1]:.1f}", f"{g[2]:.1f}",
            f"{r['mean_abs_pct']:.1f}%",
            f"{r['mean_abs_cm']:.2f}",
            scale_str,
        ]
        print("  ".join(str(v).ljust(w) for v, (_, w) in zip(row, cols)))
    print(sep)


def print_category_summary(results: list[dict]):
    by_cat: dict[str, list] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print("\nPer-category summary:")
    fmt = "  {:<12}  n={:4d}  mean={:5.1f}%  median={:5.1f}%  mean_cm={:5.2f}  median_cm={:5.2f}"
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        pcts = [r["mean_abs_pct"] for r in rs]
        cms  = [r["mean_abs_cm"]  for r in rs]
        print(fmt.format(cat, len(rs), np.mean(pcts), np.median(pcts), np.mean(cms), np.median(cms)))

    pcts = [r["mean_abs_pct"] for r in results]
    cms  = [r["mean_abs_cm"]  for r in results]
    print(fmt.format("OVERALL", len(results), np.mean(pcts), np.median(pcts), np.mean(cms), np.median(cms)))


def write_csv(path: str, results: list[dict]):
    fieldnames = [
        "uid", "category", "image_name",
        "pred_W_m", "pred_H_m", "pred_D_m",
        "gt_W_m",   "gt_H_m",  "gt_D_m",
        "err_W_pct", "err_H_pct", "err_D_pct",
        "mean_abs_pct", "median_abs_pct",
        "mean_abs_cm",  "median_abs_cm",
        "scale_x", "scale_y", "scale_z",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            p, g = r["pred_dims"], r["gt_dims"]
            sx, sy, sz = r["scale_xyz"]
            w.writerow({
                "uid":          r["uid"],
                "category":     r["category"],
                "image_name":   r.get("image_name", ""),
                "pred_W_m":     f"{p[0]:.6f}",
                "pred_H_m":     f"{p[1]:.6f}",
                "pred_D_m":     f"{p[2]:.6f}",
                "gt_W_m":       f"{g[0]:.6f}",
                "gt_H_m":       f"{g[1]:.6f}",
                "gt_D_m":       f"{g[2]:.6f}",
                "err_W_pct":    f"{r['axis_abs_pct_W']:.2f}",
                "err_H_pct":    f"{r['axis_abs_pct_H']:.2f}",
                "err_D_pct":    f"{r['axis_abs_pct_D']:.2f}",
                "mean_abs_pct": f"{r['mean_abs_pct']:.4f}",
                "median_abs_pct": f"{r['median_abs_pct']:.4f}",
                "mean_abs_cm":  f"{r['mean_abs_cm']:.4f}",
                "median_abs_cm": f"{r['median_abs_cm']:.4f}",
                "scale_x":      f"{sx:.6f}",
                "scale_y":      f"{sy:.6f}",
                "scale_z":      f"{sz:.6f}",
            })


def write_summary(path: str, results: list[dict]):
    by_cat: dict[str, list] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    def agg(rs):
        pcts = [r["mean_abs_pct"] for r in rs]
        cms  = [r["mean_abs_cm"]  for r in rs]
        ax_pcts = {
            "W": [r["axis_abs_pct_W"] for r in rs],
            "H": [r["axis_abs_pct_H"] for r in rs],
            "D": [r["axis_abs_pct_D"] for r in rs],
        }
        return {
            "n":               len(rs),
            "mean_abs_pct":    float(np.mean(pcts)),
            "median_abs_pct":  float(np.median(pcts)),
            "mean_abs_cm":     float(np.mean(cms)),
            "median_abs_cm":   float(np.median(cms)),
            "axis_mean_abs_pct": {ax: float(np.mean(v)) for ax, v in ax_pcts.items()},
        }

    summary = {
        "overall":     agg(results),
        "by_category": {cat: agg(rs) for cat, rs in sorted(by_cat.items())},
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Metric scale checkpoint produced by finetune_metric_scale.py",
    )
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument(
        "--annotations-root",
        default="/mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_nocs_real275",
    )
    parser.add_argument(
        "--rgb-root",
        default="/mnt/source/datasets_sam3d/OmniNOCS/real_test",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Restrict to these categories (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate at most N samples (useful for quick smoke-tests)")
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--output-dir", required=True,
                        help="Directory for results.csv, summary.json, and meshes/")
    parser.add_argument("--stage1-steps", type=int, default=4,
                        help="SS flow-matching inference steps")
    parser.add_argument("--stage2-steps", type=int, default=1,
                        help="SLAT flow-matching inference steps")
    parser.add_argument(
        "--inject-scale-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject the metric scale token into SLAT conditioning at inference time",
    )
    parser.add_argument(
        "--restore-slat-cross-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore trained SLAT cross-attn weights from the checkpoint",
    )
    parser.add_argument(
        "--decode-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode and save a metric-scaled mesh for each example",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--heldout-samples", type=int, default=0,
        help="If >0, replicate the training split: evaluate only on the last N samples "
             "(same logic as finetune_metric_scale.py --heldout-samples). "
             "Use --seed and --split-group to match the training run exactly.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-group", default="record",
                        choices=["record", "image", "scene"])
    parser.add_argument(
        "--scale-mode", default="isotropic", choices=["isotropic", "per-axis"],
        help="isotropic: one scale = max(pred)/max(mesh_bbox). "
             "per-axis: rank-matched scale per axis.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.decode_mesh:
        (out_dir / "meshes").mkdir(exist_ok=True)

    # ── Pipeline ──────────────────────────────────────────────────────────
    print(f"Loading pipeline from {args.config} ...")
    pipeline = load_pipeline(args.config, args.device)

    scale_head    = MetricScaleHead().to(args.device).eval()
    scale_decoder = MetricScaleDecoder().to(args.device).eval()

    slat_backbone = None
    if args.restore_slat_cross_attn:
        try:
            slat_backbone = pipeline.models["slat_generator"].reverse_fn.backbone
        except (AttributeError, KeyError):
            print("Warning: could not locate slat_generator backbone — skipping cross-attn restore")

    load_metric_checkpoint(args.checkpoint, scale_head, scale_decoder, slat_backbone)

    # ── Dataset ───────────────────────────────────────────────────────────
    full_dataset = OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split=args.split,
        categories=args.categories,
        min_mask_pixels=args.min_mask_pixels,
        max_records=args.max_samples,
    )
    if args.heldout_samples > 0:
        _, eval_subset = make_train_eval_subsets(
            full_dataset,
            train_samples=0,
            heldout_samples=args.heldout_samples,
            seed=args.seed,
            shuffle_split=False,
            split_group=args.split_group,
        )
        dataset = eval_subset
        print(f"Dataset: {len(dataset)} held-out records  (split={args.split}, heldout={args.heldout_samples})")
    else:
        dataset = full_dataset
        print(f"Dataset: {len(dataset)} records  (split={args.split})")

    # ── Evaluation loop ───────────────────────────────────────────────────
    results: list[dict] = []
    skipped = 0

    for item in tqdm(dataset, desc="eval"):
        if int(item.get("mask_pixels", args.min_mask_pixels)) < args.min_mask_pixels:
            skipped += 1
            continue

        uid = item["uid"]
        try:
            pred_dims, mesh_result = run_single(
                pipeline, scale_head, scale_decoder,
                item["image"],
                args.stage1_steps,
                args.stage2_steps,
                inject_scale_token=args.inject_scale_token,
                decode_mesh=args.decode_mesh,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n[OOM] {uid} — skipped")
            skipped += 1
            continue
        except Exception as exc:
            print(f"\n[error] {uid}: {exc}")
            skipped += 1
            continue

        gt_dims = np.array(item["metric_dims"], dtype=np.float32)
        errors  = compute_errors(pred_dims, gt_dims)
        scale_xyz = np.zeros(3)

        if args.decode_mesh and mesh_result is not None and mesh_result.success:
            try:
                scaled_mesh, scale_xyz = mesh_to_metric_trimesh(
                    mesh_result, pred_dims, per_axis=(args.scale_mode == "per-axis")
                )
                mesh_path = out_dir / "meshes" / f"{uid}.ply"
                scaled_mesh.export(str(mesh_path))
            except Exception as exc:
                print(f"\n[mesh-export-error] {uid}: {exc}")

        results.append({
            "uid":       uid,
            "category":  item["category"],
            "image_name": item.get("image_name", ""),
            "pred_dims": pred_dims,
            "gt_dims":   gt_dims,
            "scale_xyz": scale_xyz,
            **errors,
        })

    # ── Output ────────────────────────────────────────────────────────────
    if not results:
        print("No results collected.")
        return

    print_results_table(results)
    print_category_summary(results)
    print(f"\nTotal: {len(results)} evaluated, {skipped} skipped")

    csv_path     = out_dir / "results.csv"
    summary_path = out_dir / "results_summary.json"
    write_csv(str(csv_path), results)
    write_summary(str(summary_path), results)
    print(f"\nresults.csv    → {csv_path}")
    print(f"summary.json   → {summary_path}")
    if args.decode_mesh:
        print(f"meshes/        → {out_dir / 'meshes'}")


if __name__ == "__main__":
    main()
