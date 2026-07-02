#!/usr/bin/env python3
"""
Mesh bounding-box accuracy: isotropic vs per-axis scaling.

For each sample the script runs the full pipeline (SS → SLAT → mesh decode +
metric head) with the given checkpoint and compares two inference-time scaling
strategies against GT [W, H, D]:

  isotropic  — current production behaviour: scale all mesh axes by
               s = max(W_pred, H_pred, D_pred).  The mesh shape comes from SS/SLAT;
               only the overall size is controlled by the metric head.

  per-axis   — ideal upper bound: scale each mesh axis independently so the
               resulting bbox exactly equals the predicted [W, H, D] dims.
               Error = MetricScaleDecoder MAPE (currently 1.74%).

The gap between the two strategies quantifies how much the wrong voxel aspect
ratio costs in terms of absolute dimensional accuracy.

Both strategies use rank-sorted comparison (largest mesh axis matched to largest
GT dim) to avoid sensitivity to canonical-frame axis labelling.

Usage:
    python scripts/mesh_scale_eval.py \\
        --checkpoint artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v3_best.pt \\
        --config checkpoints/hf/pipeline.yaml \\
        --annotations-root /mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_nocs_real275 \\
        --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \\
        --n-samples 64 \\
        --stage1-steps 4 --stage2-steps 1 \\
        --output /tmp/mesh_scale_eval.jsonl \\
        --output-dir /tmp/mesh_scale_eval_plys
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
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
)


# ---------------------------------------------------------------------------
# Pipeline loading
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
# Bounding-box helper
# ---------------------------------------------------------------------------

def mesh_bbox_extents(verts: np.ndarray) -> np.ndarray:
    """Return per-axis extents [dx, dy, dz] from vertex array [N, 3]."""
    return verts.max(axis=0) - verts.min(axis=0)  # [3]


# ---------------------------------------------------------------------------
# PLY saving
# ---------------------------------------------------------------------------

def save_ply(verts: np.ndarray, faces: np.ndarray, path: str):
    """Save mesh as PLY using trimesh."""
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.export(path)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def run_sample(pipeline, scale_head, scale_decoder, item, stage1_steps, stage2_steps, device):
    """
    Returns a dict with error metrics and mesh vertex arrays for PLY export:

      gt_dims         [3] GT metric dims in metres (rank-sorted desc)
      pred_dims       [3] MetricScaleDecoder prediction (rank-sorted desc)
      raw_extents     [3] raw mesh bbox extents in canonical space (rank-sorted desc)
      scaled_iso      [3] mesh bbox after isotropic scaling (rank-sorted desc), metres
      scaled_peraxis  [3] mesh bbox after per-axis scaling (rank-sorted desc), metres

      verts_canonical [N,3] original mesh vertices in canonical (unit-cube) space
      verts_iso       [N,3] vertices after isotropic scaling (metres)
      verts_peraxis   [N,3] vertices after rank-matched per-axis scaling (metres)
      faces           [F,3] triangle face indices (shared across all three)
    """
    image = item["image"]  # numpy [H,W,4] uint8; pipeline handles device internally
    gt_dims = np.sort(np.array(item["metric_dims"], dtype=np.float32))[::-1].copy()  # [3] descending

    with torch.no_grad():
        pointmap_dict = pipeline.compute_pointmap(image)
        ss_input_dict = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input_dict = pipeline.preprocess_image(image, pipeline.slat_preprocessor)

        # --- Stage 1: SS ---
        ss_return_dict = pipeline.sample_sparse_structure(
            ss_input_dict,
            inference_steps=stage1_steps,
            use_distillation=False,
        )

        # --- Metric scale token ---
        ss_scale_features = extract_ss_scale_features(ss_return_dict)
        if ss_scale_features is not None:
            ss_scale_features = ss_scale_features.to(device)
        scale_token = scale_head(
            ss_return_dict["shape"],
            ss_scale_features,
            ss_input_dict.get("pointmap_scale"),
            ss_input_dict.get("pointmap_shift"),
        )

        # Inject scale token into SLAT conditioning (mirrors inference pipeline behaviour)
        import torch.nn.functional as F
        token_for_cond = F.layer_norm(scale_token, [scale_token.shape[-1]])
        slat_backbone = pipeline._get_slat_backbone()
        orig_backbone_emb = None
        orig_external_emb = None
        if slat_backbone is not None:
            orig_backbone_emb = slat_backbone.condition_embedder
            slat_backbone.condition_embedder = _ScaleAugmentedEmbedderProxy(
                orig_backbone_emb, token_for_cond
            )
        cond_embedders = getattr(pipeline, "condition_embedders", {})
        orig_external_emb = cond_embedders.get("slat_condition_embedder")
        if orig_external_emb is not None:
            pipeline.condition_embedders["slat_condition_embedder"] = _ScaleAugmentedEmbedderProxy(
                orig_external_emb, token_for_cond
            )

        try:
            slat = pipeline.sample_slat(
                slat_input_dict,
                ss_return_dict["coords"],
                inference_steps=stage2_steps,
                use_distillation=False,
            )
        finally:
            if orig_backbone_emb is not None and slat_backbone is not None:
                slat_backbone.condition_embedder = orig_backbone_emb
            if orig_external_emb is not None:
                pipeline.condition_embedders["slat_condition_embedder"] = orig_external_emb

        # --- Decode mesh ---
        decoded = pipeline.decode_slat(slat, formats=["mesh"])
        mesh = decoded["mesh"][0]

        # --- Metric prediction ---
        log_dims = scale_decoder(slat.feats, scale_token, slat.coords[:, 0])
        pred_dims = torch.exp(log_dims[0]).cpu().numpy()  # [3] in metres, order = [W, H, D]

    # --- Raw mesh vertices/faces ---
    verts_canonical = mesh.vertices.float().cpu().numpy()  # [N, 3]
    faces = mesh.faces.int().cpu().numpy()                  # [F, 3]

    raw_ext = mesh_bbox_extents(verts_canonical)   # [3], per-axis extents
    pred_sorted = np.sort(pred_dims)[::-1].copy()  # [3] descending
    raw_sorted = np.sort(raw_ext)[::-1].copy()     # [3] descending

    # --- Isotropic scaling ---
    s_iso = pred_dims.max()
    verts_iso = verts_canonical * s_iso
    scaled_iso_sorted = np.sort(mesh_bbox_extents(verts_iso))[::-1].copy()

    # --- Per-axis rank-matched scaling ---
    # Identify which mesh axis has the largest/middle/smallest extent, then
    # assign the corresponding predicted dimension as the target for that axis.
    ext_rank = np.argsort(raw_ext)[::-1]  # ext_rank[0] = axis index of largest extent
    scale_factors = np.ones(3, dtype=np.float64)
    center = (verts_canonical.min(axis=0) + verts_canonical.max(axis=0)) / 2.0
    for rank, axis in enumerate(ext_rank):
        if raw_ext[axis] > 1e-6:
            scale_factors[axis] = pred_sorted[rank] / raw_ext[axis]
        else:
            scale_factors[axis] = 1.0
    verts_peraxis = center + (verts_canonical - center) * scale_factors
    scaled_peraxis_sorted = np.sort(mesh_bbox_extents(verts_peraxis))[::-1].copy()

    return {
        "category": item.get("category", "unknown"),
        "gt_dims": gt_dims.tolist(),
        "pred_dims": pred_sorted.tolist(),
        "raw_extents": raw_sorted.tolist(),
        "scaled_iso": scaled_iso_sorted.tolist(),
        "scaled_peraxis": scaled_peraxis_sorted.tolist(),
        # mesh arrays for PLY export
        "verts_canonical": verts_canonical,
        "verts_iso": verts_iso,
        "verts_peraxis": verts_peraxis.astype(np.float32),
        "faces": faces,
    }


def pct_errors(predicted: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Absolute percentage errors per axis."""
    return np.abs(predicted - gt) / np.clip(gt, 1e-6, None) * 100.0


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def print_summary(records: list[dict]):
    iso_mapes, peraxis_mapes = [], []
    iso_per_axis = [[], [], []]
    peraxis_per_axis = [[], [], []]

    for r in records:
        gt = np.array(r["gt_dims"])
        iso = np.array(r["scaled_iso"])
        perax = np.array(r["scaled_peraxis"])

        errs_iso = pct_errors(iso, gt)
        errs_perax = pct_errors(perax, gt)

        iso_mapes.append(errs_iso.mean())
        peraxis_mapes.append(errs_perax.mean())
        for i in range(3):
            iso_per_axis[i].append(errs_iso[i])
            peraxis_per_axis[i].append(errs_perax[i])

    axis_labels = ["largest", "middle", "smallest"]
    print(f"\n{'='*60}")
    print(f"Mesh bbox eval — {len(records)} samples")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Isotropic':>10} {'Per-axis':>10}")
    print(f"{'-'*50}")
    print(f"{'Overall MAPE':<30} {np.mean(iso_mapes):>9.2f}% {np.mean(peraxis_mapes):>9.2f}%")
    print(f"{'Median MAPE':<30} {np.median(iso_mapes):>9.2f}% {np.median(peraxis_mapes):>9.2f}%")
    for i, label in enumerate(axis_labels):
        print(
            f"{'  ' + label + ' axis MAPE':<30} "
            f"{np.mean(iso_per_axis[i]):>9.2f}% "
            f"{np.mean(peraxis_per_axis[i]):>9.2f}%"
        )

    print(f"\nAspect ratio gap (iso - peraxis): "
          f"{np.mean(iso_mapes) - np.mean(peraxis_mapes):+.2f}pp")
    print(f"  (positive = isotropic is worse; target: reduce this after SS ratio loss training)")
    print(f"{'='*60}\n")

    # Per-category breakdown
    categories = {}
    for r in records:
        cat = r.get("category", "unknown")
        gt = np.array(r["gt_dims"])
        iso = np.array(r["scaled_iso"])
        perax = np.array(r["scaled_peraxis"])
        if cat not in categories:
            categories[cat] = {"iso": [], "peraxis": []}
        categories[cat]["iso"].append(pct_errors(iso, gt).mean())
        categories[cat]["peraxis"].append(pct_errors(perax, gt).mean())

    if len(categories) > 1:
        print(f"{'Category':<16} {'N':>4} {'Iso MAPE':>10} {'Peraxis MAPE':>13}")
        print("-" * 45)
        for cat, vals in sorted(categories.items()):
            print(
                f"{cat:<16} {len(vals['iso']):>4} "
                f"{np.mean(vals['iso']):>9.2f}% "
                f"{np.mean(vals['peraxis']):>12.2f}%"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--annotations-root",
        default="/mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_nocs_real275",
    )
    parser.add_argument("--rgb-root", default="/mnt/source/datasets_sam3d/OmniNOCS/real_test")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=64,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-steps", type=int, default=4)
    parser.add_argument("--stage2-steps", type=int, default=1)
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None,
                        help="Optional JSONL path for per-sample results (no mesh arrays)")
    parser.add_argument("--output-dir", default=None,
                        help="Optional directory to save PLY files (canonical/iso/peraxis per sample)")
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.config} ...")
    pipeline = load_pipeline(args.config, args.device)

    scale_head = MetricScaleHead().to(args.device)
    scale_decoder = MetricScaleDecoder().to(args.device)
    load_metric_checkpoint(args.checkpoint, scale_head, scale_decoder)
    scale_head.eval()
    scale_decoder.eval()
    print(f"Loaded metric checkpoint from {args.checkpoint}")

    dataset = OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split=args.split,
        categories=args.categories,
        min_mask_pixels=args.min_mask_pixels,
    )
    print(f"Dataset: {len(dataset)} records")

    rng = np.random.default_rng(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    if args.n_samples and args.n_samples < len(indices):
        indices = indices[: args.n_samples]
    print(f"Evaluating {len(indices)} samples ...")

    records = []
    skipped = 0
    out_fh = open(args.output, "w") if args.output else None

    for sample_i, idx in enumerate(tqdm(indices)):
        item = dataset[idx]
        if int(item.get("mask_pixels", 9999)) < args.min_mask_pixels:
            skipped += 1
            continue
        try:
            record = run_sample(
                pipeline, scale_head, scale_decoder, item,
                args.stage1_steps, args.stage2_steps, args.device,
            )

            # Save PLYs before stripping mesh arrays
            if args.output_dir:
                cat = record["category"]
                stem = f"{cat}_{sample_i:03d}"
                save_ply(
                    record["verts_canonical"], record["faces"],
                    os.path.join(args.output_dir, f"{stem}_canonical.ply"),
                )
                save_ply(
                    record["verts_iso"], record["faces"],
                    os.path.join(args.output_dir, f"{stem}_iso.ply"),
                )
                save_ply(
                    record["verts_peraxis"], record["faces"],
                    os.path.join(args.output_dir, f"{stem}_peraxis.ply"),
                )

            # Strip mesh arrays before keeping in memory / writing to JSONL
            serialisable = {k: v for k, v in record.items()
                            if k not in ("verts_canonical", "verts_iso", "verts_peraxis", "faces")}
            records.append(serialisable)
            if out_fh:
                out_fh.write(json.dumps(serialisable) + "\n")
                out_fh.flush()
        except Exception as e:
            print(f"  [skip idx={idx}] {e}")
            skipped += 1

    if out_fh:
        out_fh.close()

    if skipped:
        print(f"Skipped {skipped} samples (mask too small or error).")

    print_summary(records)

    if args.output:
        print(f"Per-sample results written to {args.output}")
    if args.output_dir:
        print(f"PLY files written to {args.output_dir}")


if __name__ == "__main__":
    main()
