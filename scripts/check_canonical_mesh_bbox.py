"""
Diagnostic: how does the canonical mesh bounding box relate to real-world WHD?

Runs SAM3D inference on a sample of NOCS-Real275 objects, extracts the raw
FlexiCubes vertex bounding box (before any postprocessing), and compares it
against the ground-truth metric dimensions from OmniNOCS.

Key question: is bbox ≈ 1.0 per axis, i.e. does SAM3D fill the unit cube?
If yes, predicting [W, H, D] IS predicting the direct per-axis mesh scale.
If no, a post-hoc bbox division is needed at inference time.
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


def run_one(pipeline, item, stage1_steps, stage2_steps, device):
    """Run MoGe → SS → SLAT → mesh decoder. Return raw vertex bbox and metadata."""
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(item["image"])
        ss_input = pipeline.preprocess_image(
            item["image"], pipeline.ss_preprocessor,
            pointmap=pointmap_dict["pointmap"],
        )
        slat_input = pipeline.preprocess_image(item["image"], pipeline.slat_preprocessor)

        with torch.no_grad():
            ss_out = pipeline.sample_sparse_structure(
                ss_input, inference_steps=stage1_steps, use_distillation=False
            )
            slat = pipeline.sample_slat(
                slat_input, ss_out["coords"],
                inference_steps=stage2_steps, use_distillation=False,
            )
            decoded = pipeline.decode_slat(slat, formats=["mesh"])

    mesh = decoded["mesh"][0]   # MeshExtractResult
    verts = mesh.vertices.float().cpu()  # [N, 3] in canonical space

    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    bbox = (vmax - vmin).numpy()          # [bbox_x, bbox_y, bbox_z]
    center = ((vmin + vmax) / 2).numpy() # should be near [0, 0, 0]

    return {
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "n_verts": int(verts.shape[0]),
        "W_real": float(item["metric_dims"][0]),
        "H_real": float(item["metric_dims"][1]),
        "D_real": float(item["metric_dims"][2]),
        "category": item.get("category", "unknown"),
        "source": item.get("source", "unknown"),
        "image_name": item.get("image_name", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument("--annotations-root",
                        default="/mnt/dest/OmniNOCS/omninocs_release_nocs_real275")
    parser.add_argument("--rgb-root", default="/mnt/dest/OmniNOCS/real_test")
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--stage1-steps", type=int, default=1)
    parser.add_argument("--stage2-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/canonical_bbox_check.json")
    args = parser.parse_args()

    from sam3d_objects.data.dataset.metric import OmniNOCSReal275Dataset

    print(f"Loading dataset from {args.annotations_root} ...")
    dataset = OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split="train",
        min_mask_pixels=500,
    )
    print(f"Dataset size: {len(dataset)}")

    # Sample deterministically, spread across categories
    rng = np.random.default_rng(args.seed)
    by_category = defaultdict(list)
    for i, rec in enumerate(dataset.records):
        by_category[rec.get("category", "unknown")].append(i)

    sampled_indices = []
    categories = sorted(by_category)
    per_cat = max(1, args.n_samples // len(categories))
    for cat in categories:
        idxs = by_category[cat]
        chosen = rng.choice(idxs, size=min(per_cat, len(idxs)), replace=False)
        sampled_indices.extend(chosen.tolist())
    # top up to n_samples
    remaining = [i for i in range(len(dataset)) if i not in set(sampled_indices)]
    rng.shuffle(remaining)
    sampled_indices = sampled_indices[:args.n_samples]
    if len(sampled_indices) < args.n_samples:
        sampled_indices += remaining[: args.n_samples - len(sampled_indices)]

    print(f"Sampled {len(sampled_indices)} examples across {len(categories)} categories")

    print(f"Loading pipeline from {args.config} ...")
    pipeline = load_pipeline(args.config, args.device)

    results = []
    errors = 0
    for rank, idx in enumerate(sampled_indices):
        item = dataset[idx]
        cat = item.get("category", "?")
        img_name = item.get("image_name", "?")
        print(f"[{rank+1}/{len(sampled_indices)}] {cat} | {img_name}", end=" ... ", flush=True)
        try:
            r = run_one(pipeline, item, args.stage1_steps, args.stage2_steps, args.device)
            results.append(r)
            print(
                f"bbox=[{r['bbox_x']:.3f}, {r['bbox_y']:.3f}, {r['bbox_z']:.3f}]  "
                f"real=[{r['W_real']:.3f}, {r['H_real']:.3f}, {r['D_real']:.3f}]m"
            )
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")

    if not results:
        print("No results — all examples failed.")
        return

    # ── Summary statistics ─────────────────────────────────────────────────
    bx = np.array([r["bbox_x"] for r in results])
    by = np.array([r["bbox_y"] for r in results])
    bz = np.array([r["bbox_z"] for r in results])
    cx = np.array([r["center_x"] for r in results])
    cy = np.array([r["center_y"] for r in results])
    cz = np.array([r["center_z"] for r in results])
    Ws = np.array([r["W_real"] for r in results])
    Hs = np.array([r["H_real"] for r in results])
    Ds = np.array([r["D_real"] for r in results])

    bbox_vol  = bx * by * bz
    real_vol  = Ws * Hs * Ds
    bbox_diag = np.cbrt(bbox_vol)
    real_diag = np.cbrt(real_vol)

    # Ratio: does bbox ≈ 1.0?
    rx = bx / np.clip(Ws, 1e-6, None)  # canonical / real  (should be uniform if WHD = scale)
    ry = by / np.clip(Hs, 1e-6, None)
    rz = bz / np.clip(Ds, 1e-6, None)

    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n" + "=" * 60)
    print("CANONICAL MESH BOUNDING BOX DIAGNOSTIC")
    print("=" * 60)
    print(f"\nN = {len(results)}  (errors={errors})")

    print("\n── Canonical bbox (in [-0.5, 0.5]³ space) ──")
    for axis, arr in [("X (width)", bx), ("Y (height)", by), ("Z (depth)", bz)]:
        print(f"  {axis}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")

    print("\n── Canonical centroid (should be near 0) ──")
    for axis, arr in [("X", cx), ("Y", cy), ("Z", cz)]:
        print(f"  {axis}: mean={arr.mean():.3f}  std={arr.std():.3f}")

    print("\n── Does bbox fill the unit cube? (bbox per axis, ideal ≈ 1.0) ──")
    for axis, arr in [("X", bx), ("Y", by), ("Z", bz)]:
        pct_near_1 = np.mean(np.abs(arr - 1.0) < 0.2) * 100
        print(f"  {axis}: mean={arr.mean():.3f}  % within ±0.2 of 1.0: {pct_near_1:.0f}%")

    print("\n── Correlation: canonical bbox axis ↔ real WHD ──")
    print(f"  corr(bbox_x, W_real) = {corr(bx, Ws):.3f}")
    print(f"  corr(bbox_y, H_real) = {corr(by, Hs):.3f}")
    print(f"  corr(bbox_z, D_real) = {corr(bz, Ds):.3f}")

    print("\n── Scale ratio canonical/real (if ≈ const → WHD = direct scale) ──")
    for axis, arr in [("X: bbox/W", rx), ("Y: bbox/H", ry), ("Z: bbox/D", rz)]:
        print(f"  {axis}: mean={arr.mean():.3f}  std={arr.std():.3f}  CV={arr.std()/max(arr.mean(),1e-6):.3f}")

    print("\n── Volume summary ──")
    print(f"  canonical vol: mean={bbox_vol.mean():.3f}  std={bbox_vol.std():.3f}")
    print(f"  real vol (m³): mean={real_vol.mean():.4f}  std={real_vol.std():.4f}")
    print(f"  corr(cbrt(bbox_vol), cbrt(real_vol)) = {corr(bbox_diag, real_diag):.3f}")

    print("\n── Per-category bbox means ──")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append([r["bbox_x"], r["bbox_y"], r["bbox_z"]])
    for cat in sorted(by_cat):
        arr = np.array(by_cat[cat])
        print(f"  {cat:12s}: x={arr[:,0].mean():.3f}  y={arr[:,1].mean():.3f}  z={arr[:,2].mean():.3f}  n={len(arr)}")

    print("\n── Interpretation ──")
    overall_mean = np.mean([bx.mean(), by.mean(), bz.mean()])
    overall_cv   = np.mean([bx.std()/max(bx.mean(),1e-6),
                            by.std()/max(by.mean(),1e-6),
                            bz.std()/max(bz.mean(),1e-6)])
    if overall_mean > 0.75 and overall_cv < 0.25:
        print("  ✓ Canonical mesh approximately fills the unit cube (mean bbox ≈ 1, low CV).")
        print("    Predicting [W, H, D] IS a direct per-axis scale for the mesh.")
        print("    No bounding-box lookup needed at inference time.")
    elif corr(bx, Ws) > 0.5 or corr(by, Hs) > 0.5 or corr(bz, Ds) > 0.5:
        print("  ~ Canonical bbox is correlated with real dims but mean ≠ 1.")
        print("    WHD prediction needs a fixed scaling factor per axis to become a direct scale.")
        print("    The constant ratio (canonical/real) can be read off and baked into inference.")
    else:
        print("  ✗ Canonical bbox is neither ≈ 1 nor strongly correlated with real dims.")
        print("    A post-hoc bbox division is needed at inference time.")
        print("    Consider training directly on s* = W_real / bbox_x (requires mesh decoding in cache step).")

    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({"results": results, "n_errors": errors}, f, indent=2)
    print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
