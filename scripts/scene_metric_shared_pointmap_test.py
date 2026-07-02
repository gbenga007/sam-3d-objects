#!/usr/bin/env python
"""
Near-term test (2026-06-16): full-scene metric reconstruction via a SHARED global MoGe-2 pointmap.

Hypothesis ([[scene-scale-and-full-scene-reconstruction]]): scene-level metric scale is NOT a
paradigm shift. Multi-object composition already exists (notebook/demo_multi_object.ipynb:
per-mask inference + make_scene). The only change needed is to feed EVERY object the SAME global
MoGe-2 METRIC pointmap instead of letting each object self-anchor to MoGe-v1. Because every
object's crop is then a crop of the same metric pointmap, all per-object sizes AND positions are
already in one shared metric world frame -> objects should co-register at metric scale with NO
model change.

Decisive checks:
  (1) METRIC SIZE  : canonical mesh extent * out["scale"] -> plausible real-world dimensions (m).
  (2) CO-REGISTRATION: pose-decoder out["translation"] vs the object's centroid in the SHARED
      MoGe-2 pointmap (masked-region median 3D point). If they agree across objects, the objects
      live in one consistent metric frame -> scene reconstruction is composition, not a new model.

Loads the pipeline directly (NOT notebook/inference.py, which drags seaborn/gradio), following
scripts/gt_pointmap_metric_oracle.py.

Run (concurrent with training is fine; ~32GB inference + ~16GB train < 80GB A100):
  CONDA_PREFIX=/opt/conda/envs/sam3d /opt/conda/envs/sam3d/bin/python \
      scripts/scene_metric_shared_pointmap_test.py \
      --image-dir notebook/images/137444513_Livingroom-graphic81 --num-masks 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/source/MoGe")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from eval_bbox_from_depth import MoGe2Estimator  # noqa: E402
import sam3d_objects  # noqa: E402,F401  (registers hydra targets)
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa: E402,F401


# ---- minimal loaders (copies of notebook.inference helpers, no viz deps) ----
def load_image(path):
    return np.array(Image.open(path)).astype(np.uint8)


def load_masks(folder_path, extension=".png"):
    masks, idx = [], 0
    while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
        m = load_image(os.path.join(folder_path, f"{idx}{extension}"))
        if m.ndim == 3:
            m = m[..., -1]
        masks.append(m > 0)
        idx += 1
    return masks


def load_pipeline(config_path):
    """Minimal equivalent of notebook.inference.Inference.__init__ (oracle pattern)."""
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = os.path.dirname(config_path)
    return instantiate(config)


def opencv_to_pytorch3d(pts: np.ndarray) -> np.ndarray:
    """[H,W,3] OpenCV (x-right,y-down,z-forward) -> PyTorch3D ([-X,-Y,Z]). NaN preserved."""
    out = pts.copy()
    out[..., 0] = -out[..., 0]
    out[..., 1] = -out[..., 1]
    return out


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def mesh_vertices(out):
    mesh = out.get("mesh", None)
    if mesh is None:
        return None
    if isinstance(mesh, (list, tuple)):
        mesh = mesh[0]
    for attr in ("vertices", "verts"):
        if hasattr(mesh, attr):
            return _np(getattr(mesh, attr)).astype(np.float64)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="notebook/images/137444513_Livingroom-graphic81")
    ap.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--num-masks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts/scene_metric_test")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = REPO / args.image_dir

    image = load_image(str(img_dir / "image.png"))
    rgb = image[..., :3].astype(np.uint8)
    masks = load_masks(str(img_dir))
    if args.num_masks > 0:
        masks = masks[: args.num_masks]
    print(f"[scene-test] image {rgb.shape}, {len(masks)} masks from {img_dir.name}", flush=True)

    # --- 1) ONE global MoGe-2 metric pointmap for the whole frame ---
    print("[scene-test] loading MoGe-2 (Ruicheng/moge-2-vitl) ...", flush=True)
    est = MoGe2Estimator(device="cuda")
    pts_cv = est.infer_points(rgb, np.eye(3))               # [H,W,3] OpenCV metric, NaN-masked
    pts_p3d = opencv_to_pytorch3d(pts_cv).astype(np.float32)
    global_pointmap = torch.from_numpy(pts_p3d)             # [H,W,3] PyTorch3D metric
    H, W = pts_cv.shape[:2]
    finite = np.isfinite(pts_cv[..., 2])
    print(f"[scene-test] global pointmap {tuple(global_pointmap.shape)}; valid z "
          f"[{np.nanmin(pts_cv[finite,2]):.3f},{np.nanmax(pts_cv[finite,2]):.3f}] m", flush=True)
    del est
    torch.cuda.empty_cache()

    # --- 2) inference per mask with the SHARED global pointmap ---
    print("[scene-test] loading SAM3D pipeline ...", flush=True)
    pipeline = load_pipeline(str(REPO / args.config))

    records = []
    for i, mask in enumerate(masks):
        print(f"[scene-test] --- object {i}/{len(masks)} ---", flush=True)
        mask_u8 = (mask.astype(np.uint8) * 255)[..., None]
        rgba = np.concatenate([rgb, mask_u8], axis=-1)
        out = pipeline.run(
            rgba, None, args.seed,
            stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False,
            with_layout_postprocess=False, use_vertex_color=True,
            stage1_inference_steps=None, pointmap=global_pointmap,
            decode_formats=["mesh"],
        )
        if i == 0:
            print(f"[scene-test] output keys: {sorted(out.keys())}", flush=True)

        info = {"object": i}
        scale = _np(out.get("scale")).reshape(-1) if out.get("scale") is not None else None
        if scale is not None:
            info["scale"] = scale.tolist()
        v = mesh_vertices(out)
        if v is not None and scale is not None:
            canon_ext = v.max(0) - v.min(0)
            s = scale if scale.size == 3 else np.repeat(scale[:1], 3)
            info["metric_whd"] = (canon_ext * s).tolist()
            info["metric_iso_max"] = float(np.max(canon_ext * s))

        # (2) co-registration: pose-decoder translation vs MoGe-2 masked-region centroid
        if out.get("translation") is not None:
            info["pred_translation"] = _np(out["translation"]).reshape(-1).tolist()
        m_small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        valid = m_small & finite
        if valid.sum() > 0:
            depth_centroid = np.nanmedian(pts_p3d[valid], axis=0)  # PyTorch3D metric
            info["depth_centroid"] = depth_centroid.tolist()
            info["mask_pixels"] = int(valid.sum())
            if "pred_translation" in info:
                d = np.asarray(info["pred_translation"]) - depth_centroid
                info["coreg_dist_m"] = float(np.linalg.norm(d))

        records.append(info)
        print(f"[scene-test] obj {i}: whd(m)={[round(x,3) for x in info.get('metric_whd',[])] or None} "
              f"pred_t={[round(x,3) for x in info.get('pred_translation',[])] or None} "
              f"depth_c={[round(x,3) for x in info.get('depth_centroid',[])] or None} "
              f"coreg={round(info['coreg_dist_m'],3) if 'coreg_dist_m' in info else None} m", flush=True)

    summary = {
        "image_dir": str(img_dir), "num_masks": len(masks),
        "shared_global_moge2_pointmap": True, "objects": records,
    }
    (out_dir / "scene_metric_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[scene-test] wrote {out_dir/'scene_metric_summary.json'}", flush=True)
    print("[scene-test] DONE", flush=True)


if __name__ == "__main__":
    main()
