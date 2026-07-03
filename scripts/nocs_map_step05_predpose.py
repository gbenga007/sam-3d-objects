#!/usr/bin/env python
"""
Step 0.5 (2026-06-16): PREDICTED-pose mAP on NOCS-Real275 (official eval).

Step 0 showed size given GT pose -> ~100 mAP@50 (size is NOT the bottleneck). Step 0.5 measures
where the FROZEN pose decoder's rotation+translation actually lands = the real ceiling. Runs the
SAM3D pipeline per GT instance (GT mask + GT-depth metric pointmap, postprocess OFF -> raw pose),
builds the predicted oriented box, and evaluates with the official symmetry-aware NOCS mAP.

Frames are in the SAME camera frame as gt_RT (both [-X,-Y,Z] = z180.OpenCV; the GT-depth pointmap
is built [-X,-Y,Z]) -> NO frame conversion; validated by a translation sanity (|pred_c - gt_c|).

"Localization given GT 2D instances" protocol (GT masks -> perfect detection/recall, class known,
1:1 matched). State this in the paper; it's the same favourable setting as the size oracle.

Eval variants (all official compute_degree_cm_mAP, symmetry-aware):
  GT-POSE sanity : pred=GT                  -> must be ~100 (re-validates harness in-script)
  FULL PRED      : pred R,t,size from model -> the headline (= pose-limited, since size~free)

Reuses scripts/nocs_map_step0_oracle.py (GT parse/align/eval) + gt_pointmap_metric_oracle helpers.
Predictions cached -> resumable; rerun with --eval-only to re-score without the GPU.

Run (concurrent with training OK):
  CONDA_PREFIX=/opt/conda/envs/sam3d /opt/conda/envs/sam3d/bin/python \
      scripts/nocs_map_step05_predpose.py --max-frames 12
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("LIDRA_SKIP_INIT", "true")  # lightweight tool: skip training-only init (matches the oracle scripts; inference/pose decoder unaffected)
REPO = "/mnt/source/sam-3d-objects"
# /mnt/source/MoGe has BOTH moge.model.v1 (pipeline depth model) and v2 (MoGe2Estimator). Must be
# on the path BEFORE anything imports `moge`, else the env's v1-only moge gets cached and v2 fails.
sys.path.insert(0, "/mnt/source/MoGe")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

# Stub tensorflow/ICP WITH a valid __spec__ before anything imports them. torch._dynamo's
# trace_rules calls importlib.find_spec("tensorflow") at import time, which raises on a
# spec-less ModuleType. Pre-populate so step0's setdefault stubs don't override these.
import importlib.machinery  # noqa: E402
import types as _types  # noqa: E402
for _name in ("tensorflow", "ICP"):
    if _name not in sys.modules:
        _m = _types.ModuleType(_name)
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules[_name] = _m

import nocs_map_step0_oracle as step0  # gives GT parse/align/eval  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from pytorch3d.transforms import quaternion_to_matrix  # noqa: E402
import sam3d_objects  # noqa: E402,F401
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa: E402,F401

nocs_utils = step0.nocs_utils
INTRINSICS = step0.INTRINSICS
SYNSET_NAMES = step0.SYNSET_NAMES
REAL_TEST = step0.REAL_TEST

# The pipeline outputs pose in the pointmap frame ([X,Y,Z]); align()'s gt_RT is z180.OpenCV
# = [-X,-Y,Z]. Empirically (step 0.5 v1) pred Z matched gt to ~cm but X,Y were sign-flipped ->
# bring predicted RT into the gt_RT frame with this diag(-1,-1,1) premultiply.
Z180 = np.diag([-1.0, -1.0, 1.0, 1.0])


def load_pipeline(config_path, metric_checkpoint=None):
    cfg = OmegaConf.load(config_path)
    cfg.rendering_engine = "pytorch3d"
    cfg.compile_model = False
    cfg.workspace_dir = os.path.dirname(config_path)
    if metric_checkpoint:
        cfg.metric_scale_checkpoint_path = metric_checkpoint  # loads head+decoder
    pipeline = instantiate(cfg)
    if metric_checkpoint:
        _load_slat_cross_attn(pipeline, metric_checkpoint)     # the 100.8M trained SLAT weights
    return pipeline


def _load_slat_cross_attn(pipeline, ckpt_path):
    """Faithfully apply the trained SLAT cross_attn+norm2 (fp32-upcast, as in training).
    Without this, metric_dimensions use stock SLAT feats != the trained head's regime."""
    import torch as _t
    from sam3d_objects.training.finetune_metric_scale import collect_slat_cross_attn_params
    ck = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    if "slat_cross_attn" not in ck:
        print("[step0.5] checkpoint has no slat_cross_attn (head+decoder only)", flush=True)
        return
    _, backbone = collect_slat_cross_attn_params(pipeline, upcast_fp32=True)  # upcast + unfreeze
    sd = ck["slat_cross_attn"]
    for i, block in enumerate(backbone.blocks):
        ca = {k[len(f"blocks.{i}.cross_attn."):]: v for k, v in sd.items()
              if k.startswith(f"blocks.{i}.cross_attn.")}
        n2 = {k[len(f"blocks.{i}.norm2."):]: v for k, v in sd.items()
              if k.startswith(f"blocks.{i}.norm2.")}
        block.cross_attn.load_state_dict(ca, strict=True)
        block.norm2.load_state_dict(n2, strict=True)
        block.cross_attn.eval(); block.norm2.eval()
    print(f"[step0.5] loaded trained slat_cross_attn into {len(backbone.blocks)} blocks", flush=True)


def build_metric_pointmap(depth_m, K):
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_m
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    return torch.from_numpy(np.stack([-X, -Y, Z], axis=-1).astype(np.float32))


def _opencv_to_pytorch3d(pts):
    """[H,W,3] OpenCV (x-right,y-down,z-forward) -> PyTorch3D [-X,-Y,Z]. NaN preserved."""
    out = pts.copy()
    out[..., 0] = -out[..., 0]
    out[..., 1] = -out[..., 1]
    return out


def moge2_pointmap(est, rgb):
    """MoGe-2 metric pointmap (self-intrinsics) -> [H,W,3] PyTorch3D metric tensor."""
    pts_cv = est.infer_points(rgb.astype(np.uint8), np.eye(3))      # OpenCV metric, NaN-masked
    return torch.from_numpy(_opencv_to_pytorch3d(pts_cv).astype(np.float32))


def depth_metres(frame_path):
    d16 = step0.load_depth(frame_path)          # uint16 mm
    if d16 is None:
        return None
    z = d16.astype(np.float32) / 1000.0
    z[z <= 0] = np.nan
    return z


def mesh_verts(out):
    m = out.get("mesh", None)
    if isinstance(m, (list, tuple)):
        m = m[0]
    for a in ("vertices", "verts"):
        if hasattr(m, a):
            v = getattr(m, a)
            return v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    return None


def predict_box(pipeline, rgb, mask, pointmap, seed, use_head_dims=True,
                layout_postprocess=False):
    """Run pipeline -> predicted oriented box in gt_RT frame: (pred_RT 4x4, pred_scales 3).
    use_head_dims=False: ignore the (possibly random/untrained) MetricScaleHead and use the
    pose-decoder size (mesh_ext x out['scale']) — required for joint-MoT ckpts where the head
    is unused/random and W,H,D comes from the trained pose."""
    mask_u8 = (mask.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate([rgb[..., :3], mask_u8], axis=-1)
    out = pipeline.run(
        rgba, None, seed, stage1_only=False,
        # layout post-optimization (ICP against the pointmap) needs a glb, which
        # requires the mesh postprocess path; texture baking stays off for speed.
        with_mesh_postprocess=layout_postprocess,
        with_texture_baking=False,
        with_layout_postprocess=layout_postprocess, use_vertex_color=True,
        stage1_inference_steps=None, pointmap=pointmap, decode_formats=["mesh"],
    )
    R = quaternion_to_matrix(out["rotation"].float())[0].cpu().numpy()      # [3,3]
    t = out["translation"].float().cpu().numpy().reshape(3)
    scale = out["scale"][0].float().cpu().numpy().reshape(-1)
    scale = scale if scale.size == 3 else np.repeat(scale[:1], 3)
    v = mesh_verts(out)
    ext = (v.max(0) - v.min(0))                      # canonical extents
    ctr = (v.max(0) + v.min(0)) / 2.0                # canonical bbox centre (mesh may be off-origin)
    if use_head_dims and out.get("metric_dimensions") is not None:  # trained head's metric [W,H,D]
        metric_ext = out["metric_dimensions"][0].float().cpu().numpy().reshape(-1)
    else:
        metric_ext = ext * scale                     # pose-decoder size (canon extents x scale)
    center_cam = R @ (scale * ctr) + t               # box centre in camera (gt_RT) frame (pose-decoder t)
    pred_RT = np.eye(4, dtype=np.float64)
    pred_RT[:3, :3] = R
    pred_RT[:3, 3] = center_cam
    return pred_RT, metric_ext.astype(np.float64), center_cam


def cache_key(fp, i):
    return f"{os.path.relpath(fp, REAL_TEST)}#{i}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts/nocs_map_step05")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-head-dims", dest="use_head_dims", action="store_false", default=True,
                    help="Ignore the MetricScaleHead; size from the pose decoder (for joint-MoT ckpts).")
    ap.add_argument("--metric-checkpoint", default=None,
                    help="trained metric-scale ckpt -> size from out['metric_dimensions'] (the user's method)")
    ap.add_argument("--pointmap", choices=["gt", "none", "moge2"], default="gt",
                    help="gt = GT-depth metric pointmap; none = MoGe-v1 internal; moge2 = live MoGe-2 (deploy)")
    ap.add_argument("--layout-postprocess", action="store_true",
                    help="enable the pipeline's layout post-optimization (ICP against the "
                         "pointmap) before reading the box — the untested mAP@50 lever")
    args = ap.parse_args()

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "pred_boxes.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[r["key"]] = r
        print(f"[step0.5] loaded {len(cache)} cached predictions", flush=True)

    frames = sorted(glob.glob(os.path.join(REAL_TEST, "*", "*_color.png")))
    frame_paths = [f[: -len("_color.png")] for f in frames]
    rng = np.random.RandomState(args.seed)
    if args.max_frames and args.max_frames < len(frame_paths):
        idx = sorted(rng.choice(len(frame_paths), size=args.max_frames, replace=False).tolist())
        frame_paths = [frame_paths[i] for i in idx]

    pipeline = None
    if not args.eval_only:
        ckpt = (args.metric_checkpoint and os.path.join(REPO, args.metric_checkpoint)
                if args.metric_checkpoint and not os.path.isabs(args.metric_checkpoint)
                else args.metric_checkpoint)
        print(f"[step0.5] loading SAM3D pipeline (ckpt={args.metric_checkpoint}, "
              f"pointmap={args.pointmap}) ...", flush=True)
        pipeline = load_pipeline(os.path.join(REPO, args.config), metric_checkpoint=ckpt)

    moge2_est = None
    if not args.eval_only and args.pointmap == "moge2":
        sys.path.insert(0, "/mnt/source/MoGe")
        from eval_bbox_from_depth import MoGe2Estimator  # noqa: E402
        print("[step0.5] loading MoGe-2 (Ruicheng/moge-2-vitl) ...", flush=True)
        moge2_est = MoGe2Estimator(device="cuda")

    final_results, trans_err = [], []
    cf = open(cache_path, "a")
    for fi, fp in enumerate(frame_paths):
        parsed = step0.load_gt_instances(fp)
        if parsed is None:
            continue
        gt_mask, gt_coord, gt_class_ids = parsed
        depth16 = step0.load_depth(fp)
        zm = depth_metres(fp)
        if depth16 is None or zm is None:
            continue
        gt_RTs, gt_scales, _, _ = nocs_utils.align(
            gt_class_ids, gt_mask, gt_coord, depth16, INTRINSICS, SYNSET_NAMES, fp, None)
        gt_bbox = nocs_utils.extract_bboxes(gt_mask)
        rgb = step0.cv2.imread(fp + "_color.png")[:, :, ::-1].copy()  # BGR->RGB
        if pipeline is None:
            pointmap = None
        elif args.pointmap == "gt":
            pointmap = build_metric_pointmap(zm, INTRINSICS)          # GT-depth metric
        elif args.pointmap == "moge2":
            pointmap = moge2_pointmap(moge2_est, rgb)                 # live MoGe-2 metric (deploy)
        else:
            pointmap = None                                          # MoGe-v1 internal ("as usual")

        n = len(gt_class_ids)
        pred_RTs = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
        pred_scales = np.ones((n, 3), dtype=np.float64)
        ok = True
        for i in range(n):
            key = cache_key(fp, i)
            if key in cache:
                rec = cache[key]
            elif pipeline is not None:
                try:
                    RT, sc, cc = predict_box(pipeline, rgb, gt_mask[:, :, i], pointmap, args.seed,
                                             use_head_dims=args.use_head_dims,
                                             layout_postprocess=args.layout_postprocess)
                except Exception as e:
                    print(f"[step0.5] {key} FAILED: {type(e).__name__}: {e}", flush=True)
                    ok = False
                    break
                gt_c = gt_RTs[i][:3, 3].tolist()
                rec = {"key": key, "pred_RT": RT.tolist(), "pred_scales": sc.tolist(),
                       "pred_center": cc.tolist(), "gt_center": gt_c,
                       "trans_err": float(np.linalg.norm(np.array(cc) - np.array(gt_c)))}
                cache[key] = rec
                cf.write(json.dumps(rec) + "\n"); cf.flush()
            else:
                ok = False
                break
            pred_RTs[i] = Z180 @ np.array(rec["pred_RT"])     # into gt_RT frame (diag(-1,-1,1))
            pred_scales[i] = np.array(rec["pred_scales"])
            corrected_center = pred_RTs[i][:3, 3]
            trans_err.append(float(np.linalg.norm(corrected_center - np.array(rec["gt_center"]))))
        if not ok:
            continue
        final_results.append({
            "gt_class_ids": np.asarray(gt_class_ids), "gt_RTs": np.asarray(gt_RTs),
            "gt_scales": np.asarray(gt_scales), "gt_handle_visibility": np.ones_like(gt_class_ids),
            "gt_bboxes": np.asarray(gt_bbox),
            "pred_class_ids": np.asarray(gt_class_ids), "pred_RTs": pred_RTs,
            "pred_scales": pred_scales, "pred_scores": np.ones(n, dtype=np.float32),
            "pred_bboxes": np.asarray(gt_bbox),
        })
        print(f"[step0.5] frame {fi+1}/{len(frame_paths)}: {n} inst "
              f"(cum {len(final_results)} frames, {len(trans_err)} inst)", flush=True)
    cf.close()

    if not final_results:
        print("[step0.5] no results assembled", flush=True)
        return

    print(f"\n[step0.5] === {len(final_results)} frames, {len(trans_err)} instances ===")
    print(f"[step0.5] translation sanity |pred_c - gt_c| (m): median "
          f"{np.median(trans_err):.3f}  mean {np.mean(trans_err):.3f}  "
          f"p90 {np.percentile(trans_err,90):.3f}", flush=True)
    print("[step0.5] reference: NOCSformer 43.5/10.6 ; CubeRCNN 14.9/4.1 ; sup. NOCS 79.6/72.4\n",
          flush=True)

    # GT-pose sanity (pred=GT) re-validates the harness inside this script
    sanity = [dict(r, pred_RTs=r["gt_RTs"].copy(), pred_scales=r["gt_scales"].copy(),
                   pred_class_ids=r["gt_class_ids"].copy()) for r in final_results]
    step0.run_map(sanity, out_dir, "GT-POSE sanity")
    step0.run_map(final_results, out_dir, "FULL PRED pose")
    print("\n[step0.5] DONE", flush=True)


if __name__ == "__main__":
    main()
