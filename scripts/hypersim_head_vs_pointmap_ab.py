#!/usr/bin/env python
"""CLEAN A/B (2026-06-18): head vs pointmap on Hypersim VAL with **GT depth pointmaps** — removes the
MoGe-2 pointmap-quality confound that made the OmniNOCS A/B provisional. Same Variant-2 inference call:

  A (head)     = out['metric_dimensions']                 -> trained metric-scale head
  B (pointmap) = mesh.bbox * out['scale'].mean()          -> frozen pose-decoder / GT-pointmap scale

GT pointmap = Hypersim depth_meters.hdf5 (Euclidean) -> planar -> camera XYZ -> pytorch3d [-X,-Y,Z],
built full-frame at 768x1024 to align with item['image']. Exact metric supervision, exact depth.
"""
import argparse, json, os, sys
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
sys.path.insert(0, "/mnt/source/MoGe"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch, h5py
from collections import defaultdict
from omegaconf import OmegaConf
from hydra.utils import instantiate
import sam3d_objects  # noqa
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset
from sam3d_objects.training.finetune_metric_scale import collect_slat_cross_attn_params

REPO = "/mnt/source/sam-3d-objects"
OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
DEPTH = f"{OMNI}/hypersim_depth"
META = f"{OMNI}/omninocs_release_hypersim/hypersim_val_metadata.json"
CKPT = f"{REPO}/artifacts/metric_scale/checkpoints/mixed_moge2_live_v1_best.pt"


def load_pipe():
    cfg = OmegaConf.load(f"{REPO}/checkpoints/hf/pipeline.yaml")
    cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False
    cfg.workspace_dir = f"{REPO}/checkpoints/hf"; cfg.metric_scale_checkpoint_path = CKPT
    pipe = instantiate(cfg)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if "slat_cross_attn" in ck:
        _, bb = collect_slat_cross_attn_params(pipe, upcast_fp32=True)
        sd = ck["slat_cross_attn"]
        for i, blk in enumerate(bb.blocks):
            ca = {k.split(f"blocks.{i}.cross_attn.", 1)[1]: v for k, v in sd.items() if k.startswith(f"blocks.{i}.cross_attn.")}
            n2 = {k.split(f"blocks.{i}.norm2.", 1)[1]: v for k, v in sd.items() if k.startswith(f"blocks.{i}.norm2.")}
            if ca: blk.cross_attn.load_state_dict(ca, strict=True)
            if n2: blk.norm2.load_state_dict(n2, strict=True)
            blk.cross_attn.eval(); blk.norm2.eval()
        print(f"[hs-ab] loaded Variant 2 SLAT cross-attn into {len(bb.blocks)} blocks", flush=True)
    return pipe


def gt_pointmap(image_name, intr):
    scene, cam, frame = image_name.split("/")[:3]
    idx = frame.replace("frame_", "")
    dp = f"{DEPTH}/{scene}/images/scene_{cam}_geometry_hdf5/frame.{idx}.depth_meters.hdf5"
    if not os.path.exists(dp):
        return None
    d = np.array(h5py.File(dp, "r")["dataset"]).astype(np.float32)   # Euclidean dist to camera
    H, W = d.shape
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    xs = (u - cx) / fx; ys = (v - cy) / fy
    pz = d / np.sqrt(xs ** 2 + ys ** 2 + 1.0)                        # -> planar Z
    pts = np.stack([-(xs * pz), -(ys * pz), pz], -1).astype(np.float32)  # OpenCV -> pytorch3d
    pts[~np.isfinite(d)] = np.nan
    return torch.from_numpy(pts)


def mape(pred, gt):
    p = np.sort(np.asarray(pred, float))[::-1]; g = np.sort(np.asarray(gt, float))[::-1]
    return float(np.mean(np.abs(p - g) / np.clip(g, 1e-6, None)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="artifacts/metric_scale/hypersim_head_vs_pointmap_ab.json")
    args = ap.parse_args()

    intr_by_name = {r["image_name"]: r["intrinsics"] for r in json.load(open(META))}
    pipe = load_pipe()
    ds = OmniNOCSObjectDataset(omninocs_root=OMNI, sources=["hypersim"],
                              rgb_roots={"hypersim": f"{OMNI}/hypersim"}, split="val",
                              max_records_per_source=999999, skip_missing_rgb=True)
    N = len(ds); step = max(1, N // args.n)
    picks = list(range(0, N, step))[:args.n]
    print(f"[hs-ab] {len(picks)} sampled of {N} val instances", flush=True)

    rows = []
    for c, i in enumerate(picks):
        item = ds[i]
        pm = gt_pointmap(item["image_name"], intr_by_name.get(item["image_name"], None)) \
            if item["image_name"] in intr_by_name else None
        if pm is None:
            continue
        try:
            out = pipe.run(item["image"], None, 42, stage1_only=False, with_mesh_postprocess=False,
                           with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True,
                           stage1_inference_steps=None, pointmap=pm, decode_formats=["mesh"])
        except Exception as e:
            print(f"[hs-ab] {item['uid']} FAILED {type(e).__name__}", flush=True); continue
        md = out.get("metric_dimensions")
        if md is None:
            continue
        A = md[0].float().cpu().numpy().reshape(-1)
        m = out["mesh"]; m = m[0] if isinstance(m, (list, tuple)) else m
        v = m.vertices.detach().float().cpu().numpy()
        scale = out["scale"][0].float().cpu().numpy().reshape(-1)
        scale = scale if scale.size == 3 else np.repeat(scale[:1], 3)
        B = (v.max(0) - v.min(0)) * scale.mean()
        gt = np.asarray(item["metric_dims"], dtype=np.float64)
        rows.append({"cat": item["category"], "gtmax": float(gt.max()),
                     "A": mape(A, gt), "B": mape(B, gt)})
        if (c + 1) % 25 == 0:
            print(f"[hs-ab] {c+1}/{len(picks)} (kept {len(rows)})", flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)

    def agg(sel, name):
        rs = [r for r in rows if sel(r)]
        if not rs:
            print(f"  {name:22s} n=0"); return
        a = np.mean([r["A"] for r in rs]); b = np.mean([r["B"] for r in rs])
        bwin = np.mean([r["B"] < r["A"] for r in rs]) * 100
        print(f"  {name:22s} n={len(rs):3d} | A(head) {a:5.1f}% | B(pointmap) {b:5.1f}% | "
              f"B wins {bwin:4.0f}% | winner={'B' if b < a else 'A'}")

    print("\n===== HYPERSIM VAL — HEAD (A) vs POINTMAP (B) with GT depth pointmaps =====")
    print("-- by GT max-dim size bucket --")
    agg(lambda r: r["gtmax"] < 0.3, "small  (<0.3 m)")
    agg(lambda r: 0.3 <= r["gtmax"] < 0.6, "medium (0.3-0.6 m)")
    agg(lambda r: 0.6 <= r["gtmax"] < 1.2, "large  (0.6-1.2 m)")
    agg(lambda r: r["gtmax"] >= 1.2, "xlarge (>=1.2 m)")
    print("-- overall --"); agg(lambda r: True, "ALL")
    print(f"\n[hs-ab] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
