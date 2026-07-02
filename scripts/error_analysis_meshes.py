#!/usr/bin/env python
"""
Failure-mode analysis for a trained metric model (2026-06-17): run on held-out instances, rank by
W,H,D error, and for the worst-K (and best-K) save the input RGB crop + the predicted METRIC mesh
(.ply) + a GT-vs-pred dims report — so you can SEE what's being reconstructed wrong.

Use (after the mAP run frees the GPU):
  CONDA_PREFIX=/opt/conda/envs/sam3d /opt/conda/envs/sam3d/bin/python scripts/error_analysis_meshes.py \
    --checkpoint artifacts/metric_scale/checkpoints/joint_mot_v1_best.pt \
    --n-per-source 40 --worst-k 20
"""
import argparse, json, os, sys
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
sys.path.insert(0, "/mnt/source/MoGe"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch, trimesh
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate
import sam3d_objects  # noqa
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset
from sam3d_objects.training.finetune_metric_scale import Moge2PointmapStore

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"


def load_pipeline(cfg_path, ckpt):
    cfg = OmegaConf.load(cfg_path); cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False
    cfg.workspace_dir = os.path.dirname(cfg_path); cfg.metric_scale_checkpoint_path = ckpt
    return instantiate(cfg)


def mesh_from_out(out):
    m = out.get("mesh"); m = m[0] if isinstance(m, (list, tuple)) else m
    v = m.vertices.detach().float().cpu().numpy() if hasattr(m.vertices, "detach") else np.asarray(m.vertices)
    f = m.faces.detach().cpu().numpy() if hasattr(getattr(m, "faces", None), "detach") else np.asarray(getattr(m, "faces", []))
    return v, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pointmap-dir", default="artifacts/metric_scale/moge2_pointmaps")
    ap.add_argument("--n-per-source", type=int, default=40)
    ap.add_argument("--worst-k", type=int, default=20)
    ap.add_argument("--out-dir", default="artifacts/error_analysis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join("/mnt/source/sam-3d-objects", args.out_dir)
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)

    pipe = load_pipeline(os.path.join("/mnt/source/sam-3d-objects", args.config),
                         os.path.join("/mnt/source/sam-3d-objects", args.checkpoint))
    store = Moge2PointmapStore(os.path.join("/mnt/source/sam-3d-objects", args.pointmap_dir))
    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI, sources=["nocs_real275", "objectron", "arkitscenes"],
        rgb_roots={"nocs_real275": f"{OMNI}/real_test", "objectron": OMNI, "arkitscenes": OMNI},
        split="train", max_records_per_source=args.n_per_source, skip_missing_rgb=True)
    print(f"[err-analysis] {len(ds)} instances", flush=True)

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        try:
            pm = store.lookup(item["image_name"])
        except KeyError:
            continue
        try:
            out = pipe.run(item["image"], None, 42, stage1_only=False, with_mesh_postprocess=False,
                           with_texture_baking=False, with_layout_postprocess=False,
                           use_vertex_color=True, stage1_inference_steps=None,
                           pointmap=pm, decode_formats=["mesh"])
        except Exception as e:
            print(f"[err-analysis] {item['uid']} FAILED: {type(e).__name__}", flush=True); continue
        scale = out["scale"][0].float().cpu().numpy().reshape(-1)
        v, f = mesh_from_out(out)
        canon_ext = v.max(0) - v.min(0)
        pred_dims = np.sort(canon_ext * (scale if scale.size == 3 else np.repeat(scale[:1], 3)))[::-1]
        gt_dims = np.sort(np.asarray(item["metric_dims"], dtype=np.float64))[::-1]
        err = float(np.mean(np.abs(pred_dims - gt_dims) / np.clip(gt_dims, 1e-6, None)) * 100)
        rows.append({"uid": item["uid"], "category": item["category"], "source": item["source"],
                     "gt_dims": gt_dims.round(3).tolist(), "pred_dims": pred_dims.round(3).tolist(),
                     "err_pct": round(err, 1), "_v": v * scale.mean(), "_f": f, "_img": item["image"]})
        if (i + 1) % 20 == 0:
            print(f"[err-analysis] {i+1}/{len(ds)} done", flush=True)

    rows.sort(key=lambda r: -r["err_pct"])
    report = ["# Error analysis — joint_mot_v1 (worst-K + best-K by W,H,D MAPE)\n",
              "| rank | err% | category | source | GT W,H,D (m) | pred W,H,D (m) | mesh |", "|---|---|---|---|---|---|---|"]
    def dump(r, tag, rank):
        base = f"{tag}_{rank:02d}_{r['category'].replace(' ','_')}_{r['err_pct']:.0f}pct"
        try:
            trimesh.Trimesh(vertices=r["_v"], faces=r["_f"]).export(os.path.join(out_dir, "meshes", base + ".ply"))
        except Exception:
            trimesh.PointCloud(r["_v"]).export(os.path.join(out_dir, "meshes", base + ".ply"))
        Image.fromarray(r["_img"][..., :3].astype(np.uint8)).save(os.path.join(out_dir, "crops", base + ".png"))
        report.append(f"| {tag}{rank} | {r['err_pct']} | {r['category']} | {r['source']} | "
                      f"{r['gt_dims']} | {r['pred_dims']} | meshes/{base}.ply |")
    for k, r in enumerate(rows[:args.worst_k]): dump(r, "WORST", k + 1)
    report.append("\n## Best (for contrast)\n| rank | err% | category | source | GT | pred | mesh |\n|---|---|---|---|---|---|---|")
    for k, r in enumerate(rows[-10:][::-1]): dump(r, "BEST", k + 1)
    open(os.path.join(out_dir, "ERROR_ANALYSIS.md"), "w").write("\n".join(report))
    json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
              open(os.path.join(out_dir, "all_errors.json"), "w"), indent=2)
    print(f"[err-analysis] DONE — {len(rows)} instances. Report: {out_dir}/ERROR_ANALYSIS.md "
          f"| meshes/ (.ply) + crops/ (.png)", flush=True)


if __name__ == "__main__":
    main()
