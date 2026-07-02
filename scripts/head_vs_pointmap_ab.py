#!/usr/bin/env python
"""A/B (2026-06-18): on the Obj+ARKit+NOCS HELDOUT, compare two size predictions from the SAME
Variant-2 inference call, by object-size bucket, to decide the size-routing hybrid:

  A (head)     = out['metric_dimensions']                       -> the trained metric-scale head
  B (pointmap) = (mesh.bbox extent) * out['scale'].mean()       -> frozen pose-decoder / MoGe-2 scale

Variant 2 (mixed_moge2_live_v1) only: its pose decoder is FROZEN, so B is the raw metric pointmap
scale (the one that landed ~=GT on the table). Heldout = last 64/200/200 per source (record-level,
after the 5x filter), matching the eval split. Uses cached MoGe-2 pointmaps.
"""
import argparse, json, os, sys
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
sys.path.insert(0, "/mnt/source/MoGe"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from collections import defaultdict
from omegaconf import OmegaConf
from hydra.utils import instantiate
import sam3d_objects  # noqa
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset
from sam3d_objects.training.finetune_metric_scale import Moge2PointmapStore, collect_slat_cross_attn_params

REPO = "/mnt/source/sam-3d-objects"
OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
CKPT = f"{REPO}/artifacts/metric_scale/checkpoints/mixed_moge2_live_v1_best.pt"
HELDOUT = {"nocs_real275": 64, "objectron": 200, "arkitscenes": 200}


def load_pipe():
    cfg = OmegaConf.load(f"{REPO}/checkpoints/hf/pipeline.yaml")
    cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False
    cfg.workspace_dir = f"{REPO}/checkpoints/hf"; cfg.metric_scale_checkpoint_path = CKPT
    pipe = instantiate(cfg)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if "slat_cross_attn" in ck:                       # Variant 2 also tuned SLAT cross-attn
        _, bb = collect_slat_cross_attn_params(pipe, upcast_fp32=True)
        sd = ck["slat_cross_attn"]
        for i, blk in enumerate(bb.blocks):
            ca = {k.split(f"blocks.{i}.cross_attn.", 1)[1]: v for k, v in sd.items() if k.startswith(f"blocks.{i}.cross_attn.")}
            n2 = {k.split(f"blocks.{i}.norm2.", 1)[1]: v for k, v in sd.items() if k.startswith(f"blocks.{i}.norm2.")}
            if ca: blk.cross_attn.load_state_dict(ca, strict=True)
            if n2: blk.norm2.load_state_dict(n2, strict=True)
            blk.cross_attn.eval(); blk.norm2.eval()
        print(f"[ab] loaded Variant 2 SLAT cross-attn into {len(bb.blocks)} blocks", flush=True)
    return pipe


def heldout_indices(ds):
    src_idx = defaultdict(list)
    for i, r in enumerate(ds.records):
        src_idx[r["source"]].append(i)
    out = {}
    for s, ids in src_idx.items():
        out[s] = ids[len(ids) - HELDOUT.get(s, 0):]
    return out


def mape(pred, gt):
    p = np.sort(np.asarray(pred, float))[::-1]; g = np.sort(np.asarray(gt, float))[::-1]
    return float(np.mean(np.abs(p - g) / np.clip(g, 1e-6, None)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=100)   # subsample (stride) of each heldout slice
    ap.add_argument("--pointmap-dir", default="artifacts/metric_scale/moge2_pointmaps")
    ap.add_argument("--out", default="artifacts/metric_scale/head_vs_pointmap_ab.json")
    args = ap.parse_args()

    pipe = load_pipe()
    store = Moge2PointmapStore(args.pointmap_dir)
    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI, sources=list(HELDOUT),
        rgb_roots={s: (f"{OMNI}/real_test" if s == "nocs_real275" else OMNI) for s in HELDOUT},
        split="train", max_records_per_source=3334, skip_missing_rgb=True)
    hi = heldout_indices(ds)
    # even stride subsample of each heldout slice (keeps size diversity)
    picks = []
    for s, ids in hi.items():
        if len(ids) > args.n_per_source:
            step = len(ids) / args.n_per_source
            ids = [ids[int(k * step)] for k in range(args.n_per_source)]
        picks += ids
    print(f"[ab] {len(picks)} heldout objects ({ {s: len(v) for s, v in hi.items()} })", flush=True)

    rows = []
    for n, i in enumerate(picks):
        item = ds[i]
        try:
            pm = store.lookup(item["image_name"])
        except KeyError:
            continue
        try:
            out = pipe.run(item["image"], None, 42, stage1_only=False, with_mesh_postprocess=False,
                           with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True,
                           stage1_inference_steps=None, pointmap=pm, decode_formats=["mesh"])
        except Exception as e:
            print(f"[ab] {item.get('uid','?')} FAILED {type(e).__name__}", flush=True); continue
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
        rows.append({"src": item["source"], "cat": item["category"], "gtmax": float(gt.max()),
                     "A": mape(A, gt), "B": mape(B, gt)})
        if (n + 1) % 25 == 0:
            print(f"[ab] {n+1}/{len(picks)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)

    def agg(sel, name):
        rs = [r for r in rows if sel(r)]
        if not rs:
            print(f"  {name:22s} n=0"); return
        a = np.mean([r["A"] for r in rs]); b = np.mean([r["B"] for r in rs])
        bwin = np.mean([r["B"] < r["A"] for r in rs]) * 100
        print(f"  {name:22s} n={len(rs):3d} | A(head) {a:5.1f}% | B(pointmap) {b:5.1f}% | "
              f"B wins {bwin:4.0f}% | winner={'B' if b < a else 'A'}")

    print("\n===== HEAD (A) vs POINTMAP (B) — W,H,D MAPE on heldout =====")
    print("-- by source --")
    for s in HELDOUT: agg(lambda r, s=s: r["src"] == s, s)
    print("-- by GT max-dim size bucket --")
    agg(lambda r: r["gtmax"] < 0.3, "small  (<0.3 m)")
    agg(lambda r: 0.3 <= r["gtmax"] < 0.6, "medium (0.3-0.6 m)")
    agg(lambda r: 0.6 <= r["gtmax"] < 1.2, "large  (0.6-1.2 m)")
    agg(lambda r: r["gtmax"] >= 1.2, "xlarge (>=1.2 m)")
    print("-- overall --")
    agg(lambda r: True, "ALL")
    print(f"\n[ab] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
