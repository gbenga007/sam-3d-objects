#!/usr/bin/env python
"""#2: filter Hypersim B by INPUT/GT properties (visibility + GT plausibility), to (a) verify the
catastrophic tail is partial-visibility / GT-outlier objects, and (b) report a clean B. Filters are on
inputs (mask size, frame-truncation, GT dims) NOT on whether B was wrong (that would be circular)."""
import sys, json, numpy as np
sys.path.insert(0, "scripts")
from hypersim_head_vs_pointmap_ab import load_pipe, gt_pointmap, OMNI, META
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset

intr = {r["image_name"]: r["intrinsics"] for r in json.load(open(META))}
pipe = load_pipe()
ds = OmniNOCSObjectDataset(omninocs_root=OMNI, sources=["hypersim"],
                           rgb_roots={"hypersim": f"{OMNI}/hypersim"}, split="val",
                           max_records_per_source=4000, skip_missing_rgb=True)
N = len(ds); picks = list(range(0, N, max(1, N // 100)))[:100]
print(f"{len(picks)} of {N}", flush=True)

rows = []
for c, i in enumerate(picks):
    it = ds[i]; pm = gt_pointmap(it["image_name"], intr.get(it["image_name"]))
    if pm is None: continue
    mask = it["image"][..., 3] > 127
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0: continue
    trunc = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1)
    occ = float(mask.sum()) / (H * W)                      # image occupancy
    try:
        out = pipe.run(it["image"], None, 42, stage1_only=False, with_mesh_postprocess=False,
                       with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True,
                       stage1_inference_steps=None, pointmap=pm, decode_formats=["mesh"])
    except Exception:
        continue
    m = out["mesh"]; m = m[0] if isinstance(m, (list, tuple)) else m
    v = m.vertices.detach().float().cpu().numpy()
    s = out["scale"][0].float().cpu().numpy().reshape(-1)
    B = np.sort((v.max(0) - v.min(0)) * (s.mean() if s.size else s))[::-1]
    G = np.sort(np.asarray(it["metric_dims"], float))[::-1]
    rows.append(dict(cat=it["category"], maskpx=int(mask.sum()), occ=occ, trunc=trunc,
                     gtmax=float(G[0]), gtaspect=float(G[0] / max(G[2], 1e-6)),
                     axis=float(np.mean(np.abs(B - G) / G) * 100)))
    if (c + 1) % 25 == 0: print(f"{c+1}/{len(picks)}", flush=True)

json.dump(rows, open("artifacts/metric_scale/hypersim_ab_filtered.json", "w"), indent=1)
R = rows


def stat(rs, name):
    if not rs: print(f"  {name:34s} n=0"); return
    e = [r["axis"] for r in rs]
    print(f"  {name:34s} n={len(rs):3d} | B mean {np.mean(e):5.1f} | median {np.median(e):5.1f}")


print(f"\nn={len(R)}  (truncated {sum(r['trunc'] for r in R)} | "
      f"maskpx<2000 {sum(r['maskpx']<2000 for r in R)} | gtmax>4m {sum(r['gtmax']>4 for r in R)} | "
      f"gtaspect>10 {sum(r['gtaspect']>10 for r in R)})")
print("\n-- correlation: error vs visibility/GT --")
stat(R, "ALL (no filter)")
stat([r for r in R if r["trunc"]], "truncated by frame")
stat([r for r in R if not r["trunc"]], "NOT truncated")
stat([r for r in R if r["maskpx"] < 2000], "tiny mask (<2000 px)")
stat([r for r in R if r["maskpx"] >= 8000], "well-seen (>=8000 px)")
stat([r for r in R if r["gtmax"] > 4], "GT > 4 m (implausible)")
stat([r for r in R if r["gtaspect"] > 10], "GT aspect > 10 (pancake)")
print("\n-- CLEAN B (filter out occluded/truncated/implausible INPUTS) --")
clean = [r for r in R if (not r["trunc"]) and r["maskpx"] >= 4000 and r["gtmax"] <= 4 and r["gtaspect"] <= 10]
stat(R, "before")
stat(clean, "after filter")
print(f"  removed {len(R)-len(clean)}/{len(R)} objects")
print("DONE", flush=True)
