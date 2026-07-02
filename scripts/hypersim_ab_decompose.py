#!/usr/bin/env python
"""Decompose B's Hypersim error into OVERALL-SIZE error (geometric-mean of dims, proportion-free) vs
PROPORTION error (per-axis MAPE after normalizing both pred & GT to unit geometric mean). Tests whether
B's ~32% per-axis MAPE is scale error (would be a problem) or proportion error from the generator's mesh."""
import sys, json, numpy as np, torch
sys.path.insert(0, "scripts")
from hypersim_head_vs_pointmap_ab import load_pipe, gt_pointmap, OMNI, META
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset

intr = {r["image_name"]: r["intrinsics"] for r in json.load(open(META))}
pipe = load_pipe()
ds = OmniNOCSObjectDataset(omninocs_root=OMNI, sources=["hypersim"],
                           rgb_roots={"hypersim": f"{OMNI}/hypersim"}, split="val",
                           max_records_per_source=4000, skip_missing_rgb=True)
N = len(ds); picks = list(range(0, N, max(1, N // 100)))[:100]
print(f"{len(picks)} sampled of {N}", flush=True)

gm = lambda x: float(np.exp(np.mean(np.log(np.clip(x, 1e-6, None)))))
rows = []
for c, i in enumerate(picks):
    it = ds[i]; pm = gt_pointmap(it["image_name"], intr.get(it["image_name"]))
    if pm is None: continue
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
    axis_mape = float(np.mean(np.abs(B - G) / G) * 100)
    size_err = abs(gm(B) - gm(G)) / gm(G) * 100                 # overall-size (proportion-free)
    Bn, Gn = B / gm(B), G / gm(G)                                # unit-gmean (size removed)
    prop_mape = float(np.mean(np.abs(Bn - Gn) / Gn) * 100)
    rows.append(dict(cat=it["category"], axis=axis_mape, size=size_err, prop=prop_mape,
                     B=B.round(3).tolist(), G=G.round(3).tolist()))
    if (c + 1) % 25 == 0: print(f"{c+1}/{len(picks)}", flush=True)

a = lambda k: (np.mean([r[k] for r in rows]), np.median([r[k] for r in rows]))
print(f"\nn={len(rows)}")
print(f"  per-axis MAPE      mean {a('axis')[0]:5.1f}  median {a('axis')[1]:5.1f}")
print(f"  OVERALL-SIZE err   mean {a('size')[0]:5.1f}  median {a('size')[1]:5.1f}   <-- proportion-free scale accuracy")
print(f"  PROPORTION MAPE    mean {a('prop')[0]:5.1f}  median {a('prop')[1]:5.1f}")
print("\nexamples (cat | B pred | GT | size-err% | prop%):")
for r in sorted(rows, key=lambda r: -r["axis"])[:6] + sorted(rows, key=lambda r: r["axis"])[:6]:
    print(f"  {r['cat']:12s} B={r['B']} G={r['G']}  size {r['size']:4.0f}%  prop {r['prop']:4.0f}%")
json.dump(rows, open("artifacts/metric_scale/hypersim_ab_decompose.json", "w"), indent=1)
print("DONE", flush=True)
