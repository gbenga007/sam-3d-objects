#!/usr/bin/env python
"""
translation_scale oracle (model-free).

Tests the hypothesis from the pose algebra (pose_target.py):
    metric_object_size  =  s_tilde * translation_scale * s_scene
                           └scale┘   └─── EXCLUDED ───┘  └external┘
where translation_scale = |t_rel|  (governed by camera distance |translation|).

The MetricScaleHead reads s_tilde (the SS `scale` token) + a canonical/category
prior but NOT translation_scale. So its achievable accuracy is bounded by how much
of the metric size a (category prior + canonical shape) explains. The *additional*
variance explained by adding the distance factor log|t| is the headroom that
supervising translation_scale would unlock.

Per source we compare two oracle regressors of log(isotropic metric size):
    M1: category one-hot              (shape/category prior -- the head CAN learn this)
    M2: category one-hot + log|t|     (adds the distance factor translation_scale carries)

Residual RMSE(ln) ~ multiplicative error; we report exp(rmse)-1 as an approx MAPE.
This is an UPPER bound on translation_scale's value, since the real head also gets a
noisy distance proxy via MoGe pointmap_scale (not modeled here).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/source/datasets_sam3d/OmniNOCS")
SOURCES = {
    "nocs_real275": "omninocs_release_nocs_real275/nocs_real275_train_metadata.json",
    "objectron":    "omninocs_release_objectron/objectron_train_metadata.json",
    "arkitscenes":  "omninocs_release_ARKitScenes/ARKitScenes_train_metadata.json",
}
CAP = 8000  # objects/source for the population stats


def load(src, rel):
    p = ROOT / rel
    if not p.exists():
        print(f"  [skip] {src}: metadata not found at {p}")
        return None
    frames = json.load(open(p))
    d, size, cat = [], [], []
    for fr in frames:
        for o in fr["objects"]:
            t = np.asarray(o["translation"], float).reshape(-1)
            s = np.asarray(o["size"], float).reshape(-1)
            if t.shape[0] < 3 or s.shape[0] < 3:
                continue
            dist = float(np.linalg.norm(t))
            siso = float(np.max(s))               # max dim == canonical-bbox-max-dim convention
            if dist <= 1e-6 or siso <= 1e-6:
                continue
            d.append(dist); size.append(siso); cat.append(o["category"])
            if len(d) >= CAP:
                break
        if len(d) >= CAP:
            break
    return np.array(d), np.array(size), np.array(cat)


def ridge_fit(X, y, lam=1e-3):
    A = X.T @ X + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ y)
    pred = X @ w
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss = float(1 - np.sum(resid**2) / np.sum((y - y.mean())**2))
    return rmse, ss


def onehot(cats):
    u = sorted(set(cats))
    idx = {c: i for i, c in enumerate(u)}
    M = np.zeros((len(cats), len(u)))
    for i, c in enumerate(cats):
        M[i, idx[c]] = 1.0
    return M


def pct(rmse_ln):  # multiplicative error approx
    return (np.exp(rmse_ln) - 1) * 100


print(f"{'source':<14}{'N':>7}{'medD(m)':>9}{'std(lnD)':>10}{'distSpread%':>12}"
      f"{'corr(lnS,lnD)':>15}")
print("-" * 67)
data = {}
for src, rel in SOURCES.items():
    out = load(src, rel)
    if out is None:
        continue
    d, size, cat = out
    data[src] = (d, size, cat)
    lnD, lnS = np.log(d), np.log(size)
    corr = float(np.corrcoef(lnS, lnD)[0, 1])
    print(f"{src:<14}{len(d):>7}{np.median(d):>9.2f}{lnD.std():>10.3f}"
          f"{pct(lnD.std()):>11.0f}%{corr:>15.2f}")

print()
print(f"{'source':<14}{'M1 cat-only':>14}{'M2 cat+lnD':>14}{'headroom':>12}   (approx MAPE, residual)")
print("-" * 70)
for src, (d, size, cat) in data.items():
    lnD, lnS = np.log(d), np.log(size)
    C = onehot(cat)
    ones = np.ones((len(d), 1))
    X1 = np.hstack([C, ones])
    X2 = np.hstack([C, ones, lnD[:, None]])
    r1, ss1 = ridge_fit(X1, lnS)
    r2, ss2 = ridge_fit(X2, lnS)
    print(f"{src:<14}{pct(r1):>13.1f}%{pct(r2):>13.1f}%{pct(r1)-pct(r2):>11.1f}%"
          f"   R2: {ss1:.2f} -> {ss2:.2f}")

print()
print("Reading: M1 = what a perfect category+shape prior achieves (the head's reachable")
print("ceiling WITHOUT distance). M2 = adds the distance factor translation_scale encodes.")
print("'headroom' = approx MAPE the missing translation_scale factor is costing (upper bound;")
print("real head has a noisy MoGe distance proxy, so true gain is somewhat less).")
