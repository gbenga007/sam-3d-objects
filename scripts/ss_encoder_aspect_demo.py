#!/usr/bin/env python
"""Proportion-corrected SS latent demo.

Takes a real SAM3D reconstruction (canonical mesh) whose proportions are WRONG
(the known aspect-ratio bug), and shows that:
  (a) encoding the native mesh -> decode reproduces the WRONG proportions, and
  (b) rescaling the mesh to the GT bounding-box proportions, then encoding ->
      decode reproduces the CORRECT proportions,
and that the two latents genuinely differ. This validates that a GT-proportion-
corrected mesh yields a GT-proportioned SS latent usable as a supervision target.

Reuses meshes + GT dims from artifacts/metric_scale/eval_v3_mesh_bbox/. No SAM3D
inference is run.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh

from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (
    SparseStructureEncoderTdfyWrapper,
    SparseStructureDecoderTdfyWrapper,
)

HF = "checkpoints/hf"
RES = 64
EVAL = "artifacts/metric_scale/eval_v3_mesh_bbox"


def load_mesh_verts_faces(path):
    m = trimesh.load(path)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def to_occupancy(mesh, resolution=RES):
    """Mesh -> [1,R,R,R] occupancy. Normalizes max-extent to 1 (canonical), so
    only PROPORTIONS survive — exactly what the SS encoder consumes."""
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vmin, vmax = verts.min(0), verts.max(0)
    center = (vmin + vmax) / 2
    max_ext = float((vmax - vmin).max())
    verts = (verts - center) / max_ext  # -> within [-0.5,0.5], aspect preserved
    verts = np.clip(verts, -0.5 + 1e-6, 0.5 - 1e-6)
    m2 = mesh.copy()
    m2.vertices = verts
    vg = m2.voxelized(1.0 / resolution)
    try:
        vg = vg.fill(method="holes")
    except Exception:
        pass
    idx = np.clip(np.asarray(vg.sparse_indices), 0, resolution - 1)
    occ = torch.zeros(1, resolution, resolution, resolution, dtype=torch.float32)
    occ[:, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return occ


def occ_proportions(occ_3d):
    """Normalized (max=1) sorted-desc bbox proportions of an occupancy grid."""
    c = torch.nonzero(occ_3d, as_tuple=False).numpy()
    if len(c) == 0:
        return np.zeros(3)
    ext = (c.max(0) - c.min(0) + 1).astype(float)
    return np.sort(ext / ext.max())[::-1]


def rescale_to_gt_proportions(mesh, gt_dims):
    """Anisotropically rescale mesh axes so its bbox proportions match gt_dims,
    rank-aligned (largest mesh axis -> largest GT dim) to dodge axis ambiguity."""
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vmin, vmax = verts.min(0), verts.max(0)
    center = (vmin + vmax) / 2
    ext = vmax - vmin                      # native per-axis extent
    order = np.argsort(ext)[::-1]          # axes large->small
    gt_sorted = np.sort(np.asarray(gt_dims, float))[::-1]
    target = np.zeros(3)
    for rank, axis in enumerate(order):
        target[axis] = gt_sorted[rank]
    scale = target / np.maximum(ext, 1e-9)
    verts = (verts - center) * scale + center
    m2 = mesh.copy()
    m2.vertices = verts
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top", type=int, default=3, help="show N most-anisotropic-error objects")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(os.path.join(EVAL, "mesh_scale_eval_v3.jsonl"))]
    # Rank objects by how much native proportions deviate from GT proportions.
    for r in records:
        raw = np.sort(np.asarray(r["raw_extents"], float))[::-1]
        raw = raw / raw.max()
        gt = np.sort(np.asarray(r["gt_dims"], float))[::-1]
        gt = gt / gt.max()
        r["_aniso_err"] = float(np.abs(raw - gt).sum())
    records.sort(key=lambda r: -r["_aniso_err"])

    plys = list(Path(os.path.join(EVAL, "plys")).glob("*_canonical.ply"))

    def match_ply(rec):
        """Find the canonical ply whose normalized-sorted extent matches raw_extents."""
        key = np.sort(np.asarray(rec["raw_extents"], float))[::-1]
        key = key / key.max()
        best, bestd = None, 1e9
        for p in plys:
            if not p.stem.startswith(rec["category"]):
                continue
            occ = to_occupancy(load_mesh_verts_faces(str(p)))
            pr = occ_proportions(occ[0])
            d = float(np.abs(pr - key).sum())
            if d < bestd:
                best, bestd = p, d
        return best, bestd

    encoder = SparseStructureEncoderTdfyWrapper(
        return_raw=True, in_channels=1, latent_channels=8,
        channels=[32, 128, 512], num_res_blocks=2, num_res_blocks_middle=2,
        pretrained_ckpt_path=os.path.join(HF, "ss_encoder.ckpt"),
    ).to(args.device).eval()
    decoder = SparseStructureDecoderTdfyWrapper(
        out_channels=1, latent_channels=8,
        channels=[512, 128, 32], num_res_blocks=2, num_res_blocks_middle=2,
        reshape_input_to_cube=False, pretrained_ckpt_path=os.path.join(HF, "ss_decoder.ckpt"),
    ).to(args.device).eval()

    def enc_dec(occ):
        x = occ.unsqueeze(0).to(args.device)
        with torch.no_grad():
            mean = encoder(x)["mean"]
            recon = (torch.sigmoid(decoder(mean)) > 0.5).float()
        return mean.cpu(), recon[0, 0].cpu()

    for rec in records[: args.top]:
        p, md = match_ply(rec)
        if p is None:
            continue
        mesh = load_mesh_verts_faces(str(p))
        gt_prop = np.sort(np.asarray(rec["gt_dims"], float))[::-1]
        gt_prop = gt_prop / gt_prop.max()

        occ_native = to_occupancy(mesh)
        occ_gt = to_occupancy(rescale_to_gt_proportions(mesh, rec["gt_dims"]))
        z_native, recon_native = enc_dec(occ_native)
        z_gt, recon_gt = enc_dec(occ_gt)

        print(f"\n=== {p.stem}  (category={rec['category']}, match_dist={md:.3f}) ===")
        print(f"  GT proportions (norm,sorted)        : {np.round(gt_prop,3)}")
        print(f"  native mesh proportions             : {np.round(occ_proportions(occ_native[0]),3)}")
        print(f"  -> decode(native latent) proportions: {np.round(occ_proportions(recon_native),3)}")
        print(f"  GT-rescaled mesh proportions        : {np.round(occ_proportions(occ_gt[0]),3)}")
        print(f"  -> decode(GT latent) proportions    : {np.round(occ_proportions(recon_gt),3)}")
        ad_native = float(np.abs(occ_proportions(recon_native) - gt_prop).sum())
        ad_gt = float(np.abs(occ_proportions(recon_gt) - gt_prop).sum())
        zdiff = float((z_native - z_gt).norm())
        print(f"  aspect-err vs GT: native={ad_native:.3f}  GT-corrected={ad_gt:.3f}")
        print(f"  || z_native - z_gt ||_2 = {zdiff:.2f}  (latents differ => correction is encoded)")


if __name__ == "__main__":
    main()
