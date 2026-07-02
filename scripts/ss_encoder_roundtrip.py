#!/usr/bin/env python
"""SS (Sparse Structure / geometry) VAE encode->decode round-trip validation.

Confirms the downloaded ss_encoder.ckpt + ss_decoder.ckpt work end-to-end:
  mesh -> voxelize to 64^3 occupancy -> SS encoder -> latent (mean)
       -> SS decoder -> reconstructed occupancy -> IoU.

A high IoU (>~0.8) means the encoder produces latents the decoder reconstructs
faithfully, i.e. the encoder is usable for making geometry latents from meshes.

Usage:
  python scripts/ss_encoder_roundtrip.py mesh1.glb [mesh2.glb ...] \
      [--device cuda:0] [--out artifacts/ss_roundtrip]
"""
import argparse
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


def normalize_mesh_verts(verts):
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    center = (vmax + vmin) / 2.0
    max_extent = float(np.max(vmax - vmin))
    if max_extent == 0:
        return verts - center, 1.0, center
    return (verts - center) * (1.0 / max_extent), 1.0 / max_extent, center


def glb_to_occupancy(glb_path, resolution=RES):
    """Mesh file -> [1, R, R, R] float occupancy, following the SS demo notebook."""
    m = trimesh.load(glb_path)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    verts = np.asarray(m.vertices, dtype=np.float64)
    # Y-up -> Z-up: (x, y, z) -> (x, z, -y)
    verts = verts @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64).T
    verts, _, _ = normalize_mesh_verts(verts)
    verts = np.clip(verts, -0.5 + 1e-6, 0.5 - 1e-6)
    m.vertices = verts

    vg = m.voxelized(1.0 / resolution)
    try:
        vg = vg.fill(method="holes")
    except Exception:
        pass
    idx = np.clip(np.asarray(vg.sparse_indices), 0, resolution - 1)
    occ = torch.zeros(1, resolution, resolution, resolution, dtype=torch.float32)
    occ[:, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return occ


def save_voxel_ply(occ_3d, path, resolution=RES):
    coords = torch.nonzero(occ_3d, as_tuple=False).numpy()
    pts = (coords + 0.5) / resolution - 0.5
    trimesh.PointCloud(pts).export(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+", help="GLB/PLY/OBJ mesh paths")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="artifacts/ss_roundtrip")
    ap.add_argument("--enc", default=os.path.join(HF, "ss_encoder.ckpt"))
    ap.add_argument("--dec", default=os.path.join(HF, "ss_decoder.ckpt"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    encoder = SparseStructureEncoderTdfyWrapper(
        return_raw=True, in_channels=1, latent_channels=8,
        channels=[32, 128, 512], num_res_blocks=2, num_res_blocks_middle=2,
        pretrained_ckpt_path=args.enc,
    ).to(args.device).eval()
    decoder = SparseStructureDecoderTdfyWrapper(
        out_channels=1, latent_channels=8,
        channels=[512, 128, 32], num_res_blocks=2, num_res_blocks_middle=2,
        reshape_input_to_cube=False, pretrained_ckpt_path=args.dec,
    ).to(args.device).eval()
    print(f"encoder params: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M | "
          f"decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M")

    for mp in args.meshes:
        name = Path(mp).stem
        occ = glb_to_occupancy(mp)
        n_in = int(occ.sum().item())
        x = occ.unsqueeze(0).to(args.device)  # [1,1,R,R,R]
        with torch.no_grad():
            enc = encoder(x)
            mean = enc["mean"]
            logits = decoder(mean)
            recon = (torch.sigmoid(logits) > 0.5).float()
        in_b = x.bool().cpu()
        re_b = recon.bool().cpu()
        inter = int((in_b & re_b).sum())
        union = int((in_b | re_b).sum())
        iou = inter / union if union else 0.0
        n_out = int(recon.sum().item())
        print(f"\n=== {name} ===")
        print(f"  latent z: {list(enc['z'].shape)} | mean mu={mean.mean():.3f} sd={mean.std():.3f}")
        print(f"  occupancy in={n_in}  out={n_out}  ({100*n_in/RES**3:.2f}% -> {100*n_out/RES**3:.2f}%)")
        print(f"  IoU = {iou:.4f}")
        save_voxel_ply(occ[0], os.path.join(args.out, f"{name}_in.ply"))
        save_voxel_ply(recon[0, 0].cpu(), os.path.join(args.out, f"{name}_recon.ply"))
        print(f"  saved {args.out}/{name}_in.ply and _recon.ply")


if __name__ == "__main__":
    main()
