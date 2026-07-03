# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fine-tune the metric-scale heads on OmniNOCS NOCS-Real275.

Two training modes:

Cached (default): SAM 3D backbone fully frozen. SS/SLAT latents are precomputed
once and cached; only MetricScaleHead + MetricScaleDecoder receive gradients.
Fast, but SLAT features are fixed.

SLAT-conditioned (--unfreeze-slat-cross-attn): The cross-attention layers in
all 24 SLatFlowModel transformer blocks are selectively unfrozen alongside the
metric heads. The scale token is injected into SLAT conditioning every step so
the denoiser learns to produce scale-aware geometry. Incompatible with caching;
requires --stage2-steps 1 to keep SLAT a single forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from sam3d_objects.data.dataset.metric import OmniNOCSObjectDataset, OmniNOCSReal275Dataset
from sam3d_objects.data.dataset.metric.omninocs import DEFAULT_SIZE_OUTLIER_LOG_TOL
from sam3d_objects.model.backbone.metric_scale_decoder import MetricScaleDecoder
from sam3d_objects.model.backbone.scale_head import (
    MetricScaleHead,
    _ScaleAugmentedEmbedderProxy,
    extract_ss_scale_features,
)
from sam3d_objects.data.dataset.tdfy.pose_target import PoseTargetConverter
from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching

# Layout modalities the SS MoT exposes (x1 keys for native flow-matching).
SS_FM_MODALITIES = ("shape", "scale", "translation", "6drotation_normalized", "translation_scale")


def shape_sample_seed(image_name: str) -> int:
    """Stable per-instance seed for the SS shape sample. Using the SAME seed in the native-FM
    training shape sample and the eval (predict_pose_metric) shape sample makes the canonical
    extent DETERMINISTIC per item, so the train target's canon == the eval decode's canon and the
    canon cancels exactly (fixes the native_fm_v1 stochastic-canon blowup). Conditioning differs
    per object (crop/mask), so a per-frame seed still yields per-object-deterministic shapes."""
    return int(hashlib.sha1(image_name.encode()).hexdigest()[:8], 16)


class AdaptiveGradClipper:
    """Tracks a rolling buffer of gradient norms and clips at the 95th percentile.

    Matches the approach in TRELLIS (grad_clip_utils.py). Uses a hard cap of
    max_norm until the buffer fills (buffer_size steps), then self-calibrates.
    Non-finite norms are passed through clip_grad_norm_ unchanged and do not
    update the buffer, so the caller's NaN guard remains responsible for skipping
    the optimizer step on bad gradients.
    """

    def __init__(self, max_norm: float = 1.0, clip_percentile: float = 95.0, buffer_size: int = 1000):
        self.max_norm = max_norm
        self._max_norm = max_norm
        self.clip_percentile = clip_percentile
        self.buffer_size = buffer_size
        self._grad_norms = np.zeros(buffer_size, dtype=np.float32)
        self._ptr = 0
        self._full = False

    def __call__(self, parameters) -> torch.Tensor:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=self._max_norm)
        if torch.isfinite(grad_norm):
            self._grad_norms[self._ptr] = float(grad_norm)
            self._ptr = (self._ptr + 1) % self.buffer_size
            if self._ptr == 0:
                self._full = True
            if self._full:
                self._max_norm = min(
                    float(np.percentile(self._grad_norms, self.clip_percentile)),
                    self.max_norm,
                )
        return grad_norm

    def log(self) -> dict:
        return {
            "max_norm": float(self._max_norm),
            "buffer_filled": bool(self._full),
        }


class DeviceOnlyPipeline:
    def __init__(self, device: str):
        self.device = torch.device(device)


def collate_instances(batch: list[dict]) -> dict:
    return {
        "images": [item["image"] for item in batch],
        "metric_dims": torch.tensor(
            [item["metric_dims"] for item in batch], dtype=torch.float32
        ),
        "translations": torch.tensor(
            [item.get("translation", [0.0, 0.0, 0.0]) for item in batch], dtype=torch.float32
        ),
        "has_translation": torch.tensor(
            [bool(item.get("has_translation", False)) for item in batch], dtype=torch.bool
        ),
        "categories": [item["category"] for item in batch],
        "uids": [item["uid"] for item in batch],
        "mask_pixels": torch.tensor(
            [item["mask_pixels"] for item in batch], dtype=torch.long
        ),
        "image_names": [item["image_name"] for item in batch],
    }


class Moge2PointmapStore:
    """
    Lookup of precomputed MoGe-2 global metric pointmaps keyed by image_name
    (scripts/precompute_moge2_pointmaps.py output: float16 [H,W,3] npy in
    PyTorch3D convention, NaN outside the MoGe-2 valid mask, plus manifest.json).

    A missing frame raises KeyError: silently falling back to live MoGe-v1 would
    mix non-metric anchors into a metric-anchor run.
    """

    def __init__(self, pointmap_dir: str):
        self.dir = Path(pointmap_dir)
        manifest_path = self.dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.frames: dict[str, str] = manifest["frames"]
        if not self.frames:
            raise ValueError(f"Empty MoGe-2 pointmap manifest: {manifest_path}")
        print(
            f"MoGe-2 pointmap store: {len(self.frames)} frames from {self.dir} "
            f"(model={manifest.get('model')})"
        )

    def lookup(self, image_name: str) -> torch.Tensor | None:
        fname = self.frames.get(image_name)
        if fname is None:
            return None  # not in store → caller falls back to live MoGe compute
        pm = np.load(self.dir / fname)
        return torch.from_numpy(pm.astype(np.float32))


ANCHOR_FEAT_DIM = 7  # log_anchor_iso (1) + log_pose_scale (3) + voxel_extent (3)


def load_anchor_tables(paths: str) -> dict:
    """
    uid -> anchor features from pose_scale_heldout_compare.py jsonl output
    (pose-decoder scale + canonical voxel extents under the live MoGe-2 pointmap,
    SS 25 steps = deployment condition). Comma-separated paths (train + heldout).
    """
    anchors: dict = {}
    for path in paths.split(","):
        path = path.strip()
        n = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line while the extraction is appending
                ps = torch.tensor(r["scale"], dtype=torch.float32).clamp(min=1e-6)
                ext = torch.tensor(r["voxel_extent"], dtype=torch.float32)
                anchors[r["uid"]] = torch.cat(
                    [
                        torch.log(torch.tensor([max(r["pred_iso"], 1e-6)])),
                        torch.log(ps),
                        ext,
                    ]
                )  # [7]
                n += 1
        print(f"anchor table {path}: {n} records")
    return anchors


def anchor_feats_for(features: dict, anchors: dict, device) -> torch.Tensor:
    return anchors[features["uid"]].to(device).unsqueeze(0)  # [1, 7]


class AnchorAugmentedDecoder(nn.Module):
    """
    v1a: same absolute log-WHD regression as MetricScaleDecoder, but with the
    pose-decoder anchor features appended to the MLP input. Trained with the
    standard log loss plus an auxiliary iso-scale loss (shared gradients).
    """

    def __init__(self, slat_feat_dim: int = 8, scale_token_dim: int = 1024,
                 scale_proj_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.scale_proj = nn.Linear(scale_token_dim, scale_proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slat_feat_dim + scale_proj_dim + ANCHOR_FEAT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    @staticmethod
    def _pool(slat_feats, batch_indices, batch_size, dtype):
        pooled = torch.zeros(batch_size, slat_feats.shape[1], device=slat_feats.device, dtype=dtype)
        counts = torch.zeros(batch_size, 1, device=slat_feats.device, dtype=dtype)
        idx = batch_indices.long().view(-1, 1).expand(-1, slat_feats.shape[1])
        pooled.scatter_add_(0, idx, slat_feats.to(dtype))
        counts.scatter_add_(0, batch_indices.long().view(-1, 1),
                            torch.ones(batch_indices.shape[0], 1, device=slat_feats.device, dtype=dtype))
        return pooled / counts.clamp(min=1)

    def forward(self, slat_feats, scale_token, batch_indices, anchor_feats):
        dtype = next(self.parameters()).dtype
        pooled = self._pool(slat_feats, batch_indices, scale_token.shape[0], dtype)
        scale_ctx = self.scale_proj(scale_token.squeeze(1).to(dtype))
        x = torch.cat([pooled, scale_ctx, anchor_feats.to(dtype)], dim=1)
        return self.mlp(x)  # [batch, 3] log(W,H,D)


class FactoredScaleDecoder(nn.Module):
    """
    v1b: MoGe-2-style decoupled prediction (Mogev2.pdf §3.2). Two MLPs on the
    same input: an iso branch predicting a log correction to the pose-decoder
    anchor iso, and a proportions branch predicting max-pinned log proportions.
    log_WHD = (log_anchor_iso + corr) + (log_props − max(log_props)), so
    max(log_WHD) == log_iso exactly: an iso loss on max(pred) reaches only the
    iso branch, a loss on max-pinned log dims reaches only the proportions
    branch — no gradient crosstalk by construction.
    """

    def __init__(self, slat_feat_dim: int = 8, scale_token_dim: int = 1024,
                 scale_proj_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.scale_proj = nn.Linear(scale_token_dim, scale_proj_dim)
        in_dim = slat_feat_dim + scale_proj_dim + ANCHOR_FEAT_DIM

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        self.iso_mlp = mlp(1)
        self.prop_mlp = mlp(3)

    def forward(self, slat_feats, scale_token, batch_indices, anchor_feats):
        dtype = next(self.parameters()).dtype
        pooled = AnchorAugmentedDecoder._pool(slat_feats, batch_indices, scale_token.shape[0], dtype)
        scale_ctx = self.scale_proj(scale_token.squeeze(1).to(dtype))
        anchor_feats = anchor_feats.to(dtype)
        x = torch.cat([pooled, scale_ctx, anchor_feats], dim=1)
        log_iso = anchor_feats[:, 0:1] + self.iso_mlp(x)        # residual to anchor iso
        log_props = self.prop_mlp(x)
        pinned = log_props - log_props.max(dim=1, keepdim=True).values  # ≤ 0, max = 0
        return log_iso + pinned  # [batch, 3] log(W,H,D); max(out) == log_iso


class BinnedScaleDecoder(nn.Module):
    """
    v2: OmniNOCS-style binned isotropic scale + regressed proportions.

    Motivated by NOCSformer (OmniNOCS §4.3): the metric *scale scalar* is
    discretized into bins with a softmax-CE loss (their ablation: discretized
    beats continuous regression). OmniNOCS bins the *absolute* size scalar, so
    we do the same — binning an absolute residual over the pose-decoder anchor
    caps the achievable correction and lets anchor outliers blow up the error.

    The single isotropic scalar — the max log-dimension — is binned over an
    ABSOLUTE fixed log range [log_min, log_max] (≈1cm..7m), ``num_bins``
    log-spaced bins. The prediction is the softmax-expected value over bin
    centers, so it is continuous within the range; cross-entropy pulls the
    distribution onto the correct bin and a GT-normalized L1 fine-tunes the
    expectation. The v1a anchor features (log_anchor_iso, log_pose_scale,
    voxel_extent) stay as MLP *inputs* — a soft prior the bin head can exploit
    — but never as a hard additive base. Proportions use the same max-pinned
    log-proportions regression branch as FactoredScaleDecoder:

        log_WHD = log_iso + (log_props - max(log_props))    # max(log_WHD)==log_iso

    so iso/bin supervision reaches only the iso (bin) branch and proportions
    supervision only the proportions branch — no gradient crosstalk, and a
    clean A/B against v1a that isolates "does binning the metric scalar help?".
    """

    def __init__(self, slat_feat_dim: int = 8, scale_token_dim: int = 1024,
                 scale_proj_dim: int = 16, hidden_dim: int = 128,
                 num_bins: int = 128, log_min: float = -5.0, log_max: float = 2.0):
        super().__init__()
        self.num_bins = num_bins
        self.scale_proj = nn.Linear(scale_token_dim, scale_proj_dim)
        in_dim = slat_feat_dim + scale_proj_dim + ANCHOR_FEAT_DIM

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        self.bin_mlp = mlp(num_bins)
        self.prop_mlp = mlp(3)
        centers = torch.linspace(log_min, log_max, num_bins)        # absolute log-metres
        self.register_buffer("bin_centers", centers)                # [num_bins]
        # midpoints between adjacent centers → nearest-bin classification target
        self.register_buffer("bin_edges", (centers[:-1] + centers[1:]) / 2)

    def forward(self, slat_feats, scale_token, batch_indices, anchor_feats,
                return_aux: bool = False):
        dtype = next(self.parameters()).dtype
        pooled = AnchorAugmentedDecoder._pool(slat_feats, batch_indices, scale_token.shape[0], dtype)
        scale_ctx = self.scale_proj(scale_token.squeeze(1).to(dtype))
        anchor_feats = anchor_feats.to(dtype)
        x = torch.cat([pooled, scale_ctx, anchor_feats], dim=1)

        bin_logits = self.bin_mlp(x)                                       # [B, num_bins]
        probs = torch.softmax(bin_logits, dim=-1)
        log_iso = (probs * self.bin_centers.to(dtype)).sum(dim=-1, keepdim=True)  # absolute soft expectation

        log_props = self.prop_mlp(x)
        pinned = log_props - log_props.max(dim=1, keepdim=True).values
        log_whd = log_iso + pinned                                         # [B, 3], max == log_iso
        if return_aux:
            return log_whd, {
                "bin_logits": bin_logits,
                "log_iso": log_iso,
                "log_anchor_iso": anchor_feats[:, 0:1],
            }
        return log_whd

    def target_bin(self, log_iso_target: torch.Tensor) -> torch.Tensor:
        """Nearest-bin index for the absolute target iso (max log-dimension)."""
        t = log_iso_target.detach().view(-1)
        return torch.bucketize(t, self.bin_edges.to(t.dtype)).clamp(0, self.num_bins - 1)


def freeze_pipeline(pipeline) -> None:
    for model in pipeline.models.values():
        if model is None:
            continue
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)


def _upcast_module_to_fp32_with_shims(module) -> None:
    """
    Convert an nn.Module's params/buffers to fp32, and wrap forward to cast
    floating-point tensor inputs to fp32 and outputs back to the input dtype.

    Used to run SLAT cross-attention in fp32 while the rest of SLAT stays in
    bf16/fp16. The bf16 attention backward through cross_attn.to_kv overflowed
    in nocs_sceneholdout_slat_conditioned_v1 and corrupted 144/192 cross_attn
    weight tensors to NaN after the very first optimizer step. fp32 math here
    prevents that overflow; the per-block memory cost is small (~17M params *
    4B = ~70MB extra) and only the cross-attn portion runs in fp32.
    """
    module.float()
    orig_forward = module.forward

    def fp32_forward(*args, **kwargs):
        in_dtype = None
        new_args = []
        for a in args:
            if torch.is_tensor(a) and a.is_floating_point():
                if in_dtype is None:
                    in_dtype = a.dtype
                new_args.append(a.float())
            else:
                new_args.append(a)
        new_kwargs = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                if in_dtype is None:
                    in_dtype = v.dtype
                new_kwargs[k] = v.float()
            else:
                new_kwargs[k] = v
        out = orig_forward(*new_args, **new_kwargs)
        if in_dtype is None or in_dtype == torch.float32:
            return out
        if torch.is_tensor(out) and out.is_floating_point():
            return out.to(in_dtype)
        if isinstance(out, tuple):
            return tuple(
                o.to(in_dtype) if torch.is_tensor(o) and o.is_floating_point() else o
                for o in out
            )
        return out

    module.forward = fp32_forward


def collect_slat_cross_attn_params(pipeline, upcast_fp32: bool = True) -> tuple[list, object]:
    """
    Selectively unfreeze the cross-attention layers (cross_attn + norm2) in
    every SLatFlowModel transformer block after freeze_pipeline has run.

    When upcast_fp32 is True, cross_attn and norm2 are converted to fp32 with
    input/output dtype shims so the cross-attention math runs in fp32 while
    the rest of SLAT stays in bf16/fp16. This prevents the bf16 backward
    overflow that NaN-corrupted SLAT cross-attn weights after step 1 of
    nocs_sceneholdout_slat_conditioned_v1.

    Returns (unfrozen_param_list, slat_backbone).  The backbone reference is
    needed to save its state dict in the checkpoint.
    """
    try:
        backbone = pipeline.models["slat_generator"].reverse_fn.backbone
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            "Cannot locate slat_generator.reverse_fn.backbone — "
            "is the pipeline fully loaded?"
        ) from exc

    unfrozen: list = []
    for block in backbone.blocks:
        if upcast_fp32:
            _upcast_module_to_fp32_with_shims(block.cross_attn)
            _upcast_module_to_fp32_with_shims(block.norm2)
        for p in block.cross_attn.parameters():
            p.requires_grad_(True)
            unfrozen.append(p)
        for p in block.norm2.parameters():
            p.requires_grad_(True)
            unfrozen.append(p)
        block.cross_attn.train()
        block.norm2.train()

    n_blocks = len(backbone.blocks)
    n_params = sum(p.numel() for p in unfrozen)
    dtype_note = " (cross_attn+norm2 upcast to fp32)" if upcast_fp32 else ""
    print(
        f"Unfrozen SLAT cross-attn in {n_blocks} blocks "
        f"({n_params:,} params across cross_attn + norm2){dtype_note}"
    )
    return unfrozen, backbone


def collect_ss_decoder_params(pipeline) -> list:
    """
    Unfreeze the SS decoder so it can receive gradients from the aspect ratio loss.

    Returns the unfrozen param list for the optimizer.  The backbone stays frozen;
    only the convolutional decoder (16^3 → 64^3) learns to map shape latents to
    more proportionally accurate occupancy volumes.
    """
    if "ss_decoder" not in pipeline.models:
        raise RuntimeError("Cannot locate ss_decoder in pipeline.models.")
    ss_decoder = pipeline.models["ss_decoder"]
    unfrozen: list = []
    for p in ss_decoder.parameters():
        p.requires_grad_(True)
        unfrozen.append(p)
    ss_decoder.train()
    n_params = sum(p.numel() for p in unfrozen)
    print(f"Unfrozen SS decoder ({n_params:,} params)")
    return unfrozen


# The MoT/SS generator predicts pose as separate modality `Latent` heads on the shared
# transformer `blocks`. These are the pose/scale OUTPUT heads (W,H,D + translation source).
SS_POSE_MODALITIES = ("scale", "translation", "6drotation_normalized", "translation_scale")


SS_LAYOUT_KEY = "6drotation_normalized"  # the merged share-transformer modality (R/t/s/translation_scale)


def collect_ss_backbone_params(
    pipeline,
    unfreeze_cross_attn: bool = False,
    upcast_fp32: bool = True,
) -> tuple[dict, object]:
    """
    Unfreeze the SS generator (MoT) backbone for layout-only SFT, routing by the MoT's
    ModuleDict MODALITY KEY (not by `latent_mapping.` substring as before).

    The SS generator is a true Mixture-of-Transformers: every per-block param is a ModuleDict
    keyed `["shape", "6drotation_normalized"]` (norm1/2/3, self_attn.to_qkv/to_out, cross_attn,
    mlp); the only shared param is per-block `adaLN_modulation` (timestep mod). Shape and layout
    are weight-DISJOINT, and the attention mask + k/v `.detach()` on the shape modality mean a
    layout loss has ZERO gradient path to shape (paper §C.2: "freeze shape, finetune layout").

    So we FREEZE the shape transformer + all shared/conditioning params (geometry preserved by
    construction — no L2-SP needed) and TRAIN only the LAYOUT transformer:

      'pose' : every `.{SS_LAYOUT_KEY}.`-keyed per-block param (norms, self-attn, mlp, and —
               when ``unfreeze_cross_attn`` — the layout cross_attn to image/mask/pointmap) +
               the scale / translation / 6drotation_normalized / translation_scale read heads.
               This is the full per-block layout capacity, NOT just the ~0.03M read heads the
               old `latent_mapping.`-only filter trained (the joint_mot_v1 size-plateau bug).
      'geometry' / 'cond' : kept for plumbing compatibility, left EMPTY (shape frozen).

    Layout cross_attn is fp32-upcast with shims (bf16 cross-attn backward overflow guard).
    Returns ({'geometry': [], 'pose': [...], 'cond': []}, backbone).
    """
    try:
        backbone = pipeline.models["ss_generator"].reverse_fn.backbone
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            "Cannot locate ss_generator.reverse_fn.backbone — is the pipeline fully loaded?"
        ) from exc

    # Freeze everything first; we re-enable only the layout-modality params below.
    backbone.requires_grad_(False)

    # fp32-upcast ONLY the layout cross_attn (so the fp32 tensors are the ones trained).
    if unfreeze_cross_attn and upcast_fp32:
        for block in backbone.blocks:
            ca = getattr(block, "cross_attn", None)
            if ca is not None and SS_LAYOUT_KEY in ca:
                _upcast_module_to_fp32_with_shims(ca[SS_LAYOUT_KEY])

    groups: dict = {"geometry": [], "pose": [], "cond": []}
    for name, p in backbone.named_parameters():
        is_layout_head = any(f"latent_mapping.{m}." in name for m in SS_POSE_MODALITIES)
        is_layout_block = name.startswith("blocks") and f".{SS_LAYOUT_KEY}." in name
        is_cross = ".cross_attn." in name
        if is_layout_head or (is_layout_block and not is_cross):
            p.requires_grad_(True)
            groups["pose"].append(p)
        elif is_layout_block and is_cross and unfreeze_cross_attn:
            p.requires_grad_(True)
            groups["pose"].append(p)
        # everything else (shape transformer, latent_mapping.shape, adaLN_modulation,
        # t_embedder, condition_embedder, shape cross_attn) stays FROZEN.

    backbone.train()
    n_train = sum(p.numel() for p in groups["pose"])
    n_total = sum(p.numel() for p in backbone.parameters())
    print(
        f"SS backbone layout-only SFT: training {len(groups['pose'])} tensors / "
        f"{n_train / 1e6:.2f}M of {n_total / 1e6:.2f}M ({100 * n_train / n_total:.1f}%); "
        f"shape transformer + shared params FROZEN."
    )
    return groups, backbone


def compute_ss_aspect_ratio_loss(
    ss_logits: torch.Tensor,
    gt_dims: torch.Tensor,
) -> torch.Tensor:
    """
    Scale-invariant aspect ratio loss between SS soft occupancy and GT metric dims.

    Uses soft marginal variance along each voxel axis as a differentiable proxy
    for bounding-box extent.  Rank-sorted descending comparison handles axis
    permutation ambiguity (no fixed mapping between voxel axes and GT W/H/D).

    ss_logits: [B, 1, 64, 64, 64]  — pre-threshold logits from the SS decoder
    gt_dims:   [B, 3]               — GT metric dims [W, H, D] in metres
    Returns:   scalar smooth-L1 loss on normalised log aspect ratios
    """
    probs = torch.sigmoid(ss_logits).squeeze(1)  # [B, VD, VH, VW]
    B, VD, VH, VW = probs.shape
    device, dtype = probs.device, probs.dtype
    eps = 1e-6

    d_idx = torch.arange(VD, device=device, dtype=dtype)
    h_idx = torch.arange(VH, device=device, dtype=dtype)
    w_idx = torch.arange(VW, device=device, dtype=dtype)

    p_d = probs.sum(dim=[2, 3])  # [B, VD]
    p_h = probs.sum(dim=[1, 3])  # [B, VH]
    p_w = probs.sum(dim=[1, 2])  # [B, VW]

    norm_d = p_d.sum(1, keepdim=True) + eps
    norm_h = p_h.sum(1, keepdim=True) + eps
    norm_w = p_w.sum(1, keepdim=True) + eps

    mean_d = (p_d * d_idx).sum(1, keepdim=True) / norm_d
    mean_h = (p_h * h_idx).sum(1, keepdim=True) / norm_h
    mean_w = (p_w * w_idx).sum(1, keepdim=True) / norm_w

    var_d = (p_d * (d_idx - mean_d) ** 2).sum(1) / norm_d.squeeze(1)
    var_h = (p_h * (h_idx - mean_h) ** 2).sum(1) / norm_h.squeeze(1)
    var_w = (p_w * (w_idx - mean_w) ** 2).sum(1) / norm_w.squeeze(1)

    # Per-axis std-dev in voxels [B, 3], rank-sorted descending
    log_ext = torch.stack([var_w.sqrt(), var_h.sqrt(), var_d.sqrt()], dim=1).clamp(min=eps).log()
    log_ext_sorted, _ = log_ext.sort(dim=1, descending=True)

    log_gt = torch.log(gt_dims.to(device=device, dtype=dtype).clamp(min=eps))
    log_gt_sorted, _ = log_gt.sort(dim=1, descending=True)

    # Normalise to remove global scale — aspect ratio comparison only
    log_ext_norm = log_ext_sorted - log_ext_sorted.mean(dim=1, keepdim=True)
    log_gt_norm = log_gt_sorted - log_gt_sorted.mean(dim=1, keepdim=True)

    return F.smooth_l1_loss(log_ext_norm, log_gt_norm)


def load_pipeline(config_path: str, device: str, compile_model: bool = False):
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "Hydra/OmegaConf are required to instantiate the SAM 3D pipeline. "
            "Install the repo requirements in the active environment, including "
            "hydra-core==1.3.2."
        ) from exc

    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)
    config.compile_model = compile_model
    config.device = device
    pipeline = instantiate(config)
    freeze_pipeline(pipeline)
    return pipeline


def build_dataset(args):
    # negative tol => disable the size-outlier filter
    size_tol = args.size_outlier_log_tol if args.size_outlier_log_tol >= 0 else None
    if args.dataset == "omninocs-mixed":
        rgb_roots = {
            "nocs_real275": args.rgb_root,
            "objectron": args.objectron_rgb_root,
            "arkitscenes": args.arkitscenes_rgb_root,
            "hypersim": args.hypersim_rgb_root,
        }
        dataset = OmniNOCSObjectDataset(
            omninocs_root=args.omninocs_root,
            sources=args.omninocs_sources,
            rgb_roots=rgb_roots,
            split=args.split,
            categories=args.categories,
            min_mask_pixels=args.min_mask_pixels,
            max_records=args.max_records,
            max_records_per_source=args.max_records_per_source,
            skip_missing_rgb=args.skip_missing_rgb,
            size_outlier_log_tol=size_tol,
        )
        print(
            "Loaded OmniNOCS mixed dataset "
            f"sources={args.omninocs_sources} records={len(dataset)} "
            f"skipped_missing_rgb={dataset.skipped_missing_rgb} "
            f"skipped_missing_mask={dataset.skipped_missing_mask}"
        )
        return dataset

    return OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split=args.split,
        categories=args.categories,
        min_mask_pixels=args.min_mask_pixels,
        max_records=args.max_records,
    )


def predict_log_dims(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    image,
    stage1_steps: int | None,
    stage2_steps: int | None,
    inject_scale_token: bool = False,
    unfreeze_cross_attn: bool = False,
    p_uncond_scale_token: float = 0.0,
    ss_ratio_loss_weight: float = 0.0,
    gt_dims: torch.Tensor | None = None,
    pointmap: torch.Tensor | None = None,
    unfreeze_layout_for_head: bool = False,
    return_translation: bool = False,
) -> tuple[torch.Tensor, bool, torch.Tensor, torch.Tensor | None]:
    """
    Run the full SAM3D pipeline and return predicted log-dimensions.

    pointmap: optional precomputed global pointmap [H,W,3] (e.g. MoGe-2 metric,
        PyTorch3D convention) passed through to compute_pointmap in place of the
        live MoGe-v1 call — makes both the SS conditioning and pointmap_scale/shift
        metric.

    inject_scale_token: append the scale token to SLAT conditioning before
        each denoiser step.

    unfreeze_cross_attn: when True, SLAT runs WITHOUT torch.no_grad() so
        gradients flow through the unfrozen cross-attention layers back to the
        scale token and MetricScaleHead.  scale_token and slat.feats are not
        detached.  Requires inject_scale_token=True and --stage2-steps 1 to
        keep the backward graph tractable.

    p_uncond_scale_token: with this probability, zero the SLAT-injected scale
        token (CFG-style dropout from TRELLIS p_uncond=0.1).  The
        MetricScaleDecoder still receives the unzeroed token so the metric
        regression signal is preserved, while SLAT cross-attention learns to
        produce reasonable geometry without depending on the metric token.

    ss_ratio_loss_weight: when > 0, run the SS decoder a second time with
        gradient enabled (SS backbone stays frozen) to compute a differentiable
        aspect ratio loss on the soft occupancy output.  Requires gt_dims.
        Needs --unfreeze-ss-decoder to have a gradient path into the decoder.

    Returns (log_dims, scale_token_dropped, ss_ratio_loss).
    """
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(image, pointmap=pointmap)
        ss_input_dict = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input_dict = pipeline.preprocess_image(image, pipeline.slat_preprocessor)

        # SS stage: frozen unless --unfreeze-ss-layout-for-head.  sample_sparse_structure
        # has its own internal grad_ctx (default torch.no_grad); with_grad=True passes
        # torch.enable_grad() so the layout modality outputs carry a gradient back into
        # the layout transformer weights.  Shape transformer is structurally protected by
        # the stop-grad k/v detach in mm_scale_dot_product_attention.
        ss_return_dict = pipeline.sample_sparse_structure(
            ss_input_dict,
            inference_steps=stage1_steps,
            use_distillation=False,
            with_grad=unfreeze_layout_for_head,
        )

        # SS aspect ratio loss (decoder-only, backbone stays frozen).
        # Run the SS decoder again outside no_grad on the detached shape_latent so
        # gradients flow into the decoder weights only.  Requires --unfreeze-ss-decoder.
        ss_ratio_loss = torch.zeros((), device=pipeline.device)
        if ss_ratio_loss_weight > 0.0 and gt_dims is not None:
            if "ss_decoder" in pipeline.models:
                _ss_dec = pipeline.models["ss_decoder"]
            else:
                _ss_dec = None
            if _ss_dec is not None:
                _shape = ss_return_dict["shape"].detach()
                _B = _shape.shape[0]
                _ss_logits = _ss_dec(
                    _shape.permute(0, 2, 1).contiguous().view(_B, 8, 16, 16, 16)
                )
                ss_ratio_loss = compute_ss_aspect_ratio_loss(
                    _ss_logits, gt_dims.to(device=pipeline.device)
                )

        # Scale token computed with gradient — feeds both SLAT conditioning and
        # the decoder directly.  ss_scale_features (the 3-dim log-SSI scale readout)
        # is the gradient path into the layout transformer: detached only when SS is
        # frozen; left live under --unfreeze-ss-layout-for-head so the layout MoT
        # blocks learn to produce a better scale token.  shape_latent (8-dim pooled
        # proportions) stays detached ALWAYS — the shape transformer is frozen by
        # collect_ss_backbone_params, so a live shape_latent would only retain its
        # backward graph for params that never update (wasted memory, no learning).
        ss_scale_features = extract_ss_scale_features(ss_return_dict)
        if ss_scale_features is not None:
            ss_scale_features = ss_scale_features.to(pipeline.device)
            if not unfreeze_layout_for_head:
                ss_scale_features = ss_scale_features.detach()

        # Unified recipe (dims head + joint-MoT translation loss): pose-decode the SAME
        # gradient-carrying SS sample so compute_translation_loss can train the layout
        # transformer + translation read-head alongside the metric head. Decoded on a
        # shallow COPY so the pose output never overwrites the raw `scale` modality the
        # head consumes (the pipeline.run scale-overwrite bug, fixed 2026-07-02).
        # Translation is unaffected by downsample_factor (only `scale` is dsf-rescaled).
        pred_translation = None
        if return_translation:
            _pose_out = pipeline.pose_decoder(
                dict(ss_return_dict),
                scene_scale=ss_input_dict.get("pointmap_scale"),
                scene_shift=ss_input_dict.get("pointmap_shift"),
            )
            pred_translation = _pose_out.get("translation")

        scale_token = scale_head(
            ss_return_dict["shape"].detach(),
            ss_scale_features,
            ss_input_dict.get("pointmap_scale"),
            ss_input_dict.get("pointmap_shift"),
        )

        # Inject the scale token into SLAT conditioning.
        # When cross-attn is frozen: detach the token so gradients stay in the
        # metric heads only.  When cross-attn is unfrozen: keep the token live
        # so gradients flow through cross_attn.to_kv back to MetricScaleHead.
        orig_backbone_emb = None
        orig_external_emb = None
        slat_backbone = None
        scale_token_dropped = False
        if inject_scale_token and hasattr(pipeline, "_get_slat_backbone"):
            # Normalize the token for SLAT injection only — the MetricScaleHead output
            # magnitude is unconstrained (trained for MetricScaleDecoder, not cross-attn).
            # Layer-normalizing a copy keeps the decoder's warm-start valid while
            # preventing bfloat16 attention overflow in SLAT cross_attn.
            import torch.nn.functional as F
            token_for_cond_base = F.layer_norm(scale_token, [scale_token.shape[-1]])
            if p_uncond_scale_token > 0.0 and torch.rand(1).item() < p_uncond_scale_token:
                # CFG-style dropout: SLAT sees a zero token; decoder still sees the real one.
                token_for_cond = torch.zeros_like(token_for_cond_base)
                scale_token_dropped = True
            else:
                token_for_cond = token_for_cond_base if unfreeze_cross_attn else token_for_cond_base.detach()
            slat_backbone = pipeline._get_slat_backbone()
            if slat_backbone is not None:
                orig_backbone_emb = slat_backbone.condition_embedder
                slat_backbone.condition_embedder = _ScaleAugmentedEmbedderProxy(
                    orig_backbone_emb, token_for_cond
                )
            cond_embedders = getattr(pipeline, "condition_embedders", {})
            orig_external_emb = cond_embedders.get("slat_condition_embedder")
            if orig_external_emb is not None:
                pipeline.condition_embedders["slat_condition_embedder"] = (
                    _ScaleAugmentedEmbedderProxy(orig_external_emb, token_for_cond)
                )

        try:
            slat = pipeline.sample_slat(
                slat_input_dict,
                ss_return_dict["coords"],
                inference_steps=stage2_steps,
                use_distillation=False,
                with_grad=unfreeze_cross_attn,
            )
        finally:
            if orig_backbone_emb is not None and slat_backbone is not None:
                slat_backbone.condition_embedder = orig_backbone_emb
            if orig_external_emb is not None:
                pipeline.condition_embedders["slat_condition_embedder"] = orig_external_emb

        # When cross-attn is unfrozen, don't detach slat.feats — gradients must
        # flow from the decoder through SLAT back to the scale token.
        slat_feats = slat.feats if unfreeze_cross_attn else slat.feats.detach()
        log_dims = scale_decoder(
            slat_feats,
            scale_token,
            slat.coords[:, 0],
        )
        return log_dims, scale_token_dropped, ss_ratio_loss, pred_translation


def compute_translation_loss(
    pred_translation: torch.Tensor, gt_translation: torch.Tensor
) -> torch.Tensor:
    """Smooth-L1 on metric translation — the mAP lever (lets the MoT learn to absorb the
    depth-source bias, e.g. MoGe-2's systematic ~13% too-close). [B,3] in metres."""
    return F.smooth_l1_loss(pred_translation, gt_translation.to(pred_translation))


def _voxel_canonical_extents(ss_return_dict: dict) -> torch.Tensor:
    """Per-axis canonical extent [B,3] (stopgrad) = the occupied SS-voxel bbox in the canonical
    [-0.5,0.5]^3 frame (coords/64). Confirmed by the SS-VAE round-trip notebook: normalize_mesh_verts
    puts geometry in [-0.5,0.5]^3 (max axis = 1.0), and voxel/mesh/gaussian extents all agree (~O(1)).
    Free (coords already computed) — no SLAT decode. metric_dims = scale x canon_ext; the absolute
    calibration is LEARNED by the scale modality head via L_whd (mesh & box stay consistent since
    rendered_mesh = canon_ext x scale = the box)."""
    coords = ss_return_dict["coords"]
    device = coords.device
    bs = int(coords[:, 0].max().item()) + 1 if coords.numel() else 1
    exts = []
    for b in range(bs):
        v = coords[coords[:, 0] == b][:, 1:].float()
        if v.numel() == 0:
            exts.append(torch.ones(3, device=device))
            continue
        ext = (v.max(0).values - v.min(0).values + 1.0) / 64.0   # [-0.5,0.5] frame fraction
        exts.append(ext.clamp(min=1e-3))
    return torch.stack(exts, 0).detach()


def predict_pose_metric(
    pipeline,
    image,
    stage1_steps: int | None,
    pointmap: torch.Tensor | None = None,
    with_grad: bool = True,
    shape_seed: int | None = None,
) -> dict:
    """
    Joint-MoT metric path: run the SS generator (MoT) and read metric per-axis W,H,D +
    translation directly off its pose outputs via the pose convention (ssi_to_metric), instead
    of the bolt-on MetricScaleHead / scale-token / MetricScaleDecoder.

    Only the SS flow runs (stage1_steps Euler steps); SLAT is NOT needed — W,H,D = scale_per_axis
    (un-collapsed instance_scale_l2c) x stopgrad(voxel canon_ext), translation = instance_position_l2c,
    all from the SS pose decoder + coords. No mesh/SLAT decode.

    with_grad: run SS WITHOUT torch.no_grad so gradients reach the unfrozen MoT
        (collect_ss_backbone_params). Gradient backprops through all stage1_steps denoiser calls —
        keep stage1_steps modest (recipe uses 4; grad-checkpointing on the SS blocks cushions it).

    Returns {log_dims [B,3], scale_per_axis, canon_extents, translation [B,3], rotation [B,4]}.
    """
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(image, pointmap=pointmap)
        ss_input_dict = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        # Deterministic shape sample (eval): same seed as the native-FM training shape sample for
        # this item -> identical canon -> exact cancellation. None preserves the old joint-MoT path.
        if shape_seed is not None:
            torch.manual_seed(shape_seed)
            torch.cuda.manual_seed_all(shape_seed)
        ss_return_dict = pipeline.sample_sparse_structure(
            ss_input_dict, inference_steps=stage1_steps, use_distillation=False,
            with_grad=with_grad,
        )
        pose = pipeline.pose_decoder(
            ss_return_dict,
            scene_scale=ss_input_dict.get("pointmap_scale"),
            scene_shift=ss_input_dict.get("pointmap_shift"),
        )
        # downsample_factor rescales metric scale exactly as the pipeline does for "scale".
        dsf = ss_return_dict.get("downsample_factor", 1.0)
        # instance_scale_l2c is a multiplier on the tiny TRELLIS-canonical mesh, NOT the metric
        # W,H,D directly. Recover dims = scale x stopgrad(canonical_extents), where canon_extents
        # is the REAL decoded-geometry bbox (gaussian xyz) — so `scale` stays a valid mesh-render
        # multiplier and `scale x canon_ext` is metric (validated by step 0.5: mesh_ext x scale).
        scale_per_axis = pose["scale_per_axis"] * dsf            # [B,3]
        canon_ext = _voxel_canonical_extents(ss_return_dict)     # [B,3] stopgrad, [-0.5,0.5] frame
        metric_dims = scale_per_axis * canon_ext.to(scale_per_axis)
        return {
            "log_dims": metric_dims.clamp(min=1e-6).log(),
            "scale_per_axis": scale_per_axis,
            "canon_extents": canon_ext,
            "translation": pose["translation"],
            "rotation": pose["rotation"],
        }


def _voxel_canonical_extents(ss_return_dict: dict) -> torch.Tensor:
    """Per-axis canonical extent proxy [B,3] from the occupied SS voxel bbox (stopgrad).
    coords: [N,4] = (batch_idx, x, y, z) voxel indices on the 64^3 grid."""
    coords = ss_return_dict["coords"]
    device = coords.device
    bs = int(coords[:, 0].max().item()) + 1 if coords.numel() else 1
    exts = []
    for b in range(bs):
        v = coords[coords[:, 0] == b][:, 1:].float()
        if v.numel() == 0:
            exts.append(torch.ones(3, device=device))
            continue
        ext = (v.max(0).values - v.min(0).values + 1.0) / 64.0   # grid fraction in [0,1]
        exts.append(ext.clamp(min=1e-3))
    return torch.stack(exts, 0).detach()                          # stopgrad target


def encode_gt_layout_x1(gt_scale_l2c, gt_translation, scene_scale, scene_shift, device):
    """GT metric (per-axis instance scale + translation) -> SSI modality latents (x1) for the
    scale / translation / translation_scale modalities. Validated by /tmp/roundtrip_x1.py (encode
    GT -> decode reconstructs GT to 1e-7). Rotation x1 is independent of these (round-trip used a
    random quaternion), so we pass identity here and keep the model's self-predicted rotation x1."""
    ident_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    pt = PoseTargetConverter.dicts_instance_pose_to_pose_target(
        "ScaleShiftInvariantWTranslationScale",
        instance_scale_l2c=gt_scale_l2c,
        instance_position_l2c=gt_translation,
        instance_quaternion_l2c=ident_q,
        scene_scale=scene_scale,
        scene_shift=scene_shift,
    )
    return {
        "scale": torch.log(pt["x_instance_scale"].clamp_min(1e-9)),
        "translation": pt["x_instance_translation"],
        "translation_scale": torch.log(pt["x_translation_scale"].clamp_min(1e-9)),
    }


def native_fm_layout_loss(pipeline, image, gt_dims, gt_translation, pointmap,
                          shape_steps, loss_weights, shape_seed=None):
    """Native rectified flow-matching SFT of the LAYOUT modalities (the paper's L_CFM, §C.2).

    Pretraining-consistent: noise every modality at ONE random tau, a single forward, velocity
    MSE weighted per modality — NOT multi-step sample-then-regress (predict_pose_metric). Cheaper
    (one grad forward) and what the model was trained with.

    Shape + rotation x1 = the model's own frozen self-prediction (no GT mesh / no GT rotation
    needed; their loss weights are 0). Scale + translation x1 = GT-derived SSI targets
    (encode_gt_layout_x1). loss_weights override zeroes shape/rotation, weights scale+translation.
    """
    ss_generator = pipeline.models["ss_generator"]
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(image, pointmap=pointmap)
        ss_input_dict = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        scene_scale = ss_input_dict.get("pointmap_scale")
        scene_shift = ss_input_dict.get("pointmap_shift")

        # 1) Frozen self-prediction: x1 for every modality (shape + rotation context) + canon extents.
        #    Seed the shape sample so canon is DETERMINISTIC per item (matches eval's canon -> the
        #    canon cancels exactly). RNG state restored after so the FM forward's random tau is intact.
        if shape_seed is not None:
            _rng, _crng = torch.get_rng_state(), torch.cuda.get_rng_state()
            torch.manual_seed(shape_seed)
            torch.cuda.manual_seed_all(shape_seed)
        with torch.no_grad():
            ss_ret = pipeline.sample_sparse_structure(
                ss_input_dict, inference_steps=shape_steps,
                use_distillation=False, with_grad=False,
            )
        if shape_seed is not None:
            torch.set_rng_state(_rng)
            torch.cuda.set_rng_state(_crng)
        x1 = {k: ss_ret[k].detach() for k in SS_FM_MODALITIES}

        # 2) Override scale + translation with GT (SSI). instance_scale_l2c = dims/(dsf*canon_ext),
        #    the inverse of predict_pose_metric's metric_dims = scale_per_axis * dsf * canon_ext.
        dsf = ss_ret.get("downsample_factor", 1.0)
        canon_ext = _voxel_canonical_extents(ss_ret)                    # [1,3] stopgrad
        gt_scale_l2c = (gt_dims.to(pipeline.device).reshape(1, 3) /
                        (dsf * canon_ext)).clamp_min(1e-6)
        ss = scene_scale.reshape(1, 3) if scene_scale.numel() == 3 else scene_scale.reshape(1, -1)[:, :3]
        sh = scene_shift.reshape(1, 3) if scene_shift.numel() == 3 else scene_shift.reshape(1, -1)[:, :3]
        gt_layout = encode_gt_layout_x1(
            gt_scale_l2c, gt_translation.to(pipeline.device).reshape(1, 3), ss, sh, pipeline.device,
        )
        for k in ("scale", "translation", "translation_scale"):
            x1[k] = gt_layout[k].reshape(1, 1, -1).to(x1[k])

        # 3) Conditioning + native FM loss (random tau, ONE forward WITH grad -> layout transformer).
        cond_args, cond_kwargs = pipeline.get_condition_input(
            pipeline.condition_embedders["ss_condition_embedder"],
            ss_input_dict, pipeline.ss_condition_input_mapping,
        )
        prev_w = ss_generator.loss_weights
        prev_g = ss_generator.random_generator
        ss_generator.loss_weights = loss_weights
        # The seeded CPU random_generator clashes with the cuda-default device context
        # (`with pipeline.device`): torch.randn(generator=cpu_gen) under a cuda default errors.
        # None -> default (cuda) generator; training wants random tau anyway.
        ss_generator.random_generator = None
        try:
            with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
                total_loss, _ = FlowMatching.loss(ss_generator, x1, *cond_args, **cond_kwargs)
        finally:
            ss_generator.loss_weights = prev_w
            ss_generator.random_generator = prev_g
        return total_loss


def encode_metric_scale_features(
    pipeline,
    item: dict,
    metric_dims: torch.Tensor,
    stage1_steps: int | None,
    stage2_steps: int | None,
    pointmap_store: Moge2PointmapStore | None = None,
) -> dict:
    pointmap = (
        pointmap_store.lookup(item["image_name"]) if pointmap_store is not None else None
    )
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(item["image"], pointmap=pointmap)
        ss_input_dict = pipeline.preprocess_image(
            item["image"], pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input_dict = pipeline.preprocess_image(item["image"], pipeline.slat_preprocessor)

        with torch.no_grad():
            ss_return_dict = pipeline.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_steps,
                use_distillation=False,
            )
            slat = pipeline.sample_slat(
                slat_input_dict,
                ss_return_dict["coords"],
                inference_steps=stage2_steps,
                use_distillation=False,
            )

    pointmap_scale = ss_input_dict.get("pointmap_scale")
    pointmap_shift = ss_input_dict.get("pointmap_shift")
    ss_scale_features = extract_ss_scale_features(ss_return_dict)
    return {
        "shape": ss_return_dict["shape"].detach().cpu(),
        "ss_scale_features": (
            None if ss_scale_features is None else ss_scale_features.detach().cpu()
        ),
        "pointmap_scale": None if pointmap_scale is None else pointmap_scale.detach().cpu(),
        "pointmap_shift": None if pointmap_shift is None else pointmap_shift.detach().cpu(),
        "slat_feats": slat.feats.detach().cpu(),
        "batch_indices": slat.coords[:, 0].detach().cpu(),
        "metric_dims": metric_dims.detach().cpu(),
        "uid": item.get("uid"),
        "category": item.get("category"),
        "source": item.get("source"),
        "image_name": item.get("image_name"),
    }


def predict_cached_log_dims(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder,
    features: dict,
    anchors: dict | None = None,
    return_aux: bool = False,
) -> torch.Tensor:
    shape = features["shape"].to(pipeline.device)
    pointmap_scale = features["pointmap_scale"]
    pointmap_shift = features["pointmap_shift"]
    if pointmap_scale is not None:
        pointmap_scale = pointmap_scale.to(pipeline.device)
    if pointmap_shift is not None:
        pointmap_shift = pointmap_shift.to(pipeline.device)

    # Older caches (pre-2026-04-30) lack ss_scale_features — pass None and let
    # MetricScaleHead zero-fill so this script can still consume them.
    ss_scale_features = features.get("ss_scale_features")
    if ss_scale_features is not None:
        ss_scale_features = ss_scale_features.to(pipeline.device)

    scale_token = scale_head(shape, ss_scale_features, pointmap_scale, pointmap_shift)
    if isinstance(scale_decoder, BinnedScaleDecoder):
        return scale_decoder(
            features["slat_feats"].to(pipeline.device),
            scale_token,
            features["batch_indices"].to(pipeline.device),
            anchor_feats_for(features, anchors, pipeline.device),
            return_aux=return_aux,
        )
    if anchors is not None:
        return scale_decoder(
            features["slat_feats"].to(pipeline.device),
            scale_token,
            features["batch_indices"].to(pipeline.device),
            anchor_feats_for(features, anchors, pipeline.device),
        )
    return scale_decoder(
        features["slat_feats"].to(pipeline.device),
        scale_token,
        features["batch_indices"].to(pipeline.device),
    )


def evaluate_cached_features(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder,
    feature_cache: list[dict],
    anchors: dict | None = None,
) -> dict:
    losses = []
    rel_errors = []
    abs_errors_cm = []
    category_errors = defaultdict(list)
    category_errors_cm = defaultdict(list)
    source_errors = defaultdict(list)
    source_errors_cm = defaultdict(list)
    scale_head.eval()
    scale_decoder.eval()
    with torch.no_grad():
        for features in feature_cache:
            log_pred = predict_cached_log_dims(
                pipeline, scale_head, scale_decoder, features, anchors=anchors
            )[0]
            target = features["metric_dims"].to(pipeline.device)
            log_target = torch.log(target.clamp(min=1e-6))
            pred = torch.exp(log_pred)
            rel = (pred - target).abs() / target.clamp(min=1e-6)
            abs_cm = (pred - target).abs() * 100.0
            losses.append(F.smooth_l1_loss(log_pred, log_target).detach().cpu())
            rel_cpu = rel.detach().cpu()
            abs_cm_cpu = abs_cm.detach().cpu()
            rel_errors.append(rel_cpu)
            abs_errors_cm.append(abs_cm_cpu)
            category_errors[features.get("category", "unknown")].append(rel_cpu.mean())
            category_errors_cm[features.get("category", "unknown")].append(abs_cm_cpu.mean())
            source_errors[features.get("source", "unknown")].append(rel_cpu.mean())
            source_errors_cm[features.get("source", "unknown")].append(abs_cm_cpu.mean())
    scale_head.train()
    scale_decoder.train()
    rel_tensor = torch.stack(rel_errors)
    abs_cm_tensor = torch.stack(abs_errors_cm)
    return {
        "loss": float(torch.stack(losses).mean()),
        "mean_abs_pct": float(rel_tensor.mean()) * 100,
        "median_abs_pct": float(rel_tensor.median()) * 100,
        "axis_mean_abs_pct": [float(v) * 100 for v in rel_tensor.mean(dim=0)],
        "mean_abs_cm": float(abs_cm_tensor.mean()),
        "median_abs_cm": float(abs_cm_tensor.median()),
        "axis_mean_abs_cm": [float(v) for v in abs_cm_tensor.mean(dim=0)],
        "category_mean_abs_pct": {
            category: float(torch.stack(values).mean()) * 100
            for category, values in sorted(category_errors.items())
        },
        "category_mean_abs_cm": {
            category: float(torch.stack(values).mean())
            for category, values in sorted(category_errors_cm.items())
        },
        "source_mean_abs_pct": {
            source: float(torch.stack(values).mean()) * 100
            for source, values in sorted(source_errors.items())
        },
        "source_mean_abs_cm": {
            source: float(torch.stack(values).mean())
            for source, values in sorted(source_errors_cm.items())
        },
    }


def evaluate_live_dataset(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    dataset,
    min_mask_pixels: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    inject_scale_token: bool,
    desc: str,
    max_samples: int | None = None,
    pointmap_store: Moge2PointmapStore | None = None,
    unfreeze_ss_backbone: bool = False,
    native_fm: bool = False,
) -> dict | None:
    """
    Evaluate with a live SS/SLAT forward pass so metrics reflect the current
    SLAT cross-attention weights, not stale cached latents.

    max_samples: when set (>0) and smaller than the dataset, evaluate a strided
    subset of roughly this many examples. Striding (rather than taking the first
    N) keeps the subset balanced across the source-ordered held-out set, so the
    cheap intra-epoch eval is not dominated by a single source.
    """
    if dataset is None or len(dataset) == 0:
        return None

    eval_stride = 1
    if max_samples and max_samples > 0 and len(dataset) > max_samples:
        eval_stride = (len(dataset) + max_samples - 1) // max_samples

    losses = []
    rel_errors = []
    abs_errors_cm = []
    category_errors = defaultdict(list)
    category_errors_cm = defaultdict(list)
    source_errors = defaultdict(list)
    source_errors_cm = defaultdict(list)

    scale_head_was_training = scale_head.training
    scale_decoder_was_training = scale_decoder.training
    scale_head.eval()
    scale_decoder.eval()
    slat_backbone = (
        pipeline._get_slat_backbone()
        if hasattr(pipeline, "_get_slat_backbone")
        else None
    )
    checkpoint_modes = None
    if slat_backbone is not None:
        checkpoint_modes = [block.use_checkpoint for block in slat_backbone.blocks]
        for block in slat_backbone.blocks:
            block.use_checkpoint = False

    eval_skipped = 0
    try:
        with torch.no_grad():
            for _eval_idx, item in enumerate(tqdm(dataset, desc=desc)):
                if eval_stride > 1 and (_eval_idx % eval_stride) != 0:
                    continue
                if int(item["mask_pixels"]) < min_mask_pixels:
                    continue
                try:
                    if unfreeze_ss_backbone:
                        # Joint-MoT: eval the trained pose path (W,H,D from MoT), not the head.
                        _pm = predict_pose_metric(
                            pipeline, item["image"], stage1_steps,
                            pointmap=(
                                pointmap_store.lookup(item["image_name"])
                                if pointmap_store is not None else None
                            ),
                            with_grad=False,
                            # native-FM: seed eval shape to match the training canon (exact cancel).
                            shape_seed=(shape_sample_seed(item["image_name"])
                                        if native_fm else None),
                        )
                        log_pred = _pm["log_dims"]
                    else:
                        log_pred, _, _ratio, _ = predict_log_dims(
                            pipeline,
                            scale_head,
                            scale_decoder,
                            item["image"],
                            stage1_steps,
                            stage2_steps,
                            inject_scale_token=inject_scale_token,
                            unfreeze_cross_attn=False,
                            p_uncond_scale_token=0.0,
                            pointmap=(
                                pointmap_store.lookup(item["image_name"])
                                if pointmap_store is not None
                                else None
                            ),
                        )
                except (IndexError, RuntimeError) as pipe_err:
                    msg = str(pipe_err)
                    if "out of memory" in msg.lower():
                        raise
                    eval_skipped += 1
                    print(
                        f"[eval-skip] {type(pipe_err).__name__}: {msg[:200]} "
                        f"(total eval skips={eval_skipped})"
                    )
                    torch.cuda.empty_cache()
                    continue
                log_pred = log_pred[0]
                target = torch.as_tensor(
                    item["metric_dims"], dtype=torch.float32, device=pipeline.device
                )
                log_target = torch.log(target.clamp(min=1e-6))
                pred = torch.exp(log_pred)
                rel = (pred - target).abs() / target.clamp(min=1e-6)
                abs_cm = (pred - target).abs() * 100.0
                losses.append(F.smooth_l1_loss(log_pred, log_target).detach().cpu())
                rel_cpu = rel.detach().cpu()
                abs_cm_cpu = abs_cm.detach().cpu()
                rel_errors.append(rel_cpu)
                abs_errors_cm.append(abs_cm_cpu)
                category = item.get("category", "unknown")
                source = item.get("source", "unknown")
                category_errors[category].append(rel_cpu.mean())
                category_errors_cm[category].append(abs_cm_cpu.mean())
                source_errors[source].append(rel_cpu.mean())
                source_errors_cm[source].append(abs_cm_cpu.mean())
    finally:
        if checkpoint_modes is not None:
            for block, use_checkpoint in zip(slat_backbone.blocks, checkpoint_modes):
                block.use_checkpoint = use_checkpoint
        if scale_head_was_training:
            scale_head.train()
        if scale_decoder_was_training:
            scale_decoder.train()

    if not losses:
        return None

    rel_tensor = torch.stack(rel_errors)
    abs_cm_tensor = torch.stack(abs_errors_cm)
    return {
        "loss": float(torch.stack(losses).mean()),
        "mean_abs_pct": float(rel_tensor.mean()) * 100,
        "median_abs_pct": float(rel_tensor.median()) * 100,
        "axis_mean_abs_pct": [float(v) * 100 for v in rel_tensor.mean(dim=0)],
        "mean_abs_cm": float(abs_cm_tensor.mean()),
        "median_abs_cm": float(abs_cm_tensor.median()),
        "axis_mean_abs_cm": [float(v) for v in abs_cm_tensor.mean(dim=0)],
        "category_mean_abs_pct": {
            category: float(torch.stack(values).mean()) * 100
            for category, values in sorted(category_errors.items())
        },
        "category_mean_abs_cm": {
            category: float(torch.stack(values).mean())
            for category, values in sorted(category_errors_cm.items())
        },
        "source_mean_abs_pct": {
            source: float(torch.stack(values).mean()) * 100
            for source, values in sorted(source_errors.items())
        },
        "source_mean_abs_cm": {
            source: float(torch.stack(values).mean())
            for source, values in sorted(source_errors_cm.items())
        },
    }


def format_metrics(prefix: str, epoch: int | None, metrics: dict) -> str:
    epoch_part = "" if epoch is None else f" epoch={epoch}"
    axis_pct = ",".join(f"{value:.2f}" for value in metrics["axis_mean_abs_pct"])
    axis_cm = ",".join(f"{value:.2f}" for value in metrics["axis_mean_abs_cm"])
    return (
        f"{prefix}{epoch_part} "
        f"loss={metrics['loss']:.6f} "
        f"mean_abs_pct={metrics['mean_abs_pct']:.2f} "
        f"median_abs_pct={metrics['median_abs_pct']:.2f} "
        f"axis_mean_abs_pct=[{axis_pct}] "
        f"mean_abs_cm={metrics['mean_abs_cm']:.2f} "
        f"median_abs_cm={metrics['median_abs_cm']:.2f} "
        f"axis_mean_abs_cm=[{axis_cm}]"
    )


def print_category_metrics(prefix: str, metrics: dict) -> None:
    category_metrics = metrics.get("category_mean_abs_pct", {})
    if not category_metrics:
        return
    rendered = ", ".join(
        f"{category}={value:.2f}%" for category, value in category_metrics.items()
    )
    print(f"{prefix}_by_category {rendered}")
    category_metrics_cm = metrics.get("category_mean_abs_cm", {})
    if category_metrics_cm:
        rendered_cm = ", ".join(
            f"{category}={value:.2f}cm" for category, value in category_metrics_cm.items()
        )
        print(f"{prefix}_by_category_cm {rendered_cm}")
    source_metrics = metrics.get("source_mean_abs_pct", {})
    if source_metrics:
        rendered_source = ", ".join(
            f"{source}={value:.2f}%" for source, value in source_metrics.items()
        )
        print(f"{prefix}_by_source {rendered_source}")
    source_metrics_cm = metrics.get("source_mean_abs_cm", {})
    if source_metrics_cm:
        rendered_source_cm = ", ".join(
            f"{source}={value:.2f}cm" for source, value in source_metrics_cm.items()
        )
        print(f"{prefix}_by_source_cm {rendered_source_cm}")


def write_metrics(path: str | None, split: str, epoch: int | None, metrics: dict) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "epoch": epoch, **metrics}
    with open(output_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def file_metadata(path: str | None, hash_file: bool = False) -> dict | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    metadata = {
        "path": str(file_path),
        "exists": True,
        "size_bytes": file_path.stat().st_size,
    }
    if hash_file:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata["sha256"] = digest.hexdigest()
    return metadata


def category_counts(feature_cache: list[dict]) -> dict:
    counts = defaultdict(int)
    for features in feature_cache:
        counts[features.get("category", "unknown")] += 1
    return dict(sorted(counts.items()))


def scene_counts(feature_cache: list[dict]) -> dict:
    counts = defaultdict(int)
    for features in feature_cache:
        image_name = features.get("image_name")
        if image_name is None:
            counts["unknown"] += 1
        else:
            counts[Path(image_name).parts[-2]] += 1
    return dict(sorted(counts.items()))


def source_counts(feature_cache: list[dict]) -> dict:
    counts = defaultdict(int)
    for features in feature_cache:
        counts[features.get("source", "unknown")] += 1
    return dict(sorted(counts.items()))


def feature_schema(feature_cache: list[dict]) -> dict:
    if not feature_cache:
        return {}
    sample = feature_cache[0]
    schema = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            schema[key] = {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            schema[key] = {"type": type(value).__name__}
    return schema


def build_manifest(
    args,
    train_cache: list[dict],
    heldout_cache: list[dict],
    checkpoint_path: str | None = None,
    best_checkpoint_path: str | None = None,
    best_metrics: dict | None = None,
    cache_path: str | None = None,
) -> dict:
    return {
        "manifest_version": 1,
        "script": "sam3d_objects/training/finetune_metric_scale.py",
        "args": vars(args),
        "dataset": {
            "dataset": args.dataset,
            "omninocs_root": args.omninocs_root,
            "omninocs_sources": args.omninocs_sources,
            "annotations_root": args.annotations_root,
            "rgb_root": args.rgb_root,
            "objectron_rgb_root": args.objectron_rgb_root,
            "arkitscenes_rgb_root": args.arkitscenes_rgb_root,
            "hypersim_rgb_root": args.hypersim_rgb_root,
            "skip_missing_rgb": args.skip_missing_rgb,
            "split": args.split,
            "categories": args.categories,
            "max_records": args.max_records,
            "max_records_per_source": args.max_records_per_source,
            "min_mask_pixels": args.min_mask_pixels,
        },
        "split": {
            "split_group": args.split_group,
            "seed": args.seed,
            "shuffle_split": args.shuffle_split,
            "train_count": len(train_cache),
            "heldout_count": len(heldout_cache),
            "train_categories": category_counts(train_cache),
            "heldout_categories": category_counts(heldout_cache),
            "train_sources": source_counts(train_cache),
            "heldout_sources": source_counts(heldout_cache),
            "train_scenes": scene_counts(train_cache),
            "heldout_scenes": scene_counts(heldout_cache),
        },
        "feature_schema": {
            "train": feature_schema(train_cache),
            "heldout": feature_schema(heldout_cache),
        },
        "artifacts": {
            "feature_cache": file_metadata(cache_path, hash_file=False),
            "checkpoint": file_metadata(checkpoint_path, hash_file=True),
            "best_checkpoint": file_metadata(best_checkpoint_path, hash_file=True),
            "metrics_output": file_metadata(args.metrics_output, hash_file=True),
        },
        "best_metrics": best_metrics,
    }


def write_manifest(
    path: str | None,
    args,
    train_cache: list[dict],
    heldout_cache: list[dict],
    checkpoint_path: str | None = None,
    best_checkpoint_path: str | None = None,
    best_metrics: dict | None = None,
    cache_path: str | None = None,
) -> None:
    if path is None:
        return
    manifest = build_manifest(
        args,
        train_cache,
        heldout_cache,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        best_metrics=best_metrics,
        cache_path=cache_path,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote reproducibility manifest to {output_path}")


def validate_feature_cache_manifest(
    manifest_path: str | None,
    train_cache: list[dict],
    heldout_cache: list[dict],
) -> None:
    if manifest_path is None:
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    split = manifest.get("split", {})
    expected_train = split.get("train_count")
    expected_heldout = split.get("heldout_count")
    if expected_train is not None and expected_train != len(train_cache):
        raise RuntimeError(
            f"Manifest train_count={expected_train} but loaded cache has {len(train_cache)}."
        )
    if expected_heldout is not None and expected_heldout != len(heldout_cache):
        raise RuntimeError(
            f"Manifest heldout_count={expected_heldout} but loaded cache has {len(heldout_cache)}."
        )
    expected_schema = manifest.get("feature_schema", {}).get("train", {})
    actual_schema = feature_schema(train_cache)
    for key, expected in expected_schema.items():
        actual = actual_schema.get(key)
        if expected.get("type") == "tensor" and actual != expected:
            raise RuntimeError(
                f"Manifest schema mismatch for {key}: expected {expected}, got {actual}."
            )
    print(f"Validated feature cache against manifest {manifest_path}")


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    flattened = {}
    axis_names = ("width", "height", "depth")
    for key, value in metrics.items():
        metric_key = f"{prefix}/{key}"
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{metric_key}/{nested_key}"] = nested_value
        elif isinstance(value, list):
            names = axis_names if len(value) == len(axis_names) else range(len(value))
            for name, item in zip(names, value):
                flattened[f"{metric_key}/{name}"] = item
        else:
            flattened[metric_key] = value
    return flattened


def init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging was requested with --wandb, but wandb is not "
            "installed in the active environment. Install it or rerun without --wandb."
        ) from exc

    tags = args.wandb_tags.split(",") if args.wandb_tags else None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags,
        mode=args.wandb_mode,
        config=vars(args),
    )


def wandb_log(wandb_run, payload: dict, step: int | None = None) -> None:
    if wandb_run is None:
        return
    wandb_run.log(payload, step=step)


def save_metric_checkpoint(
    path: str,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    args,
    epoch: int | None = None,
    metrics: dict | None = None,
    slat_backbone=None,
    ss_decoder=None,
    ss_backbone=None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_scale_head": scale_head.state_dict(),
        "metric_scale_decoder": scale_decoder.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "metrics": metrics,
    }
    if ss_backbone is not None:
        # Joint-MoT: the unfrozen SS generator backbone (pose/scale heads + geometry) IS the
        # trained model — must persist or the checkpoint is inference-invalid ([[checkpoint
        # completeness]]). Full backbone state (the cross_attn fp32 upcast is included).
        payload["ss_backbone"] = ss_backbone.state_dict()
    if slat_backbone is not None:
        # Save only the cross_attn + norm2 params that were actually trained.
        cross_attn_state = {}
        for i, block in enumerate(slat_backbone.blocks):
            for k, v in block.cross_attn.state_dict().items():
                cross_attn_state[f"blocks.{i}.cross_attn.{k}"] = v
            for k, v in block.norm2.state_dict().items():
                cross_attn_state[f"blocks.{i}.norm2.{k}"] = v
        payload["slat_cross_attn"] = cross_attn_state
    if ss_decoder is not None:
        payload["ss_decoder"] = ss_decoder.state_dict()
    torch.save(payload, output_path)


def category_mean_baseline(train_cache: list[dict], eval_cache: list[dict]) -> dict | None:
    category_dims = defaultdict(list)
    for features in train_cache:
        category_dims[features.get("category", "unknown")].append(features["metric_dims"])
    if not category_dims or not eval_cache:
        return None

    category_means = {
        category: torch.stack(values).mean(dim=0)
        for category, values in category_dims.items()
    }
    global_mean = torch.stack(
        [features["metric_dims"] for features in train_cache]
    ).mean(dim=0)

    rel_errors = []
    abs_errors_cm = []
    category_errors = defaultdict(list)
    category_errors_cm = defaultdict(list)
    source_errors = defaultdict(list)
    source_errors_cm = defaultdict(list)
    for features in eval_cache:
        category = features.get("category", "unknown")
        source = features.get("source", "unknown")
        pred = category_means.get(category, global_mean)
        target = features["metric_dims"]
        rel = (pred - target).abs() / target.clamp(min=1e-6)
        abs_cm = (pred - target).abs() * 100.0
        rel_errors.append(rel)
        abs_errors_cm.append(abs_cm)
        category_errors[category].append(rel.mean())
        category_errors_cm[category].append(abs_cm.mean())
        source_errors[source].append(rel.mean())
        source_errors_cm[source].append(abs_cm.mean())

    rel_tensor = torch.stack(rel_errors)
    abs_cm_tensor = torch.stack(abs_errors_cm)
    return {
        "loss": float("nan"),
        "mean_abs_pct": float(rel_tensor.mean()) * 100,
        "median_abs_pct": float(rel_tensor.median()) * 100,
        "axis_mean_abs_pct": [float(v) * 100 for v in rel_tensor.mean(dim=0)],
        "mean_abs_cm": float(abs_cm_tensor.mean()),
        "median_abs_cm": float(abs_cm_tensor.median()),
        "axis_mean_abs_cm": [float(v) for v in abs_cm_tensor.mean(dim=0)],
        "category_mean_abs_pct": {
            category: float(torch.stack(values).mean()) * 100
            for category, values in sorted(category_errors.items())
        },
        "category_mean_abs_cm": {
            category: float(torch.stack(values).mean())
            for category, values in sorted(category_errors_cm.items())
        },
        "source_mean_abs_pct": {
            source: float(torch.stack(values).mean()) * 100
            for source, values in sorted(source_errors.items())
        },
        "source_mean_abs_cm": {
            source: float(torch.stack(values).mean())
            for source, values in sorted(source_errors_cm.items())
        },
    }


def split_group_key(record: dict, split_group: str) -> str:
    if split_group == "record":
        return record["uid"]
    image_name = record["image_name"]
    if split_group == "image":
        return image_name
    if split_group == "scene":
        return Path(image_name).parts[-2]
    raise ValueError(f"Unsupported split group: {split_group}")


def make_train_eval_subsets(
    dataset,
    train_samples: int | None,
    heldout_samples: int,
    seed: int,
    shuffle_split: bool,
    split_group: str,
    heldout_per_source: dict[str, int] | None = None,
) -> tuple[Subset, Subset | None]:
    dataset_len = len(dataset)
    if heldout_per_source is not None:
        if not hasattr(dataset, "records"):
            raise ValueError(
                "--heldout-per-source requires a dataset that exposes "
                "`records` (OmniNOCSObjectDataset). Got "
                f"{type(dataset).__name__}."
            )
        source_indices: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(dataset.records):
            source_indices[record["source"]].append(idx)
        train_indices: list[int] = []
        eval_indices: list[int] = []
        for source, indices in source_indices.items():
            n_heldout = heldout_per_source.get(source, 0)
            if n_heldout > len(indices):
                print(
                    f"Warning: requested {n_heldout} heldout records for "
                    f"source={source} but only {len(indices)} available; "
                    f"using all but 1 as heldout."
                )
                n_heldout = max(len(indices) - 1, 0)
            split_point = len(indices) - n_heldout
            train_indices.extend(indices[:split_point])
            eval_indices.extend(indices[split_point:])
        train_subset = Subset(dataset, train_indices)
        eval_subset = Subset(dataset, eval_indices) if eval_indices else None
        per_source_report = ", ".join(
            f"{src}: train={len([i for i in train_indices if dataset.records[i]['source'] == src])}"
            f" / heldout={len([i for i in eval_indices if dataset.records[i]['source'] == src])}"
            for src in sorted(source_indices)
        )
        print(
            f"Split (per-source) train={len(train_subset)} "
            f"heldout={0 if eval_subset is None else len(eval_subset)} "
            f"[{per_source_report}]"
        )
        return train_subset, eval_subset
    if split_group == "record":
        indices = list(range(dataset_len))
        if shuffle_split:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(dataset_len, generator=generator).tolist()
        eval_count = max(heldout_samples, 0)
        if train_samples is None or train_samples <= 0:
            train_count = max(dataset_len - eval_count, 0)
        else:
            train_count = min(train_samples, dataset_len)
        eval_start = train_count
        eval_stop = min(eval_start + eval_count, dataset_len)
        train_indices = indices[:train_count]
        eval_indices = indices[eval_start:eval_stop]
    else:
        grouped_indices = defaultdict(list)
        for idx, record in enumerate(dataset.records):
            grouped_indices[split_group_key(record, split_group)].append(idx)
        groups = sorted(grouped_indices)
        generator = torch.Generator().manual_seed(seed)
        if shuffle_split:
            permutation = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[idx] for idx in permutation]
        heldout_target = max(heldout_samples, 0)
        if train_samples is None or train_samples <= 0:
            train_target = max(dataset_len - heldout_target, 0)
        else:
            train_target = train_samples
        train_indices = []
        eval_indices = []
        for group in groups:
            group_indices = grouped_indices[group]
            if len(train_indices) < train_target:
                train_indices.extend(group_indices)
            elif len(eval_indices) < heldout_target:
                eval_indices.extend(group_indices)
            if len(train_indices) >= train_target and len(eval_indices) >= heldout_target:
                break

    train_subset = Subset(dataset, train_indices)
    eval_subset = None
    if heldout_samples:
        eval_subset = Subset(dataset, eval_indices)
        if len(eval_subset) < heldout_samples:
            print(
                f"Requested {heldout_samples} held-out samples but only "
                f"{len(eval_subset)} were available after the train split."
            )
    print(
        f"Split group={split_group} train={len(train_subset)} "
        f"heldout={0 if eval_subset is None else len(eval_subset)}"
    )
    return train_subset, eval_subset


def cache_metric_scale_features(
    pipeline,
    dataset,
    min_mask_pixels: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    desc: str,
    pointmap_store: Moge2PointmapStore | None = None,
    partial_path: str | None = None,
    partial_every: int = 250,
) -> list[dict]:
    feature_cache = []
    done_uids: set = set()
    if partial_path and Path(partial_path).exists():
        feature_cache = torch.load(partial_path, weights_only=False)
        done_uids = {f["uid"] for f in feature_cache}
        print(f"{desc}: resumed {len(feature_cache)} entries from {partial_path}")
    skipped_errors = 0
    for item in tqdm(dataset, desc=desc):
        if int(item["mask_pixels"]) < min_mask_pixels:
            continue
        if item["uid"] in done_uids:
            continue
        try:
            feature_cache.append(
                encode_metric_scale_features(
                    pipeline,
                    item,
                    torch.as_tensor(item["metric_dims"], dtype=torch.float32),
                    stage1_steps,
                    stage2_steps,
                    pointmap_store=pointmap_store,
                )
            )
        except KeyError:
            raise  # missing precomputed pointmap — abort rather than mix anchors
        except Exception as exc:
            skipped_errors += 1
            print(
                "Skipping feature-cache example due to pipeline error: "
                f"uid={item.get('uid')} source={item.get('source')} "
                f"image_name={item.get('image_name')} error={exc}"
            )
            continue
        if partial_path and len(feature_cache) % partial_every == 0:
            tmp = f"{partial_path}.tmp"
            torch.save(feature_cache, tmp)
            os.replace(tmp, partial_path)
    if skipped_errors:
        print(f"{desc}: skipped {skipped_errors} examples due to pipeline errors")
    return feature_cache


def save_feature_cache(path: str, train_cache: list[dict], heldout_cache: list[dict], args) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": train_cache,
            "heldout": heldout_cache,
            "args": vars(args),
        },
        cache_path,
    )
    print(
        f"Saved feature cache to {cache_path} "
        f"(train={len(train_cache)}, heldout={len(heldout_cache)})"
    )
    write_manifest(
        args.manifest_output,
        args,
        train_cache,
        heldout_cache,
        cache_path=str(cache_path),
    )


def load_feature_cache(path: str, require_train: bool = True) -> tuple[list[dict], list[dict], dict]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    train_cache = cache.get("train", [])
    heldout_cache = cache.get("heldout", [])
    if require_train and not train_cache:
        raise RuntimeError(f"No train features found in cache: {path}")
    if not train_cache and not heldout_cache:
        raise RuntimeError(f"No features found in cache: {path}")
    print(
        f"Loaded feature cache from {path} "
        f"(train={len(train_cache)}, heldout={len(heldout_cache)})"
    )
    return train_cache, heldout_cache, cache.get("args", {})


def _adapt_scale_head_state_dict(ckpt_sd: dict, head: MetricScaleHead) -> dict:
    """
    Handle shape mismatches when loading MetricScaleHead from an older checkpoint.
    Specifically: if the checkpoint was trained without ss_scale_features (10-dim
    input) and the current model has ss_scale_features (13-dim input), we insert
    zero columns for the new ss_scale dims and shift the pointmap columns.
    Other size mismatches (e.g. ctx_channels change) are skipped via strict=False.
    """
    model_sd = head.state_dict()
    result = {}
    for key, ckpt_val in ckpt_sd.items():
        if key not in model_sd:
            continue
        model_val = model_sd[key]
        if ckpt_val.shape == model_val.shape:
            result[key] = ckpt_val
        elif key == "mlp.0.weight" and model_val.shape[1] - ckpt_val.shape[1] == head.ss_scale_dim:
            # ss_scale dims were inserted between pooled and pointmap columns.
            # ckpt layout: [pooled | log_ps | shift_z]
            # model layout: [pooled | ss_scale | log_ps | shift_z]
            new_w = torch.zeros_like(model_val)
            insert_at = ckpt_val.shape[1] - 2  # pointmap is always last 2 cols
            new_w[:, :insert_at] = ckpt_val[:, :insert_at]
            new_w[:, insert_at + head.ss_scale_dim:] = ckpt_val[:, insert_at:]
            result[key] = new_w
            print(f"  scale_head: partial load for {key}: "
                  f"ckpt {ckpt_val.shape} → model {model_val.shape} "
                  f"(zeroed ss_scale cols {insert_at}:{insert_at + head.ss_scale_dim})")
        # else: size mismatch we don't know how to handle — skip (strict=False)
    return result


def load_metric_checkpoint(
    path: str,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    slat_backbone=None,
    ss_decoder=None,
    ss_backbone=None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # strict=False: old checkpoints may have ctx_channels=768 or 10-dim input;
    # _adapt_scale_head_state_dict handles the input-dim expansion case cleanly.
    adapted_head_sd = _adapt_scale_head_state_dict(checkpoint["metric_scale_head"], scale_head)
    scale_head.load_state_dict(adapted_head_sd, strict=False)
    scale_decoder.load_state_dict(checkpoint["metric_scale_decoder"], strict=False)
    if slat_backbone is not None and "slat_cross_attn" in checkpoint:
        slat_state = checkpoint["slat_cross_attn"]
        for i, block in enumerate(slat_backbone.blocks):
            block_ca = {k.split(f"blocks.{i}.cross_attn.", 1)[1]: v
                        for k, v in slat_state.items()
                        if k.startswith(f"blocks.{i}.cross_attn.")}
            block_n2 = {k.split(f"blocks.{i}.norm2.", 1)[1]: v
                        for k, v in slat_state.items()
                        if k.startswith(f"blocks.{i}.norm2.")}
            if block_ca:
                block.cross_attn.load_state_dict(block_ca, strict=True)
            if block_n2:
                block.norm2.load_state_dict(block_n2, strict=True)
        print(f"Restored SLAT cross-attn weights from {path}")
    if ss_decoder is not None and "ss_decoder" in checkpoint:
        ss_decoder.load_state_dict(checkpoint["ss_decoder"], strict=True)
        print(f"Restored SS decoder weights from {path}")
    if ss_backbone is not None and "ss_backbone" in checkpoint:
        ss_backbone.load_state_dict(checkpoint["ss_backbone"], strict=False)
        print(f"Restored SS generator backbone (joint-MoT) weights from {path}")
    print(f"Loaded metric scale checkpoint from {path}")
    return checkpoint


def evaluate_and_record(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder,
    feature_cache: list[dict],
    heldout_cache: list[dict],
    metrics_output: str | None,
    wandb_run=None,
    epoch: int | None = None,
    anchors: dict | None = None,
) -> dict:
    if feature_cache:
        train_metrics = evaluate_cached_features(
            pipeline, scale_head, scale_decoder, feature_cache, anchors=anchors
        )
        print(format_metrics("train_eval", epoch, train_metrics))
        print_category_metrics("train_eval", train_metrics)
        write_metrics(metrics_output, "train", epoch, train_metrics)
        wandb_payload = flatten_metrics("train_eval", train_metrics)
    else:
        train_metrics = None
        wandb_payload = {}
    if heldout_cache:
        heldout_metrics = evaluate_cached_features(
            pipeline, scale_head, scale_decoder, heldout_cache, anchors=anchors
        )
        print(format_metrics("heldout_eval", epoch, heldout_metrics))
        print_category_metrics("heldout_eval", heldout_metrics)
        write_metrics(metrics_output, "heldout", epoch, heldout_metrics)
        wandb_payload.update(flatten_metrics("heldout_eval", heldout_metrics))
    else:
        heldout_metrics = None
    if epoch is not None:
        wandb_payload["epoch"] = epoch
    wandb_log(wandb_run, wandb_payload)
    return {"train": train_metrics, "heldout": heldout_metrics}


def default_best_output_path(output: str) -> str:
    output_path = Path(output)
    return str(output_path.with_name(f"{output_path.stem}_best{output_path.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument(
        "--dataset",
        choices=["omninocs-nocs-real275", "omninocs-mixed"],
        default="omninocs-nocs-real275",
        help="Dataset loader to use for metric-scale training.",
    )
    parser.add_argument("--omninocs-root", default="/mnt/dest/OmniNOCS")
    parser.add_argument(
        "--omninocs-sources",
        nargs="+",
        default=["nocs_real275"],
        choices=["nocs_real275", "objectron", "arkitscenes", "hypersim"],
        help="Sources to include with --dataset omninocs-mixed.",
    )
    parser.add_argument(
        "--annotations-root",
        default="/mnt/dest/OmniNOCS/omninocs_release_nocs_real275",
    )
    parser.add_argument("--rgb-root", default="/mnt/dest/OmniNOCS/real_test")
    parser.add_argument("--objectron-rgb-root", default=None)
    parser.add_argument("--arkitscenes-rgb-root", default=None)
    parser.add_argument("--hypersim-rgb-root", default=None)
    parser.add_argument(
        "--skip-missing-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For mixed OmniNOCS, skip records whose source RGB frame is unavailable.",
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--size-outlier-log-tol", type=float, default=DEFAULT_SIZE_OUTLIER_LOG_TOL,
        help="Drop GT boxes whose max-dim deviates from their category median "
             "log-size by more than this (default log(5)≈1.61 => >5x/<1/5x). "
             "Removes gross annotation errors (e.g. 41m Objectron cups). "
             "Set <0 to disable.")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional per-source cap for --dataset omninocs-mixed.",
    )
    parser.add_argument("--overfit-samples", type=int, default=10)
    parser.add_argument(
        "--heldout-samples",
        type=int,
        default=0,
        help=(
            "Reserve this many examples after the train split for held-out eval. "
            "In live training, these examples are evaluated with live SS/SLAT "
            "forward passes so current SLAT cross-attn weights are measured."
        ),
    )
    parser.add_argument(
        "--heldout-per-source",
        default=None,
        help=(
            "Per-source heldout counts as a comma-separated source:count list, e.g. "
            "'nocs_real275:64,objectron:200,arkitscenes:200'. When set, overrides "
            "--heldout-samples; each source carves the specified number of records "
            "off the tail of its own records (record-order split). Requires "
            "--dataset omninocs-mixed."
        ),
    )
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a WeightedRandomSampler with weight = 1/source_count so each batch "
            "is balanced 1:1:... across the OmniNOCS sources, regardless of raw "
            "source size. Combine with --max-records-per-source to cap dataset size "
            "before sampling. Requires --dataset omninocs-mixed."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle-split",
        action="store_true",
        help="Use a deterministic random train/held-out split instead of record order.",
    )
    parser.add_argument(
        "--split-group",
        choices=["record", "image", "scene"],
        default="record",
        help="Keep grouped records together when forming train/held-out splits.",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument(
        "--anchor-tables", default=None,
        help="Comma-separated jsonl paths from pose_scale_heldout_compare.py "
             "(pose-decoder scale + voxel extents per uid). Required for "
             "--decoder anchor/factored; cache entries without an anchor are dropped.")
    parser.add_argument(
        "--decoder", choices=["baseline", "anchor", "factored", "binned"], default="baseline",
        help="baseline: stock MetricScaleDecoder. anchor (v1a): + anchor input "
             "features, absolute log-WHD + aux iso loss, shared gradients. "
             "factored (v1b): MoGe-2-style decoupled iso/proportions branches. "
             "binned (v2): OmniNOCS-style softmax-CE over iso-scale bins (residual "
             "on the anchor iso) + GT-normalized L1 + max-pinned proportions.")
    parser.add_argument(
        "--scale-loss-weight", type=float, default=0.0,
        help="Weight on the iso-scale loss: v1a aux (log max(pred) vs log max(gt))^2, "
             "v1b exclusive iso loss, or v2 GT-normalized L1 |s-s_gt|/s_gt.")
    parser.add_argument(
        "--prop-loss-weight", type=float, default=1.0,
        help="Weight on the max-pinned log-proportions loss (factored/binned decoder).")
    parser.add_argument(
        "--bin-ce-weight", type=float, default=1.0,
        help="Weight on the softmax cross-entropy over iso-scale bins (binned decoder).")
    parser.add_argument(
        "--num-bins", type=int, default=128,
        help="Number of absolute iso-scale bins (binned decoder).")
    parser.add_argument(
        "--bin-log-min", type=float, default=-5.0,
        help="Lower edge of the absolute log-metre bin grid (binned decoder). "
             "-5.0 => ~0.67cm.")
    parser.add_argument(
        "--bin-log-max", type=float, default=2.0,
        help="Upper edge of the absolute log-metre bin grid (binned decoder). "
             "2.0 => ~7.4m.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-steps", type=int, default=None)
    # --- Joint-MoT metric training (unfreeze SS generator; pose predicts W,H,D + translation) ---
    parser.add_argument("--unfreeze-ss-backbone", action="store_true",
                        help="Train the SS generator (MoT): pose decoder predicts metric W,H,D + "
                             "translation directly (drops MetricScaleHead/scale-token path). Live only.")
    parser.add_argument("--ss-geometry-lr", type=float, default=2e-6,
                        help="LR for the SS geometry prior (blocks + shape latent) — SLOW, protect it.")
    parser.add_argument("--ss-pose-lr", type=float, default=1e-4,
                        help="LR for the SS pose/scale modality heads — FAST (the metric signal).")
    parser.add_argument("--ss-cond-lr", type=float, default=1e-5,
                        help="LR for SS cross-attn conditioning (only if --unfreeze-ss-cross-attn).")
    parser.add_argument("--unfreeze-ss-cross-attn", action="store_true",
                        help="Also train the SS per-block cross-attn (fp32-upcast). Off by default.")
    parser.add_argument("--unfreeze-ss-layout-for-head", action="store_true",
                        help="Hybrid mode: keep the MetricScaleHead + SLAT-injection path (like "
                             "mixed_moge2_live_v1) but also train the SS layout transformer so "
                             "ss_scale_features carry a gradient back into the MoT layout blocks. "
                             "Shape transformer stays frozen via the MoT stop-grad k/v detach. "
                             "Mutually exclusive with --unfreeze-ss-backbone / --native-fm.")
    # Gradient checkpointing trades compute for GPU memory (recomputes each block's forward
    # during backward). Default on. Disable when GPU memory has headroom — removes the
    # recompute, faster backward, RESULT-NEUTRAL. The SS toggle matters most: the SS flow
    # runs --stage1-steps forwards, each recomputed when checkpointed.
    parser.add_argument("--ss-grad-checkpoint", dest="ss_grad_checkpoint",
                        action="store_true", default=True,
                        help="Gradient-checkpoint the SS transformer blocks (default).")
    parser.add_argument("--no-ss-grad-checkpoint", dest="ss_grad_checkpoint",
                        action="store_false",
                        help="Store SS activations instead of recomputing — faster backward when "
                             "GPU memory allows. Result-neutral.")
    parser.add_argument("--slat-grad-checkpoint", dest="slat_grad_checkpoint",
                        action="store_true", default=True,
                        help="Gradient-checkpoint the SLAT transformer blocks (default).")
    parser.add_argument("--no-slat-grad-checkpoint", dest="slat_grad_checkpoint",
                        action="store_false",
                        help="Store SLAT activations instead of recomputing. Result-neutral.")
    parser.add_argument("--trans-loss-weight", type=float, default=1.0,
                        help="Weight on the metric translation smooth-L1 loss (the mAP lever).")
    parser.add_argument("--gt-translation-flip-xy", action="store_true", default=True,
                        help="Negate GT translation x,y to map OpenCV->PyTorch3D (pose-decoder frame). "
                             "Default True (principled + matches step 0.5); first run validates via "
                             "L_trans decreasing. Use --no-gt-translation-flip-xy to disable.")
    parser.add_argument("--no-gt-translation-flip-xy", dest="gt_translation_flip_xy",
                        action="store_false")
    parser.add_argument("--flow-loss-weight", type=float, default=0.0,
                        help="Weight on the SS flow/geometry-anchor regularizer (TODO; 0 = rely on "
                             "the slow geometry LR as the interim safeguard).")
    parser.add_argument("--native-fm", action="store_true",
                        help="Native random-tau flow-matching layout SFT (paper L_CFM) instead of "
                             "predict_pose_metric sample-then-regress. Supervises scale+translation "
                             "modalities via FlowMatching.loss; shape+rotation x1 = self-prediction. "
                             "Requires --unfreeze-ss-backbone. Cheaper (1 grad forward, random tau).")
    parser.add_argument("--fm-shape-steps", type=int, default=4,
                        help="Frozen-shape sampling steps for the self-predicted shape/rotation x1 context.")
    parser.add_argument("--fm-scale-weight", type=float, default=1.0,
                        help="Native-FM loss weight on the scale (W,H,D) modality.")
    parser.add_argument("--fm-trans-weight", type=float, default=1.0,
                        help="Native-FM loss weight on the translation (dir + norm) modalities.")
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--cache-latents",
        action="store_true",
        help="Precompute frozen SAM3D features once, then overfit only the metric heads.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="When caching latents, print same-set eval metrics every N epochs.",
    )
    parser.add_argument("--save-feature-cache", default=None)
    parser.add_argument("--load-feature-cache", default=None)
    parser.add_argument(
        "--cache-heldout-only",
        action="store_true",
        help=(
            "With --cache-latents: skip the train split and cache only the held-out "
            "records (e.g. to re-encode the eval set at inference-grade flow steps). "
            "The split itself is unchanged — same dataset args + seed give the same "
            "held-out set."
        ),
    )
    parser.add_argument(
        "--moge2-pointmap-dir",
        default=None,
        help=(
            "Directory of precomputed MoGe-2 global metric pointmaps "
            "(scripts/precompute_moge2_pointmaps.py output). When set, every "
            "pipeline forward (train / live eval / latent caching) injects the "
            "cached pointmap via compute_pointmap(pointmap=...) instead of live "
            "MoGe-v1, making pointmap_scale/shift and SS conditioning metric. "
            "A frame missing from the manifest is a hard error."
        ),
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="Load a saved metric-head checkpoint before training or eval-only metrics.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Resume live training from a recovery checkpoint saved by --checkpoint-every. "
            "Loads metric-head + SLAT cross-attn weights and continues from the epoch "
            "stored in the checkpoint. Implies --load-checkpoint for the same path."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Save a rolling recovery checkpoint every N epochs during live training "
            "(non-cached path). Overwrites the same file each time to limit disk usage. "
            "Strongly recommended when --unfreeze-slat-cross-attn is set."
        ),
    )
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help=(
            "Run a live held-out eval (and update the best checkpoint) every N "
            "optimizer steps WITHIN an epoch, not just at epoch end. Gives fast "
            "feedback on long epochs and makes the best checkpoint reflect "
            "mid-epoch progress. 0 = end-of-epoch eval only."
        ),
    )
    parser.add_argument(
        "--eval-steps-max-samples",
        type=int,
        default=0,
        help=(
            "When --eval-every-steps fires, cap the intra-epoch eval to ~this many "
            "held-out samples (strided across the set to stay source-balanced) for "
            "speed. 0 = use the full held-out set. End-of-epoch eval is always full."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help=(
            "Save a rolling recovery checkpoint every N optimizer steps within an "
            "epoch (overwrites the same _resume file). Independent of "
            "--eval-every-steps; set this smaller for tighter restart safety so a "
            "mid-epoch crash loses minutes of weights, not a whole epoch. Resume is "
            "still epoch-granular (re-runs the epoch's earlier steps), but the "
            "trained weights survive. 0 = end-of-epoch only."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate a loaded checkpoint on cached train/held-out features, then exit.",
    )
    parser.add_argument(
        "--inject-scale-token-into-slat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In the non-cached training path, inject the metric scale token from "
            "stage-1 into SLAT stage-2 conditioning (both the conditional and "
            "unconditional CFG passes) before the SLAT sampler runs. This aligns "
            "training with inference behavior. Has no effect when --cache-latents "
            "or --load-feature-cache is used, since SLAT features are precomputed "
            "in those modes."
        ),
    )
    parser.add_argument(
        "--unfreeze-slat-cross-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Selectively unfreeze the cross-attention layers (cross_attn + norm2) "
            "in all 24 SLatFlowModel transformer blocks so the denoiser learns to "
            "use the metric scale token. Implies --inject-scale-token-into-slat. "
            "Incompatible with --cache-latents and --load-feature-cache."
        ),
    )
    parser.add_argument(
        "--slat-lr",
        type=float,
        default=1e-6,
        help=(
            "Learning rate for the SLAT cross-attention params when "
            "--unfreeze-slat-cross-attn is set. Should be lower than --lr to "
            "avoid catastrophic forgetting of the DINOv2 conditioning. "
            "Lowered from 1e-5 → 1e-6 after slat_conditioned_v1 NaN'd "
            "cross_attn weights on its first optimizer step."
        ),
    )
    parser.add_argument(
        "--slat-lr-warmup-steps",
        type=int,
        default=500,
        help=(
            "Linearly warm up the SLAT cross-attn group LR from 0 to "
            "--slat-lr over this many optimizer steps. The metric-head group "
            "is not warmed up. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--fp32-slat-cross-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run SLAT cross_attn + norm2 in fp32 with input/output dtype shims "
            "while keeping the rest of SLAT in bf16/fp16. Prevents bf16 "
            "backward overflow that NaN-corrupted cross_attn weights after "
            "the first optimizer step of slat_conditioned_v1."
        ),
    )
    parser.add_argument(
        "--ss-ratio-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the SS aspect-ratio loss added to the metric regression loss. "
            "Computes a scale-invariant loss on the soft voxel extent (SS decoder "
            "soft occupancy output) vs GT [W, H, D] dims. Requires "
            "--unfreeze-ss-decoder to have a gradient path. Start at 0.01–0.1."
        ),
    )
    parser.add_argument(
        "--unfreeze-ss-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Unfreeze the SS decoder (convolutional 16^3 → 64^3 network) so the "
            "aspect-ratio loss can update its weights. The SS backbone stays frozen. "
            "Only active in the non-cached training path."
        ),
    )
    parser.add_argument(
        "--max-consecutive-nan-skips",
        type=int,
        default=50,
        help=(
            "Abort live training if this many consecutive optimizer steps are "
            "skipped due to NaN/Inf gradients. The previous run silently spun "
            "for ~26h after a single NaN cascade because the loop kept "
            "running forward+backward on every NaN-skip without progress. "
            "Set 0 to disable the watchdog."
        ),
    )
    parser.add_argument(
        "--p-uncond-scale-token",
        type=float,
        default=0.0,
        help=(
            "Probability of zeroing the SLAT-injected scale token during training "
            "(CFG-style dropout from TRELLIS p_uncond=0.1). Active only in the "
            "live (non-cached) training path. The MetricScaleDecoder still sees "
            "the unzeroed token so the metric regression signal is preserved. "
            "Recommended 0.1 with --unfreeze-slat-cross-attn."
        ),
    )
    parser.add_argument(
        "--log-step-every",
        type=int,
        default=1,
        help=(
            "Log per-step training stability metrics (loss, total grad_norm, "
            "per-group grad_norm, adaptive clip threshold, NaN/OOM skips, "
            "scale-token dropout indicator) to wandb every N steps. 0 disables "
            "step-level logging (epoch-level logging is always on)."
        ),
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSONL path for baseline/train/held-out eval metrics.",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional JSON path recording dataset, split, feature schema, and artifacts.",
    )
    parser.add_argument(
        "--validate-manifest",
        default=None,
        help="Validate a loaded feature cache against a previous manifest JSON.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build and save feature cache, then exit before metric-head training.",
    )
    parser.add_argument("--output", default="checkpoints/metric_scale_overfit.pt")
    parser.add_argument(
        "--best-output",
        default=None,
        help=(
            "Optional path for the best checkpoint by held-out mean_abs_pct. "
            "Defaults to <output stem>_best<suffix> when held-out eval is available."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training, evaluation, baseline, and checkpoint metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="sam3d-metric-scale")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode override.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    if args.unfreeze_slat_cross_attn and (args.cache_latents or args.load_feature_cache):
        raise RuntimeError(
            "--unfreeze-slat-cross-attn is incompatible with --cache-latents and "
            "--load-feature-cache. SLAT must be re-run every step so the cross-attn "
            "layers see the current scale token."
        )

    if args.unfreeze_slat_cross_attn and not args.inject_scale_token_into_slat:
        print(
            "Note: --unfreeze-slat-cross-attn implies --inject-scale-token-into-slat; "
            "enabling injection automatically."
        )
        args.inject_scale_token_into_slat = True

    wandb_run = init_wandb(args)

    if args.load_feature_cache:
        feature_cache, heldout_cache, cache_args = load_feature_cache(
            args.load_feature_cache, require_train=not args.eval_only
        )
        validate_feature_cache_manifest(
            args.validate_manifest,
            feature_cache,
            heldout_cache,
        )
        pipeline = DeviceOnlyPipeline(args.device)
        train_dataset = None
        eval_dataset = None
        loader = None
        pointmap_store = None  # cached training runs no pipeline forwards
    else:
        dataset = build_dataset(args)

        heldout_per_source: dict[str, int] | None = None
        if args.heldout_per_source:
            if args.dataset != "omninocs-mixed":
                raise ValueError(
                    "--heldout-per-source requires --dataset omninocs-mixed."
                )
            heldout_per_source = {}
            for piece in args.heldout_per_source.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                if ":" not in piece:
                    raise ValueError(
                        f"--heldout-per-source entry {piece!r} must be 'source:count'."
                    )
                src, count_str = piece.split(":", 1)
                heldout_per_source[src.strip()] = int(count_str)

        train_dataset, eval_dataset = make_train_eval_subsets(
            dataset,
            args.overfit_samples,
            args.heldout_samples,
            args.seed,
            args.shuffle_split,
            args.split_group,
            heldout_per_source=heldout_per_source,
        )

        sampler = None
        shuffle = True
        if args.balanced_sampling:
            if args.dataset != "omninocs-mixed":
                raise ValueError(
                    "--balanced-sampling requires --dataset omninocs-mixed."
                )
            if not hasattr(dataset, "records"):
                raise ValueError(
                    "--balanced-sampling requires a dataset exposing `records`."
                )
            source_counts: dict[str, int] = defaultdict(int)
            for idx in train_dataset.indices:
                source_counts[dataset.records[idx]["source"]] += 1
            weights = [
                1.0 / source_counts[dataset.records[idx]["source"]]
                for idx in train_dataset.indices
            ]
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
            shuffle = False
            counts_report = ", ".join(
                f"{src}={cnt}" for src, cnt in sorted(source_counts.items())
            )
            print(
                f"Balanced sampler: per-source counts {{{counts_report}}}; "
                f"num_samples={len(train_dataset)} replacement=True"
            )

        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_instances,
        )

        pipeline = load_pipeline(args.config, args.device, args.compile_model)
        pointmap_store = (
            Moge2PointmapStore(args.moge2_pointmap_dir)
            if args.moge2_pointmap_dir
            else None
        )
        feature_cache = []
        heldout_cache = []

        if args.cache_latents:
            if not args.cache_heldout_only:
                feature_cache = cache_metric_scale_features(
                    pipeline,
                    train_dataset,
                    args.min_mask_pixels,
                    args.stage1_steps,
                    args.stage2_steps,
                    "caching train features",
                    pointmap_store=pointmap_store,
                    partial_path=(
                        f"{args.save_feature_cache}.train.partial"
                        if args.save_feature_cache
                        else None
                    ),
                )
                if not feature_cache:
                    raise RuntimeError(
                        "No metric-scale training examples remained after filtering."
                    )
            if eval_dataset is not None:
                heldout_cache = cache_metric_scale_features(
                    pipeline,
                    eval_dataset,
                    args.min_mask_pixels,
                    args.stage1_steps,
                    args.stage2_steps,
                    "caching held-out features",
                    pointmap_store=pointmap_store,
                    partial_path=(
                        f"{args.save_feature_cache}.heldout.partial"
                        if args.save_feature_cache
                        else None
                    ),
                )
                if not heldout_cache:
                    print("Held-out split had no usable examples after mask filtering.")
            if args.save_feature_cache:
                save_feature_cache(
                    args.save_feature_cache, feature_cache, heldout_cache, args
                )
                for suffix in (".train.partial", ".heldout.partial"):
                    partial = Path(f"{args.save_feature_cache}{suffix}")
                    partial.unlink(missing_ok=True)
            if args.cache_only:
                return

    scale_head = MetricScaleHead().to(pipeline.device).train()
    if args.decoder == "anchor":
        scale_decoder = AnchorAugmentedDecoder().to(pipeline.device).train()
    elif args.decoder == "factored":
        scale_decoder = FactoredScaleDecoder().to(pipeline.device).train()
    elif args.decoder == "binned":
        scale_decoder = BinnedScaleDecoder(
            num_bins=args.num_bins, log_min=args.bin_log_min, log_max=args.bin_log_max
        ).to(pipeline.device).train()
    else:
        scale_decoder = MetricScaleDecoder().to(pipeline.device).train()

    anchors = None
    if args.decoder != "baseline":
        if not args.anchor_tables:
            raise RuntimeError(f"--decoder {args.decoder} requires --anchor-tables.")
        if not (args.cache_latents or args.load_feature_cache):
            raise RuntimeError(f"--decoder {args.decoder} is cached-regime only.")
        anchors = load_anchor_tables(args.anchor_tables)
        for name, cache_list in (("train", feature_cache), ("heldout", heldout_cache)):
            before = len(cache_list)
            cache_list[:] = [f for f in cache_list if f["uid"] in anchors]
            if len(cache_list) != before:
                print(f"{name} cache: dropped {before - len(cache_list)} of {before} "
                      f"entries without anchor records")

    # Selectively unfreeze SLAT cross-attn after all other pipeline params are
    # frozen.  collect_slat_cross_attn_params flips requires_grad back to True
    # on cross_attn + norm2 in all 24 blocks and returns the param list.
    slat_backbone = None
    slat_cross_attn_params = []
    if args.unfreeze_slat_cross_attn:
        slat_cross_attn_params, slat_backbone = collect_slat_cross_attn_params(
            pipeline, upcast_fp32=args.fp32_slat_cross_attn
        )
        for block in slat_backbone.blocks:
            block.use_checkpoint = args.slat_grad_checkpoint
        print(f"{'Enabled' if args.slat_grad_checkpoint else 'Disabled'} gradient checkpointing "
              f"on {len(slat_backbone.blocks)} SLAT transformer blocks")

    if args.native_fm and not args.unfreeze_ss_backbone:
        raise SystemExit("--native-fm requires --unfreeze-ss-backbone (trains the layout transformer).")

    ss_decoder_params = []
    if args.unfreeze_ss_decoder:
        ss_decoder_params = collect_ss_decoder_params(pipeline)

    # Joint-MoT: unfreeze the SS generator backbone into 3 discriminative-LR groups.
    ss_backbone_groups = {"geometry": [], "pose": [], "cond": []}
    ss_backbone = None
    if args.unfreeze_ss_backbone:
        ss_backbone_groups, ss_backbone = collect_ss_backbone_params(
            pipeline,
            unfreeze_cross_attn=args.unfreeze_ss_cross_attn,
            upcast_fp32=True,
        )
        for block in ss_backbone.blocks:
            block.use_checkpoint = args.ss_grad_checkpoint
        print(f"{'Enabled' if args.ss_grad_checkpoint else 'Disabled'} gradient checkpointing "
              f"on {len(ss_backbone.blocks)} SS transformer blocks")

    # Hybrid: MetricScaleHead path + SS layout transformer trains via gradient through
    # ss_scale_features.  Keeps all of mixed_moge2_live_v1's machinery (scale token,
    # SLAT injection, MetricScaleDecoder) while letting the MoT layout blocks adapt.
    if getattr(args, "unfreeze_ss_layout_for_head", False):
        if args.unfreeze_ss_backbone:
            raise SystemExit("--unfreeze-ss-layout-for-head and --unfreeze-ss-backbone are mutually exclusive.")
        if args.native_fm:
            raise SystemExit("--unfreeze-ss-layout-for-head and --native-fm are mutually exclusive.")
        if args.cache_latents or args.load_feature_cache:
            raise SystemExit("--unfreeze-ss-layout-for-head is incompatible with cached-latent mode.")
        ss_backbone_groups, ss_backbone = collect_ss_backbone_params(
            pipeline,
            unfreeze_cross_attn=args.unfreeze_ss_cross_attn,
            upcast_fp32=True,
        )
        for block in ss_backbone.blocks:
            block.use_checkpoint = args.ss_grad_checkpoint
        print(f"{'Enabled' if args.ss_grad_checkpoint else 'Disabled'} gradient checkpointing "
              f"on {len(ss_backbone.blocks)} SS transformer blocks (layout-for-head hybrid mode)")
        if args.stage1_steps is None:
            args.stage1_steps = 4
            print("--unfreeze-ss-layout-for-head: --stage1-steps unset, defaulting to 4")

    # L_flow geometry anchor (L2-SP): snapshot the pretrained geometry weights so we can penalize
    # drift from them — the explicit safeguard against trading geometry for metric (the flow-matching
    # objective isn't exposed; L2-SP is the standard transfer-learning substitute). Off if weight 0.
    ss_geometry_init = None
    if args.unfreeze_ss_backbone and args.flow_loss_weight > 0:
        ss_geometry_init = [p.detach().clone() for p in ss_backbone_groups["geometry"]]
        n = sum(p.numel() for p in ss_geometry_init)
        print(f"L2-SP geometry anchor armed: {len(ss_geometry_init)} tensors / {n/1e6:.1f}M params "
              f"(flow_loss_weight={args.flow_loss_weight})")

    # Joint-MoT: default the SS flow to 4 steps (the recipe value) when unset — gradient backprops
    # through every step, so an unset (config-default, ~25) would be needlessly heavy.
    if args.unfreeze_ss_backbone and args.stage1_steps is None:
        args.stage1_steps = 4
        print("Joint-MoT: --stage1-steps unset, defaulting to 4")

    resume_start_epoch = 0
    resume_path = args.resume_from or args.load_checkpoint
    if resume_path:
        ckpt = load_metric_checkpoint(resume_path, scale_head, scale_decoder, slat_backbone,
                                      ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None,
                                      ss_backbone=ss_backbone)
        if args.resume_from and ckpt.get("epoch") is not None:
            resume_start_epoch = int(ckpt["epoch"])
            print(f"Resuming from epoch {resume_start_epoch}")

    if args.eval_only:
        if not args.load_feature_cache:
            raise RuntimeError("--eval-only requires --load-feature-cache.")
        if not args.load_checkpoint:
            raise RuntimeError("--eval-only requires --load-checkpoint.")
        baseline_metrics = category_mean_baseline(feature_cache, heldout_cache)
        if baseline_metrics is not None:
            print(format_metrics("category_baseline_heldout", None, baseline_metrics))
            print_category_metrics("category_baseline_heldout", baseline_metrics)
            write_metrics(
                args.metrics_output,
                "category_baseline_heldout",
                None,
                baseline_metrics,
            )
            wandb_log(
                wandb_run,
                flatten_metrics("category_baseline_heldout", baseline_metrics),
            )
        eval_metrics = evaluate_and_record(
            pipeline,
            scale_head,
            scale_decoder,
            feature_cache,
            heldout_cache,
            args.metrics_output,
            wandb_run,
            None,
            anchors=anchors,
        )
        write_manifest(
            args.manifest_output,
            args,
            feature_cache,
            heldout_cache,
            checkpoint_path=args.load_checkpoint,
            cache_path=args.load_feature_cache,
            best_metrics=eval_metrics.get("heldout"),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    param_groups = [
        {
            "params": list(scale_head.parameters()) + list(scale_decoder.parameters()),
            "lr": args.lr,
        }
    ]
    if slat_cross_attn_params:
        param_groups.append({"params": slat_cross_attn_params, "lr": args.slat_lr})
    if ss_decoder_params:
        param_groups.append({"params": ss_decoder_params, "lr": args.lr})
    # SS-backbone discriminative-LR groups (geometry slow / pose fast / cond mid).
    if ss_backbone_groups["geometry"]:
        param_groups.append({"params": ss_backbone_groups["geometry"], "lr": args.ss_geometry_lr})
    if ss_backbone_groups["pose"]:
        param_groups.append({"params": ss_backbone_groups["pose"], "lr": args.ss_pose_lr})
    if ss_backbone_groups["cond"]:
        param_groups.append({"params": ss_backbone_groups["cond"], "lr": args.ss_cond_lr})
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95.0, buffer_size=1000)

    if args.cache_latents or args.load_feature_cache:
        baseline_metrics = category_mean_baseline(feature_cache, heldout_cache)
        if baseline_metrics is not None:
            print(format_metrics("category_baseline_heldout", None, baseline_metrics))
            print_category_metrics("category_baseline_heldout", baseline_metrics)
            write_metrics(
                args.metrics_output,
                "category_baseline_heldout",
                None,
                baseline_metrics,
            )
            wandb_log(
                wandb_run,
                flatten_metrics("category_baseline_heldout", baseline_metrics),
            )
        best_heldout_mean_abs_pct = float("inf")
        best_output = args.best_output or default_best_output_path(args.output)
        best_metrics = None
        for epoch in range(args.epochs):
            running_loss = 0.0
            running_count = 0
            order = torch.randperm(len(feature_cache)).tolist()
            progress = tqdm(
                range(0, len(order), args.batch_size), desc=f"epoch {epoch + 1}/{args.epochs}"
            )
            for offset in progress:
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for idx in order[offset : offset + args.batch_size]:
                    features = feature_cache[idx]
                    aux = None
                    if args.decoder == "binned":
                        log_whd, aux = predict_cached_log_dims(
                            pipeline, scale_head, scale_decoder, features,
                            anchors=anchors, return_aux=True,
                        )
                        log_pred = log_whd[0]
                    else:
                        log_pred = predict_cached_log_dims(
                            pipeline, scale_head, scale_decoder, features, anchors=anchors
                        )[0]
                    log_target = torch.log(
                        features["metric_dims"].to(pipeline.device).clamp(min=1e-6)
                    )
                    if args.decoder == "binned":
                        # OmniNOCS-style: softmax-CE over iso-scale bins
                        # (classification of the metric scalar) + GT-normalized
                        # L1 on the iso scalar + max-pinned proportions loss.
                        iso_target = log_target.max()
                        target_bin = scale_decoder.target_bin(iso_target.view(1))
                        ce = F.cross_entropy(aux["bin_logits"], target_bin)
                        iso_pred = log_pred.max()  # == log_iso by construction
                        norm_l1 = (iso_pred.exp() - iso_target.exp()).abs() / iso_target.exp().clamp(min=1e-6)
                        prop_loss = F.smooth_l1_loss(
                            log_pred - log_pred.max(),
                            log_target - log_target.max(),
                        )
                        losses.append(
                            args.bin_ce_weight * ce
                            + args.scale_loss_weight * norm_l1
                            + args.prop_loss_weight * prop_loss
                        )
                    elif args.decoder == "factored":
                        # Decoupled losses (MoGe-2 §3.2): max(log_pred) == log_iso
                        # reaches the iso branch only; max-pinned log dims reach
                        # the proportions branch only.
                        iso_loss = (log_pred.max() - log_target.max()) ** 2
                        prop_loss = F.smooth_l1_loss(
                            log_pred - log_pred.max(),
                            log_target - log_target.max(),
                        )
                        losses.append(
                            args.scale_loss_weight * iso_loss
                            + args.prop_loss_weight * prop_loss
                        )
                    elif args.scale_loss_weight > 0:
                        # v1a: absolute log loss + auxiliary iso-scale loss,
                        # shared gradients through one prediction.
                        losses.append(
                            F.smooth_l1_loss(log_pred, log_target)
                            + args.scale_loss_weight
                            * (log_pred.max() - log_target.max()) ** 2
                        )
                    else:
                        losses.append(F.smooth_l1_loss(log_pred, log_target))

                loss = torch.stack(losses).mean()
                loss.backward()
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                grad_norm = grad_clipper(all_params)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

                batch_count = len(losses)
                running_loss += float(loss.detach().cpu()) * batch_count
                running_count += batch_count
                progress.set_postfix(loss=running_loss / max(running_count, 1))

            epoch_loss = running_loss / max(running_count, 1)
            wandb_log(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "train_epoch/loss": epoch_loss,
                    "train_epoch/examples": running_count,
                    "train_epoch/lr": optimizer.param_groups[0]["lr"],
                },
            )

            if args.eval_every and (epoch + 1) % args.eval_every == 0:
                eval_metrics = evaluate_and_record(
                    pipeline,
                    scale_head,
                    scale_decoder,
                    feature_cache,
                    heldout_cache,
                    args.metrics_output,
                    wandb_run,
                    epoch + 1,
                    anchors=anchors,
                )
                heldout_metrics = eval_metrics.get("heldout")
                if heldout_metrics is not None:
                    heldout_mean_abs_pct = heldout_metrics["mean_abs_pct"]
                    if heldout_mean_abs_pct < best_heldout_mean_abs_pct:
                        best_heldout_mean_abs_pct = heldout_mean_abs_pct
                        best_metrics = heldout_metrics
                        save_metric_checkpoint(
                            best_output,
                            scale_head,
                            scale_decoder,
                            args,
                            epoch + 1,
                            heldout_metrics,
                            slat_backbone=slat_backbone,
                            ss_backbone=ss_backbone,
                            ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None,
                        )
                        print(
                            f"Saved best metric scale checkpoint to {best_output} "
                            f"(epoch={epoch + 1}, heldout_mean_abs_pct={heldout_mean_abs_pct:.2f})"
                        )
                        wandb_log(
                            wandb_run,
                            {
                                "best/epoch": epoch + 1,
                                "best/heldout_mean_abs_pct": heldout_mean_abs_pct,
                                "best/checkpoint_path": best_output,
                            },
                        )
    else:
        best_heldout_mean_abs_pct = float("inf")
        best_output = args.best_output or default_best_output_path(args.output)
        best_metrics = None

        # Baseline eval against live held-out examples before any training.
        if eval_dataset is not None and args.eval_every and resume_start_epoch == 0:
            heldout_metrics = evaluate_live_dataset(
                pipeline,
                scale_head,
                scale_decoder,
                eval_dataset,
                args.min_mask_pixels,
                args.stage1_steps,
                args.stage2_steps,
                inject_scale_token=args.inject_scale_token_into_slat,
                desc="live heldout eval baseline",
                pointmap_store=pointmap_store,
                unfreeze_ss_backbone=args.unfreeze_ss_backbone,
                native_fm=args.native_fm,
            )
            if heldout_metrics is not None:
                print(format_metrics("live_heldout_eval", None, heldout_metrics))
                print_category_metrics("live_heldout_eval", heldout_metrics)
                write_metrics(args.metrics_output, "live_heldout", None, heldout_metrics)
                wandb_log(wandb_run, flatten_metrics("live_heldout_eval", heldout_metrics))

        recovery_output = (
            str(Path(args.output).with_name(f"{Path(args.output).stem}_resume{Path(args.output).suffix}"))
            if (args.checkpoint_every or args.checkpoint_every_steps) else None
        )

        slat_grad_verified = not args.unfreeze_slat_cross_attn
        # Independently verify the SS layout transformer receives gradient on the first
        # backward.  Covers BOTH the joint-MoT path (--unfreeze-ss-backbone) and the
        # hybrid head path (--unfreeze-ss-layout-for-head); a silent no-grad here (e.g. a
        # stray no_grad/detach reintroduced upstream) would train nothing while looking healthy.
        ss_grad_verified = not (
            args.unfreeze_ss_backbone
            or getattr(args, "unfreeze_ss_layout_for_head", False)
        )
        oom_skipped = 0
        nan_skipped = 0
        consecutive_nan_skipped = 0
        pipeline_skipped = 0
        global_step = 0

        # Per-group param lists for separate grad-norm logging.
        head_params = optimizer.param_groups[0]["params"]
        slat_params = (
            optimizer.param_groups[1]["params"] if len(optimizer.param_groups) > 1 else []
        )

        def _record_nan_skip(reason: str, epoch_idx: int, step_idx: int, **extra) -> bool:
            """Common skip path: zero grads, bump counters, log to wandb, return True
            if the watchdog should fire (caller raises). Reason is one of:
              loss_nan, grad_nan, clipped_grad_nan
            """
            nonlocal nan_skipped, consecutive_nan_skipped
            optimizer.zero_grad(set_to_none=True)
            nan_skipped += 1
            consecutive_nan_skipped += 1
            payload = {
                "train_skip/reason": reason,
                "train_skip/nan_skipped": nan_skipped,
                "train_skip/consecutive_nan_skipped": consecutive_nan_skipped,
                "train_skip/oom_skipped": oom_skipped,
                "train_skip/epoch": epoch_idx + 1,
                "train_skip/step_idx": step_idx,
                "train_skip/global_step": global_step,
            }
            payload.update(extra)
            wandb_log(wandb_run, payload)
            if (
                args.max_consecutive_nan_skips
                and consecutive_nan_skipped >= args.max_consecutive_nan_skips
            ):
                return True
            return False

        def _heldout_eval_and_save(
            epoch_label: int, step_label: int | None, max_samples: int | None
        ) -> None:
            """Run a live held-out eval, log it, and save the best checkpoint on
            improvement. Shared by the end-of-epoch path (step_label=None, full
            eval) and the intra-epoch path (step_label set, optionally strided).
            Mutates best_heldout_mean_abs_pct / best_metrics via nonlocal.
            evaluate_live_dataset flips the heads to eval() and restores their
            prior (train) mode in its own finally block."""
            nonlocal best_heldout_mean_abs_pct, best_metrics
            if eval_dataset is None:
                return
            tag = (
                f"epoch {epoch_label}"
                if step_label is None
                else f"epoch {epoch_label} step {step_label}"
            )
            heldout_metrics = evaluate_live_dataset(
                pipeline,
                scale_head,
                scale_decoder,
                eval_dataset,
                args.min_mask_pixels,
                args.stage1_steps,
                args.stage2_steps,
                inject_scale_token=args.inject_scale_token_into_slat,
                desc=f"live heldout eval {tag}",
                max_samples=max_samples,
                pointmap_store=pointmap_store,
                unfreeze_ss_backbone=args.unfreeze_ss_backbone,
                native_fm=args.native_fm,
            )
            if heldout_metrics is None:
                return
            print(format_metrics("live_heldout_eval", epoch_label, heldout_metrics))
            print_category_metrics("live_heldout_eval", heldout_metrics)
            write_metrics(args.metrics_output, "live_heldout", epoch_label, heldout_metrics)
            payload = flatten_metrics("live_heldout_eval", heldout_metrics)
            if step_label is not None:
                payload["live_heldout_eval/global_step"] = step_label
            wandb_log(wandb_run, payload)
            heldout_mean_abs_pct = heldout_metrics["mean_abs_pct"]
            if heldout_mean_abs_pct < best_heldout_mean_abs_pct:
                best_heldout_mean_abs_pct = heldout_mean_abs_pct
                best_metrics = heldout_metrics
                save_metric_checkpoint(
                    best_output,
                    scale_head,
                    scale_decoder,
                    args,
                    epoch_label,
                    heldout_metrics,
                    slat_backbone=slat_backbone,
                    ss_backbone=ss_backbone,
                    ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None,
                )
                print(
                    f"Saved best checkpoint to {best_output} "
                    f"(epoch={epoch_label}, step={step_label}, "
                    f"heldout_mean_abs_pct={heldout_mean_abs_pct:.2f})"
                )
                wandb_log(
                    wandb_run,
                    {
                        "best/epoch": epoch_label,
                        "best/heldout_mean_abs_pct": heldout_mean_abs_pct,
                        "best/checkpoint_path": best_output,
                    },
                )

        def _save_recovery(epoch_label: int) -> None:
            """Overwrite the rolling recovery checkpoint so a crash loses at most
            the steps since the last save (weights only; resume is epoch-granular)."""
            if not recovery_output:
                return
            save_metric_checkpoint(
                recovery_output,
                scale_head,
                scale_decoder,
                args,
                epoch_label,
                slat_backbone=slat_backbone,
                ss_backbone=ss_backbone,
                ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None,
            )
            print(f"Saved recovery checkpoint to {recovery_output} (epoch={epoch_label})")

        for epoch in range(resume_start_epoch, args.epochs):
            running_loss = 0.0
            running_count = 0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for step_idx, batch in enumerate(progress):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                step_dropouts = 0
                try:
                    for image, metric_dims, mask_pixels, image_name, gt_trans, has_trans in zip(
                        batch["images"], batch["metric_dims"], batch["mask_pixels"],
                        batch["image_names"], batch["translations"], batch["has_translation"],
                    ):
                        if int(mask_pixels) < args.min_mask_pixels:
                            continue
                        pred_translation = None
                        try:
                            if args.native_fm:
                                # Native random-tau flow matching (paper L_CFM): one forward,
                                # velocity MSE on the layout modalities. Self-skips the
                                # predict_pose_metric sample-then-regress path below.
                                _lw = {
                                    "shape": 0.0,
                                    "scale": args.fm_scale_weight,
                                    "translation": args.fm_trans_weight,
                                    "translation_scale": args.fm_trans_weight,
                                    "6drotation_normalized": 0.0,
                                }
                                native_loss = native_fm_layout_loss(
                                    pipeline, image, metric_dims, gt_trans,
                                    pointmap=(pointmap_store.lookup(image_name)
                                              if pointmap_store is not None else None),
                                    shape_steps=args.fm_shape_steps, loss_weights=_lw,
                                    shape_seed=shape_sample_seed(image_name),
                                )
                                losses.append(native_loss)
                                continue
                            if args.unfreeze_ss_backbone:
                                # Joint-MoT: pose decoder predicts metric W,H,D + translation.
                                # Seed the shape sample so canon_ext is deterministic per item
                                # (same seed as eval) → stable regression target for scale_per_axis.
                                # Without seeding, stochastic canon_ext makes smooth_l1 targets
                                # inconsistent across steps on the same training sample.
                                _pm_out = predict_pose_metric(
                                    pipeline,
                                    image,
                                    args.stage1_steps,
                                    pointmap=(
                                        pointmap_store.lookup(image_name)
                                        if pointmap_store is not None else None
                                    ),
                                    with_grad=True,
                                    shape_seed=shape_sample_seed(image_name),
                                )
                                log_pred = _pm_out["log_dims"]
                                pred_translation = _pm_out["translation"]
                                dropped = False
                                ratio_loss = torch.zeros((), device=pipeline.device)
                            else:
                                log_pred, dropped, ratio_loss, pred_translation = predict_log_dims(
                                    pipeline,
                                    scale_head,
                                    scale_decoder,
                                    image,
                                    args.stage1_steps,
                                    args.stage2_steps,
                                    inject_scale_token=args.inject_scale_token_into_slat,
                                    unfreeze_cross_attn=args.unfreeze_slat_cross_attn,
                                    p_uncond_scale_token=args.p_uncond_scale_token,
                                    ss_ratio_loss_weight=args.ss_ratio_loss_weight,
                                    gt_dims=metric_dims.unsqueeze(0),
                                    pointmap=(
                                        pointmap_store.lookup(image_name)
                                        if pointmap_store is not None
                                        else None
                                    ),
                                    unfreeze_layout_for_head=getattr(
                                        args, "unfreeze_ss_layout_for_head", False
                                    ),
                                    return_translation=(
                                        args.trans_loss_weight > 0
                                        and getattr(args, "unfreeze_ss_layout_for_head", False)
                                    ),
                                )
                        except (IndexError, RuntimeError) as pipe_err:
                            # SS / SLAT pipeline can fail on degenerate inputs — e.g.
                            # SS predicts zero voxels and `prune_sparse_structure` calls
                            # `coords.min(0)` on an empty tensor (IndexError). We log and
                            # skip the offending sample so a single bad input doesn't
                            # abort training. CUDA OOM has its own outer handler.
                            msg = str(pipe_err)
                            if "out of memory" in msg.lower():
                                raise
                            pipeline_skipped += 1
                            print(
                                f"\n[pipeline-skip] epoch {epoch + 1} step {step_idx}: "
                                f"{type(pipe_err).__name__}: {msg[:200]} "
                                f"(total skips={pipeline_skipped})"
                            )
                            torch.cuda.empty_cache()
                            continue
                        if dropped:
                            step_dropouts += 1
                        log_target = torch.log(metric_dims.to(pipeline.device).clamp(min=1e-6))
                        sample_loss = (
                            F.smooth_l1_loss(log_pred[0], log_target)
                            + args.ss_ratio_loss_weight * ratio_loss
                        )
                        if (pred_translation is not None and bool(has_trans)
                                and args.trans_loss_weight > 0):
                            # GT translation -> pose-decoder (instance_position_l2c, PyTorch3D)
                            # frame: negate x,y (OpenCV->PyTorch3D), per step 0.5. Toggle via flag.
                            _gt_t = gt_trans.to(pipeline.device).clone()
                            if args.gt_translation_flip_xy:
                                _gt_t[0] = -_gt_t[0]
                                _gt_t[1] = -_gt_t[1]
                            _pt = pred_translation
                            _pt = _pt[0] if _pt.dim() > 1 else _pt
                            sample_loss = sample_loss + args.trans_loss_weight * compute_translation_loss(_pt, _gt_t)
                        losses.append(sample_loss)

                    if not losses:
                        continue

                    loss = torch.stack(losses).mean()
                    if ss_geometry_init is not None:
                        # L2-SP geometry anchor: mean squared drift of geometry weights from their
                        # pretrained init (scale-stable so flow_loss_weight is interpretable).
                        _gp = ss_backbone_groups["geometry"]
                        _sq = sum(((p - p0) ** 2).sum() for p, p0 in zip(_gp, ss_geometry_init))
                        _n = sum(p.numel() for p in _gp)
                        loss = loss + args.flow_loss_weight * (_sq / max(_n, 1))
                    if torch.isnan(loss) or torch.isinf(loss):
                        if _record_nan_skip(
                            "loss_nan", epoch, step_idx,
                            **{"train_skip/loss": float(loss.detach().cpu())},
                        ):
                            raise RuntimeError(
                                f"NaN watchdog: {consecutive_nan_skipped} consecutive "
                                f"NaN-skipped steps (threshold={args.max_consecutive_nan_skips}). "
                                "Aborting training. Check fp32-cross-attn flag and slat-lr."
                            )
                        continue
                    loss.backward()
                    if slat_cross_attn_params and not slat_grad_verified:
                        if not any(p.grad is not None for p in slat_cross_attn_params):
                            raise RuntimeError(
                                "SLAT cross-attention parameters received no gradient on the "
                                "first backward pass. This typically means sample_slat is wrapping "
                                "the generator forward in torch.no_grad() — pass with_grad=True "
                                "to allow gradient flow through cross_attn.to_kv."
                            )
                        slat_grad_verified = True
                    if not ss_grad_verified:
                        if not any(p.grad is not None for p in ss_backbone_groups["pose"]):
                            raise RuntimeError(
                                "SS layout transformer received no gradient — "
                                "sample_sparse_structure must be called with with_grad=True "
                                "(predict_pose_metric for --unfreeze-ss-backbone, or "
                                "predict_log_dims with unfreeze_layout_for_head=True for "
                                "--unfreeze-ss-layout-for-head)."
                            )
                        ss_grad_verified = True

                    # Explicit per-param finite check (TRELLIS pattern). Catches
                    # the rare case of a corrupted single-param gradient that
                    # leaves total norm finite.
                    all_params = [p for g in optimizer.param_groups for p in g["params"]]
                    nan_in_grads = any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in all_params
                    )
                    if nan_in_grads:
                        if _record_nan_skip("grad_nan", epoch, step_idx):
                            raise RuntimeError(
                                f"NaN watchdog: {consecutive_nan_skipped} consecutive "
                                f"NaN-skipped steps (threshold={args.max_consecutive_nan_skips}). "
                                "Aborting training. Check fp32-cross-attn flag and slat-lr."
                            )
                        continue

                    # Pre-clip per-group grad norms for diagnostics. max_norm=inf
                    # returns the norm without modifying gradients.
                    heads_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(head_params, max_norm=float("inf"))
                    )
                    slat_grad_norm = (
                        float(torch.nn.utils.clip_grad_norm_(slat_params, max_norm=float("inf")))
                        if slat_params
                        else 0.0
                    )

                    grad_norm = grad_clipper(all_params)
                    if not torch.isfinite(grad_norm):
                        if _record_nan_skip(
                            "clipped_grad_nan", epoch, step_idx,
                            **{
                                "train_skip/heads_grad_norm": heads_grad_norm,
                                "train_skip/slat_grad_norm": slat_grad_norm,
                            },
                        ):
                            raise RuntimeError(
                                f"NaN watchdog: {consecutive_nan_skipped} consecutive "
                                f"NaN-skipped steps (threshold={args.max_consecutive_nan_skips}). "
                                "Aborting training. Check fp32-cross-attn flag and slat-lr."
                            )
                        continue

                    # Linear warmup on the SLAT cross-attn LR group only. Heads
                    # do not need warmup — they're at lr=1e-4 and their warm-start
                    # weights handle that scale fine.
                    if (
                        slat_params
                        and args.slat_lr_warmup_steps
                        and global_step < args.slat_lr_warmup_steps
                    ):
                        warm_factor = (global_step + 1) / args.slat_lr_warmup_steps
                        optimizer.param_groups[1]["lr"] = args.slat_lr * warm_factor

                    optimizer.step()
                    global_step += 1
                    consecutive_nan_skipped = 0

                    batch_count = len(losses)
                    running_loss += float(loss.detach().cpu()) * batch_count
                    running_count += batch_count
                    progress.set_postfix(loss=running_loss / max(running_count, 1))

                    if args.log_step_every and global_step % args.log_step_every == 0:
                        clip_log = grad_clipper.log()
                        wandb_log(
                            wandb_run,
                            {
                                "train_step/loss": float(loss.detach().cpu()),
                                "train_step/grad_norm": float(grad_norm),
                                "train_step/heads_grad_norm": heads_grad_norm,
                                "train_step/slat_grad_norm": slat_grad_norm,
                                "train_step/clip_threshold": clip_log["max_norm"],
                                "train_step/clip_buffer_filled": int(
                                    clip_log["buffer_filled"]
                                ),
                                "train_step/nan_skipped": nan_skipped,
                                "train_step/oom_skipped": oom_skipped,
                                "train_step/scale_token_dropped": step_dropouts,
                                "train_step/epoch": epoch + 1,
                                "train_step/global_step": global_step,
                            },
                        )

                    if (step_idx + 1) % 500 == 0:
                        torch.cuda.empty_cache()

                    # Intra-epoch eval + recovery (fast feedback + restart safety).
                    # Runs only on a successful optimizer step, so global_step is a
                    # clean monotonic trigger and never double-fires on a skip/OOM.
                    if (
                        args.eval_every_steps
                        and global_step % args.eval_every_steps == 0
                    ):
                        _heldout_eval_and_save(
                            epoch + 1, global_step, args.eval_steps_max_samples or None
                        )
                    if (
                        args.checkpoint_every_steps
                        and global_step % args.checkpoint_every_steps == 0
                    ):
                        _save_recovery(epoch + 1)

                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    torch.cuda.empty_cache()
                    oom_skipped += 1
                    print(
                        f"\n[OOM] epoch {epoch + 1} step {step_idx}: skipped "
                        f"(total OOM skips={oom_skipped})"
                    )
                    wandb_log(
                        wandb_run,
                        {
                            "train_skip/reason": "oom",
                            "train_skip/oom_skipped": oom_skipped,
                            "train_skip/nan_skipped": nan_skipped,
                            "train_skip/consecutive_nan_skipped": consecutive_nan_skipped,
                            "train_skip/epoch": epoch + 1,
                            "train_skip/step_idx": step_idx,
                            "train_skip/global_step": global_step,
                        },
                    )

            epoch_loss = running_loss / max(running_count, 1)
            wandb_payload = {
                "epoch": epoch + 1,
                "train_epoch/loss": epoch_loss,
                "train_epoch/examples": running_count,
                "train_epoch/lr": optimizer.param_groups[0]["lr"],
                "train_epoch/oom_skipped": oom_skipped,
                "train_epoch/nan_skipped": nan_skipped,
                "train_epoch/global_step": global_step,
            }
            if slat_cross_attn_params:
                wandb_payload["train_epoch/slat_lr"] = optimizer.param_groups[1]["lr"]
            wandb_log(wandb_run, wandb_payload)

            # End-of-epoch live eval against held-out examples (always the full
            # set). Re-runs SS/SLAT so metrics reflect current cross-attn weights.
            if args.eval_every and (epoch + 1) % args.eval_every == 0:
                _heldout_eval_and_save(epoch + 1, None, None)

            if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
                _save_recovery(epoch + 1)

    output_path = Path(args.output)
    save_metric_checkpoint(
        str(output_path), scale_head, scale_decoder, args, slat_backbone=slat_backbone,
        ss_backbone=ss_backbone,
        ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None,
    )
    print(f"Saved metric scale checkpoint to {output_path}")
    has_cache = args.cache_latents or args.load_feature_cache
    write_manifest(
        args.manifest_output,
        args,
        feature_cache,
        heldout_cache,
        checkpoint_path=str(output_path),
        best_checkpoint_path=(
            best_output if best_metrics is not None else None
        ),
        best_metrics=best_metrics,
        cache_path=args.load_feature_cache or args.save_feature_cache,
    )
    wandb_log(wandb_run, {"final/checkpoint_path": str(output_path)})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
