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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from sam3d_objects.data.dataset.metric import OmniNOCSObjectDataset, OmniNOCSReal275Dataset
from sam3d_objects.model.backbone.metric_scale_decoder import MetricScaleDecoder
from sam3d_objects.model.backbone.scale_head import (
    MetricScaleHead,
    _ScaleAugmentedEmbedderProxy,
    extract_ss_scale_features,
)


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
        "categories": [item["category"] for item in batch],
        "uids": [item["uid"] for item in batch],
        "mask_pixels": torch.tensor(
            [item["mask_pixels"] for item in batch], dtype=torch.long
        ),
    }


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
) -> tuple[torch.Tensor, bool, torch.Tensor]:
    """
    Run the full SAM3D pipeline and return predicted log-dimensions.

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
        pointmap_dict = pipeline.compute_pointmap(image)
        ss_input_dict = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input_dict = pipeline.preprocess_image(image, pipeline.slat_preprocessor)

        # SS stage: always frozen.
        with torch.no_grad():
            ss_return_dict = pipeline.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_steps,
                use_distillation=False,
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
        # the decoder directly.  SS scale features are detached (SS is frozen).
        ss_scale_features = extract_ss_scale_features(ss_return_dict)
        if ss_scale_features is not None:
            ss_scale_features = ss_scale_features.detach().to(pipeline.device)
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
        return log_dims, scale_token_dropped, ss_ratio_loss


def encode_metric_scale_features(
    pipeline,
    item: dict,
    metric_dims: torch.Tensor,
    stage1_steps: int | None,
    stage2_steps: int | None,
) -> dict:
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(item["image"])
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
    scale_decoder: MetricScaleDecoder,
    features: dict,
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
    return scale_decoder(
        features["slat_feats"].to(pipeline.device),
        scale_token,
        features["batch_indices"].to(pipeline.device),
    )


def evaluate_cached_features(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    feature_cache: list[dict],
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
                pipeline, scale_head, scale_decoder, features
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
) -> dict | None:
    """
    Evaluate with a live SS/SLAT forward pass so metrics reflect the current
    SLAT cross-attention weights, not stale cached latents.
    """
    if dataset is None or len(dataset) == 0:
        return None

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
            for item in tqdm(dataset, desc=desc):
                if int(item["mask_pixels"]) < min_mask_pixels:
                    continue
                try:
                    log_pred, _, _ratio = predict_log_dims(
                        pipeline,
                        scale_head,
                        scale_decoder,
                        item["image"],
                        stage1_steps,
                        stage2_steps,
                        inject_scale_token=inject_scale_token,
                        unfreeze_cross_attn=False,
                        p_uncond_scale_token=0.0,
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
) -> list[dict]:
    feature_cache = []
    skipped_errors = 0
    for item in tqdm(dataset, desc=desc):
        if int(item["mask_pixels"]) < min_mask_pixels:
            continue
        try:
            feature_cache.append(
                encode_metric_scale_features(
                    pipeline,
                    item,
                    torch.as_tensor(item["metric_dims"], dtype=torch.float32),
                    stage1_steps,
                    stage2_steps,
                )
            )
        except Exception as exc:
            skipped_errors += 1
            print(
                "Skipping feature-cache example due to pipeline error: "
                f"uid={item.get('uid')} source={item.get('source')} "
                f"image_name={item.get('image_name')} error={exc}"
            )
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


def load_feature_cache(path: str) -> tuple[list[dict], list[dict], dict]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    train_cache = cache.get("train", [])
    heldout_cache = cache.get("heldout", [])
    if not train_cache:
        raise RuntimeError(f"No train features found in cache: {path}")
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
    print(f"Loaded metric scale checkpoint from {path}")
    return checkpoint


def evaluate_and_record(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    feature_cache: list[dict],
    heldout_cache: list[dict],
    metrics_output: str | None,
    wandb_run=None,
    epoch: int | None = None,
) -> dict:
    train_metrics = evaluate_cached_features(
        pipeline, scale_head, scale_decoder, feature_cache
    )
    print(format_metrics("train_eval", epoch, train_metrics))
    print_category_metrics("train_eval", train_metrics)
    write_metrics(metrics_output, "train", epoch, train_metrics)
    wandb_payload = flatten_metrics("train_eval", train_metrics)
    if heldout_cache:
        heldout_metrics = evaluate_cached_features(
            pipeline, scale_head, scale_decoder, heldout_cache
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-steps", type=int, default=None)
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
        feature_cache, heldout_cache, cache_args = load_feature_cache(args.load_feature_cache)
        validate_feature_cache_manifest(
            args.validate_manifest,
            feature_cache,
            heldout_cache,
        )
        pipeline = DeviceOnlyPipeline(args.device)
        train_dataset = None
        eval_dataset = None
        loader = None
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
        feature_cache = []
        heldout_cache = []

        if args.cache_latents:
            feature_cache = cache_metric_scale_features(
                pipeline,
                train_dataset,
                args.min_mask_pixels,
                args.stage1_steps,
                args.stage2_steps,
                "caching train features",
            )
            if not feature_cache:
                raise RuntimeError("No metric-scale training examples remained after filtering.")
            if eval_dataset is not None:
                heldout_cache = cache_metric_scale_features(
                    pipeline,
                    eval_dataset,
                    args.min_mask_pixels,
                    args.stage1_steps,
                    args.stage2_steps,
                    "caching held-out features",
                )
                if not heldout_cache:
                    print("Held-out split had no usable examples after mask filtering.")
            if args.save_feature_cache:
                save_feature_cache(
                    args.save_feature_cache, feature_cache, heldout_cache, args
                )
            if args.cache_only:
                return

    scale_head = MetricScaleHead().to(pipeline.device).train()
    scale_decoder = MetricScaleDecoder().to(pipeline.device).train()

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
            block.use_checkpoint = True
        print(f"Enabled gradient checkpointing on {len(slat_backbone.blocks)} SLAT transformer blocks")

    ss_decoder_params = []
    if args.unfreeze_ss_decoder:
        ss_decoder_params = collect_ss_decoder_params(pipeline)

    resume_start_epoch = 0
    resume_path = args.resume_from or args.load_checkpoint
    if resume_path:
        ckpt = load_metric_checkpoint(resume_path, scale_head, scale_decoder, slat_backbone,
                                      ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None)
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
                    log_pred = predict_cached_log_dims(
                        pipeline, scale_head, scale_decoder, features
                    )
                    log_target = torch.log(
                        features["metric_dims"].to(pipeline.device).clamp(min=1e-6)
                    )
                    losses.append(F.smooth_l1_loss(log_pred[0], log_target))

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
            )
            if heldout_metrics is not None:
                print(format_metrics("live_heldout_eval", None, heldout_metrics))
                print_category_metrics("live_heldout_eval", heldout_metrics)
                write_metrics(args.metrics_output, "live_heldout", None, heldout_metrics)
                wandb_log(wandb_run, flatten_metrics("live_heldout_eval", heldout_metrics))

        recovery_output = (
            str(Path(args.output).with_name(f"{Path(args.output).stem}_resume{Path(args.output).suffix}"))
            if args.checkpoint_every else None
        )

        slat_grad_verified = not args.unfreeze_slat_cross_attn
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

        for epoch in range(resume_start_epoch, args.epochs):
            running_loss = 0.0
            running_count = 0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for step_idx, batch in enumerate(progress):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                step_dropouts = 0
                try:
                    for image, metric_dims, mask_pixels in zip(
                        batch["images"], batch["metric_dims"], batch["mask_pixels"]
                    ):
                        if int(mask_pixels) < args.min_mask_pixels:
                            continue
                        try:
                            log_pred, dropped, ratio_loss = predict_log_dims(
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
                        losses.append(
                            F.smooth_l1_loss(log_pred[0], log_target)
                            + args.ss_ratio_loss_weight * ratio_loss
                        )

                    if not losses:
                        continue

                    loss = torch.stack(losses).mean()
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
                    if not slat_grad_verified:
                        if not any(p.grad is not None for p in slat_cross_attn_params):
                            raise RuntimeError(
                                "SLAT cross-attention parameters received no gradient on the "
                                "first backward pass. This typically means sample_slat is wrapping "
                                "the generator forward in torch.no_grad() — pass with_grad=True "
                                "to allow gradient flow through cross_attn.to_kv."
                            )
                        slat_grad_verified = True

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

            # Periodic live eval against held-out examples. This re-runs SS/SLAT
            # so metrics reflect the current SLAT cross-attention weights.
            if args.eval_every and (epoch + 1) % args.eval_every == 0 and eval_dataset is not None:
                heldout_metrics = evaluate_live_dataset(
                    pipeline,
                    scale_head,
                    scale_decoder,
                    eval_dataset,
                    args.min_mask_pixels,
                    args.stage1_steps,
                    args.stage2_steps,
                    inject_scale_token=args.inject_scale_token_into_slat,
                    desc=f"live heldout eval epoch {epoch + 1}",
                )
                if heldout_metrics is not None:
                    print(format_metrics("live_heldout_eval", epoch + 1, heldout_metrics))
                    print_category_metrics("live_heldout_eval", heldout_metrics)
                    write_metrics(
                        args.metrics_output,
                        "live_heldout",
                        epoch + 1,
                        heldout_metrics,
                    )
                    wandb_log(
                        wandb_run,
                        flatten_metrics("live_heldout_eval", heldout_metrics),
                    )
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
                            ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None,
                        )
                        print(
                            f"Saved best checkpoint to {best_output} "
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

            if recovery_output and args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
                save_metric_checkpoint(
                    recovery_output,
                    scale_head,
                    scale_decoder,
                    args,
                    epoch + 1,
                    slat_backbone=slat_backbone,
                    ss_decoder=pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None,
                )
                print(f"Saved recovery checkpoint to {recovery_output} (epoch={epoch + 1})")

    output_path = Path(args.output)
    save_metric_checkpoint(
        str(output_path), scale_head, scale_decoder, args, slat_backbone=slat_backbone,
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
