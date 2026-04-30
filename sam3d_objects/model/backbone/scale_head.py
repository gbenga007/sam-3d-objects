# Copyright (c) Meta Platforms, Inc. and affiliates.
import math

import torch
import torch.nn as nn
from typing import Optional


# SS generator scale token width: only the supervised `scale` output is
# consumed (3 dims = log-SSI scale per axis).  Rotation, translation, and
# translation_scale are intentionally excluded — they encode *pose*, not
# size, and would inject noise into a head whose target is metric WHD.
SS_SCALE_FEAT_DIM = 3


class MetricScaleHead(nn.Module):
    """
    Produces a 1024-d scale conditioning token from:
      - mean-pooled SS shape latent (object proportions),
      - the SS generator's `scale` token (log-SSI scale, supervised with
        loss_weight=0.1 in pretraining; carries full-image + cropped
        DINOv2 + pointmap context absorbed by the SS condition embedder),
      - MoGe pointmap_scale and shift_z (metric anchor).

    A learned `metric_modality_embed` is added to the output token so that
    SLAT cross-attention can distinguish it from DINOv2 image/mask tokens
    even though it shares the same 1024-d space (the existing EmbedderFuser
    only assigns pos_group identifiers to its own modalities; an injected
    token has no such identifier without this embedding).
    """

    def __init__(
        self,
        latent_dim: int = 8,
        ss_scale_dim: int = SS_SCALE_FEAT_DIM,
        hidden_dim: int = 64,
        ctx_channels: int = 1024,
    ):
        super().__init__()
        self.ss_scale_dim = ss_scale_dim
        in_dim = latent_dim + ss_scale_dim + 2  # +2 for log_pointmap_scale and shift_z
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, ctx_channels),
        )
        self.metric_modality_embed = nn.Parameter(
            torch.empty(1, 1, ctx_channels)
        )
        nn.init.normal_(
            self.metric_modality_embed, mean=0.0, std=1.0 / math.sqrt(ctx_channels)
        )

    def forward(
        self,
        shape_latent: torch.Tensor,
        ss_scale_features: Optional[torch.Tensor],
        pointmap_scale: Optional[torch.Tensor],
        pointmap_shift: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            shape_latent:       [batch, 4096, 8]   — SS shape latent
            ss_scale_features:  [batch, ss_scale_dim] or None — SS generator
                                `scale` token (log-SSI scale per axis).
                                Pass None for backward compatibility
                                (zero-filled).
            pointmap_scale:     [batch, ...] or None — MoGe scale statistic
            pointmap_shift:     [batch, ...] or None — MoGe shift statistic

        Returns:
            scale_token: [batch, 1, ctx_channels]
        """
        param_dtype = next(self.parameters()).dtype
        pooled = shape_latent.mean(dim=1).to(param_dtype)  # [batch, 8]
        batch, device, dtype = pooled.shape[0], pooled.device, pooled.dtype

        if ss_scale_features is None:
            ss_scale_features = torch.zeros(batch, self.ss_scale_dim, device=device, dtype=dtype)
        else:
            ss_scale_features = ss_scale_features.to(device=device, dtype=dtype)
            if ss_scale_features.ndim == 1:
                ss_scale_features = ss_scale_features.unsqueeze(0)
            if ss_scale_features.shape[-1] != self.ss_scale_dim:
                raise ValueError(
                    f"ss_scale_features last dim {ss_scale_features.shape[-1]} "
                    f"does not match ss_scale_dim {self.ss_scale_dim}"
                )

        if pointmap_scale is not None:
            ps = pointmap_scale.to(dtype).view(batch, -1).mean(dim=1, keepdim=True)
            log_ps = torch.log(ps.clamp(min=1e-6))
        else:
            log_ps = torch.zeros(batch, 1, device=device, dtype=dtype)

        if pointmap_shift is not None:
            shift = pointmap_shift.to(dtype).view(batch, -1)
            shift_z = shift[:, 2:3] if shift.shape[1] >= 3 else shift[:, :1]
        else:
            shift_z = torch.zeros(batch, 1, device=device, dtype=dtype)

        x = torch.cat([pooled, ss_scale_features, log_ps, shift_z], dim=1)
        token = self.mlp(x).unsqueeze(1)                                  # [batch, 1, ctx_channels]
        return token + self.metric_modality_embed                          # broadcast over batch


def extract_ss_scale_features(ss_return_dict: dict) -> Optional[torch.Tensor]:
    """
    Extract the SS generator's `scale` token as a [batch, 3] feature vector
    for MetricScaleHead.

    Returns None if the SS output does not contain the `scale` key (e.g.
    non-MM-DiT mode), so the caller can decide on a zero-fill fallback.

    Pose-related outputs (rotation, translation, translation_scale) are
    *not* included — they encode where the object is, not how big it is,
    and have no monotonic relationship to target dimensions.
    """
    if "scale" not in ss_return_dict:
        return None
    t = ss_return_dict["scale"]
    if t.ndim == 3:  # [B, 1, 3] → [B, 3]
        t = t.squeeze(1)
    if t.shape[-1] != SS_SCALE_FEAT_DIM:
        return None
    return t.float()


class _ScaleAugmentedEmbedderProxy:
    """
    Plain Python proxy that wraps a condition embedder and appends a scale token
    to its output. Used to inject metric scale conditioning into the SLAT
    generator without modifying the generator architecture.

    Attribute access and mutation are forwarded to the base embedder so that
    CFG dropout (force_drop_modalities get/set) works transparently.
    """

    def __init__(self, base_embedder, scale_token: torch.Tensor):
        object.__setattr__(self, "_base", base_embedder)
        object.__setattr__(self, "_scale_token", scale_token)

    def __call__(self, *args, **kwargs):
        cond = self._base(*args, **kwargs)                          # [batch, seq_len, ctx_ch]
        if cond.shape[-1] != self._scale_token.shape[-1]:
            return cond
        return torch.cat([cond, self._scale_token], dim=1)         # [batch, seq_len+1, ctx_ch]

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name: str, value):
        if name in ("_base", "_scale_token"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_base"), name, value)
