# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from typing import Optional


class MetricScaleHead(nn.Module):
    """
    Predicts log(metric_scale) from a pooled SS shape latent combined with
    MoGe pointmap statistics (scale and shift_z).

    The SS latent contributes object proportions; pointmap_scale/shift provide
    the metric anchor since MoGe outputs depth in meters. Neither alone is
    sufficient — the latent has no metric grounding, and the pointmap statistics
    encode scene depth without object-specific shape context.

    Input dim: latent_dim (8) + log(pointmap_scale) (1) + pointmap_shift_z (1) = 10
    Output: log(metric_scale) scalar [batch, 1]
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        shape_latent: torch.Tensor,
        pointmap_scale: Optional[torch.Tensor],
        pointmap_shift: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            shape_latent:    [batch, 4096, 8]  — SS generator output
            pointmap_scale:  [batch, ...] or None — MoGe scale statistic
            pointmap_shift:  [batch, ...] or None — MoGe shift statistic

        Returns:
            log_scale: [batch, 1] — log(metric_scale in meters)
        """
        pooled = shape_latent.mean(dim=1)  # [batch, 8]
        batch, device, dtype = pooled.shape[0], pooled.device, pooled.dtype

        if pointmap_scale is not None:
            ps = pointmap_scale.to(dtype).view(batch, -1).mean(dim=1, keepdim=True)
            log_ps = torch.log(ps.clamp(min=1e-6))
        else:
            log_ps = torch.zeros(batch, 1, device=device, dtype=dtype)

        if pointmap_shift is not None:
            shift = pointmap_shift.to(dtype).view(batch, -1)
            # Take z-component (depth shift); fall back to first component if 1-d
            shift_z = shift[:, 2:3] if shift.shape[1] >= 3 else shift[:, :1]
        else:
            shift_z = torch.zeros(batch, 1, device=device, dtype=dtype)

        x = torch.cat([pooled, log_ps, shift_z], dim=1)  # [batch, 10]
        return self.mlp(x)  # [batch, 1]


class ScaleTokenProjector(nn.Module):
    """
    Projects a log(metric_scale) scalar into a dense conditioning token
    compatible with SLAT's cross-attention (ctx_channels typically = 768).

    Output is [batch, 1, ctx_channels] — a single extra token appended to
    the DINO token sequence before cross-attention.
    """

    def __init__(self, ctx_channels: int = 768):
        super().__init__()
        self.proj = nn.Linear(1, ctx_channels)

    def forward(self, log_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_scale: [batch, 1]
        Returns:
            scale_token: [batch, 1, ctx_channels]
        """
        return self.proj(log_scale).unsqueeze(1)


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
        cond = self._base(*args, **kwargs)          # [batch, seq_len, ctx_ch]
        return torch.cat([cond, self._scale_token], dim=1)  # [batch, seq_len+1, ctx_ch]

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name: str, value):
        if name in ("_base", "_scale_token"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_base"), name, value)
