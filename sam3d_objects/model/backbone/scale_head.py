# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from typing import Optional


class MetricScaleHead(nn.Module):
    """
    Produces a 768-dim scale conditioning token from a pooled SS shape latent
    combined with MoGe pointmap statistics (scale and shift_z).

    The SS latent contributes object proportions; pointmap_scale/shift provide
    the metric anchor since MoGe outputs depth in meters. Neither alone is
    sufficient — the latent has no metric grounding, and the pointmap statistics
    encode scene depth without object-specific shape context.

    A single linear projection from a scalar bottleneck was deliberately avoided:
    the MLP projects directly to ctx_channels so the token can encode richer
    scale-related structure beyond a single magnitude value.

    Input dim:  latent_dim (8) + log(pointmap_scale) (1) + pointmap_shift_z (1) = 10
    Output:     scale token [batch, 1, ctx_channels]
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64, ctx_channels: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, ctx_channels),
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
            scale_token: [batch, 1, ctx_channels]
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
            shift_z = shift[:, 2:3] if shift.shape[1] >= 3 else shift[:, :1]
        else:
            shift_z = torch.zeros(batch, 1, device=device, dtype=dtype)

        x = torch.cat([pooled, log_ps, shift_z], dim=1)  # [batch, 10]
        token = self.mlp(x)                               # [batch, ctx_channels]
        return token.unsqueeze(1)                         # [batch, 1, ctx_channels]


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
        return torch.cat([cond, self._scale_token], dim=1)         # [batch, seq_len+1, ctx_ch]

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name: str, value):
        if name in ("_base", "_scale_token"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_base"), name, value)
