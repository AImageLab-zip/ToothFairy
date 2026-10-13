"""Neural components used by the released EviDent-CBCT Stage 2 branches."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SmallCropEncoder(nn.Module):
    """Compact 3D encoder for fixed-size local CT and mask crops."""

    def __init__(self, in_channels: int = 6, base_channels: int = 8, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels * 2, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels * 4, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(base_channels * 4, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LabelSpecificHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [MLPHead(in_dim, 1, hidden_dim, dropout) for _ in range(int(num_labels))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class GatedFusion(nn.Module):
    """Inject structure-token context into crop features with a learned gate."""

    def __init__(
        self,
        raw_dim: int,
        context_dim: int,
        out_dim: int,
        extra_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.raw_proj = nn.Linear(raw_dim, out_dim)
        self.context_proj = nn.Linear(context_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(raw_dim + context_dim + extra_dim, out_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        raw: torch.Tensor,
        context: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [raw, context]
        if extra is not None:
            parts.append(extra)
        gate = self.gate(torch.cat(parts, dim=-1))
        fused = self.raw_proj(raw) + gate * self.context_proj(context)
        return self.dropout(self.norm(fused))
