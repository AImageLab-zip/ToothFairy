"""Shared-weight multi-view ResNet with one classification head per photo finding.

One encoder runs over each of the five intraoral views; features are pooled
(masked mean, or learned attention) so a missing view is simply masked out.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_BACKBONES = {"resnet18": models.resnet18, "resnet34": models.resnet34}


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(feats).squeeze(-1).masked_fill(~mask, float("-inf"))
        empty = ~mask.any(dim=1, keepdim=True)
        logits = torch.where(empty.expand_as(logits), torch.zeros_like(logits), logits)
        return (feats * F.softmax(logits, dim=1).unsqueeze(-1)).sum(dim=1)


class MultiViewPhotoNet(nn.Module):
    def __init__(self, head_sizes: Dict[str, int], backbone: str = "resnet18",
                 pooling: str = "mean", dropout: float = 0.3):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose from {list(_BACKBONES)}")
        net = _BACKBONES[backbone](weights=None)       # weights come from the checkpoint
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.pool = AttentionPool(self.feat_dim) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(self.feat_dim, n) for h, n in head_sizes.items()})

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """images ``(B, V, 3, H, W)``, view_mask ``(B, V)`` -> ``{head: (B, n_classes)}``."""
        b, v = images.shape[:2]
        feats = self.backbone(images.flatten(0, 1)).view(b, v, self.feat_dim)
        if self.pool is not None:
            pooled = self.pool(feats, view_mask)
        else:
            m = view_mask.unsqueeze(-1).float()
            pooled = (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return {h: head(pooled) for h, head in self.heads.items()}
