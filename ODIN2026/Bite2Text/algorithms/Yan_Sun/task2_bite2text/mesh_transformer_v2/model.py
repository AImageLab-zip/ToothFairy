"""A compact contact-aware point token Transformer for paired IOS meshes."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContactPointTransformer(nn.Module):
    def __init__(self, head_vocabs: dict[str, list[str]], dim: int = 192, tokens_per_jaw: int = 128, layers: int = 4) -> None:
        super().__init__()
        if dim % 6:
            raise ValueError("dim must be divisible by 6 for the Transformer heads")
        self.tokens_per_jaw = tokens_per_jaw
        self.point_stem = nn.Sequential(
            nn.Linear(8, 96), nn.LayerNorm(96), nn.GELU(),
            nn.Linear(96, dim), nn.LayerNorm(dim), nn.GELU(),
        )
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.jaw_embedding = nn.Parameter(torch.empty(2, dim))
        nn.init.normal_(self.jaw_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=6, dim_feedforward=dim * 3, dropout=0.15,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(384, 256), nn.GELU(), nn.Dropout(0.15),
        )
        self.heads = nn.ModuleDict({head: nn.Linear(256, len(vocab)) for head, vocab in head_vocabs.items()})

    @staticmethod
    def _contact_features(source: Tensor, opposite: Tensor) -> tuple[Tensor, Tensor]:
        """Nearest opposing-surface distance and signed vertical gap, no gradients needed."""
        with torch.no_grad():
            distances = torch.cdist(source.float(), opposite.float())
            closest_distance, closest_index = distances.min(dim=2)
            nearest_opposite = opposite.gather(1, closest_index.unsqueeze(-1).expand(-1, -1, 3))
            vertical_gap = source[..., 2] - nearest_opposite[..., 2]
        return closest_distance.unsqueeze(-1), vertical_gap.unsqueeze(-1)

    def _tokenize(self, xyz: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        """Pool randomly surface-sampled points into spatial tokens around anchor points."""
        batch, points, dim = features.shape
        count = min(self.tokens_per_jaw, points)
        # Surface sampling has already randomized point order, so evenly spaced indices
        # yield reproducible, well-distributed random anchors without a slow FPS loop.
        anchor_index = torch.linspace(0, points - 1, count, device=xyz.device).long()
        anchors = xyz[:, anchor_index]
        assignment = torch.cdist(xyz.float(), anchors.float()).argmin(dim=2)
        feature_sum = torch.zeros(batch, count, dim, device=features.device, dtype=features.dtype)
        position_sum = torch.zeros(batch, count, 3, device=xyz.device, dtype=xyz.dtype)
        feature_sum.scatter_add_(1, assignment.unsqueeze(-1).expand(-1, -1, dim), features)
        position_sum.scatter_add_(1, assignment.unsqueeze(-1).expand(-1, -1, 3), xyz)
        point_counts = torch.zeros(batch, count, 1, device=features.device, dtype=features.dtype)
        point_counts.scatter_add_(1, assignment.unsqueeze(-1), torch.ones(batch, points, 1, device=features.device, dtype=features.dtype))
        return feature_sum / point_counts.clamp_min(1), position_sum / point_counts.to(position_sum.dtype).clamp_min(1)

    def forward(self, upper_xyz: Tensor, upper_normals: Tensor, lower_xyz: Tensor, lower_normals: Tensor) -> dict[str, Tensor]:
        upper_distance, upper_gap = self._contact_features(upper_xyz, lower_xyz)
        lower_distance, lower_gap = self._contact_features(lower_xyz, upper_xyz)
        upper_features = self.point_stem(torch.cat([upper_xyz, upper_normals, upper_distance, upper_gap], dim=-1))
        lower_features = self.point_stem(torch.cat([lower_xyz, lower_normals, lower_distance, lower_gap], dim=-1))
        upper_tokens, upper_centers = self._tokenize(upper_xyz, upper_features)
        lower_tokens, lower_centers = self._tokenize(lower_xyz, lower_features)
        upper_tokens = upper_tokens + self.position(upper_centers) + self.jaw_embedding[0]
        lower_tokens = lower_tokens + self.position(lower_centers) + self.jaw_embedding[1]
        tokens = self.transformer(torch.cat([upper_tokens, lower_tokens], dim=1))
        features = self.fusion(torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=1))
        return {head: classifier(features) for head, classifier in self.heads.items()}
