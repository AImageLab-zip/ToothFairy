from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import segment_csr

from ..builder import MODELS, build_model
from pointcept.models.utils.structure import Point


class ModernHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def packed_to_padded(feat: torch.Tensor, offset: torch.Tensor):
    ends = offset.long().tolist()
    starts = [0, *ends[:-1]]
    sequences = [feat[start:end] for start, end in zip(starts, ends)]
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], device=feat.device)
    padded = pad_sequence(sequences, batch_first=True)
    padding_mask = torch.arange(padded.shape[1], device=feat.device)[None] >= lengths[:, None]
    return padded, padding_mask


@MODELS.register_module()
class AdvancedMultiTaskClassifier(nn.Module):
    """Official-scale PTv3 classifier supporting the cumulative A0-A6 stages."""

    def __init__(
        self,
        backbone,
        num_classes_list,
        backbone_embed_dim=512,
        pooling="mean",
        task_relation=False,
        adaptive_loss=False,
        class_weights=None,
        label_smoothing=0.05,
        head_dropout=0.2,
        legacy_head=False,
        attention_heads=8,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.num_classes_list = list(num_classes_list)
        self.num_tasks = len(self.num_classes_list)
        self.pooling = pooling
        self.adaptive_loss = adaptive_loss
        dim = backbone_embed_dim

        if pooling == "mean_max_attention":
            self.global_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.global_attention = nn.MultiheadAttention(
                dim, attention_heads, dropout=0.1, batch_first=True
            )
            self.pool_projection = nn.Sequential(
                nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU()
            )
        elif pooling == "task_query":
            self.task_queries = nn.Parameter(torch.randn(1, self.num_tasks, dim) * 0.02)
            self.task_attention = nn.MultiheadAttention(
                dim, attention_heads, dropout=0.1, batch_first=True
            )

        self.relation = None
        if task_relation:
            relation_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=attention_heads,
                dim_feedforward=dim * 2,
                dropout=0.15,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.relation = nn.TransformerEncoder(
                relation_layer, num_layers=1, enable_nested_tensor=False
            )

        self.heads = nn.ModuleList()
        for num_classes in self.num_classes_list:
            if legacy_head:
                head = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes),
                )
            else:
                head = ModernHead(dim, num_classes, head_dropout)
            self.heads.append(head)

        for task, weights in enumerate(class_weights or [None] * self.num_tasks):
            value = None if weights is None else torch.tensor(weights, dtype=torch.float32)
            self.register_buffer(f"class_weight_{task}", value)
        self.label_smoothing = label_smoothing
        if adaptive_loss:
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))

    def _pool(self, point: Point):
        feat, offset = point.feat, point.offset
        indptr = F.pad(offset, (1, 0))
        if self.pooling == "mean":
            pooled = segment_csr(feat, indptr=indptr, reduce="mean")
            return pooled[:, None, :].expand(-1, self.num_tasks, -1)

        padded, padding_mask = packed_to_padded(feat, offset)
        batch_size = padded.shape[0]
        if self.pooling == "mean_max_attention":
            mean = segment_csr(feat, indptr=indptr, reduce="mean")
            maximum = segment_csr(feat, indptr=indptr, reduce="max")
            query = self.global_query.expand(batch_size, -1, -1)
            attended, _ = self.global_attention(
                query, padded, padded, key_padding_mask=padding_mask, need_weights=False
            )
            pooled = self.pool_projection(torch.cat([mean, maximum, attended[:, 0]], dim=-1))
            return pooled[:, None, :].expand(-1, self.num_tasks, -1)
        if self.pooling == "task_query":
            query = self.task_queries.expand(batch_size, -1, -1)
            tokens, _ = self.task_attention(
                query, padded, padded, key_padding_mask=padding_mask, need_weights=False
            )
            return tokens
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, input_dict):
        point = self.backbone(Point(input_dict))
        tokens = self._pool(point)
        if self.relation is not None:
            tokens = self.relation(tokens)
        logits = [head(tokens[:, task]) for task, head in enumerate(self.heads)]
        output = {"logits": logits}

        if not any(key.startswith("label_") for key in input_dict):
            return output
        task_losses = []
        for task, logit in enumerate(logits):
            target = input_dict[f"label_{task}"].view(-1)
            valid = target != -1
            if valid.any():
                loss = F.cross_entropy(
                    logit[valid],
                    target[valid],
                    weight=getattr(self, f"class_weight_{task}"),
                    label_smoothing=self.label_smoothing,
                )
            else:
                loss = logit.sum() * 0.0
            task_losses.append(loss)
            output[f"loss_{task}"] = loss

        if self.adaptive_loss:
            weighted = [
                torch.exp(-self.log_vars[task]) * loss + self.log_vars[task]
                for task, loss in enumerate(task_losses)
            ]
            output["loss"] = torch.stack(weighted).mean()
        else:
            output["loss"] = torch.stack(task_losses).mean()
        return output
