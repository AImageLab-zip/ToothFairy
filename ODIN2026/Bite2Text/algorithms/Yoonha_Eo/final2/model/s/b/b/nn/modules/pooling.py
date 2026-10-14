from typing import Any, Dict, List, Tuple

import teethland
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torchtyping import TensorType

from teethland import PointTensor


class GroupedMaxPool(nn.Module):
    """Maximum pooling for point cloud tensor with grouped features."""

    def __init__(
        self,
        kernel_size: int,
    ):
        super().__init__()

        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x: PointTensor) -> PointTensor:
        x = x.new_tensor(features=x.F.transpose(1, 2))
        x.F = self.pool(x.F)
        x.F = x.F.squeeze(-1)

        return x


class MaskedAveragePooling(nn.Module):
    """Masked average pooling per instance followed by MLP for predictions."""

    def __init__(
        self,
        num_features: int,
        out_channels: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_features, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, out_channels, bias=True),
        )

        self.num_features = num_features
        self.out_channels = out_channels
    
    def param_groups(
        self,
        lr: float,
    ) -> List[Dict[str, Any]]:
        return [
            {
                'params': self.parameters(),
                'lr': lr,
                'name': 'masked_average_pooling',
            },
        ]

    def forward(
        self,
        features: PointTensor,
        instances: PointTensor,
    ) -> Tuple[PointTensor, TensorType['B', 'C', torch.float32]]:
        # apply instance masks to get prototypes
        prototypes, embeddings = [], []
        for id in instances.F.unique()[1:]:
            prototype = features[instances.F == id]

            prototypes.append(prototype)
            embeddings.append(prototype.F.mean(0))

        # get prototypes into single PointTensor
        prototypes = teethland.stack(prototypes)
        embeddings = torch.stack(embeddings)

        # process prototypes for prediction
        centroids = scatter_mean(
            instances.C[instances.F >= 0], 
            instances.F[instances.F >= 0],
            dim=0
        )
        out = PointTensor(
            coordinates=centroids,
            features=self.mlp(embeddings),
        )

        return prototypes, out
