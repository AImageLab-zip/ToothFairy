import torch.nn as nn

from teethland import PointTensor
from teethland.nn.modules.torch_points3d import FastBatchNorm1d


class BatchNorm(nn.Module):
    """Implements BatchNorm for point cloud tensor."""

    def __init__(
        self,
        channels: int,
        momentum: float=0.02,
    ):
        super().__init__()

        self.norm = FastBatchNorm1d(channels, momentum)
        
        nn.init.constant_(self.norm.batch_norm.weight, 1)
        nn.init.constant_(self.norm.batch_norm.bias, 0)

    def forward(self, x: PointTensor) -> PointTensor:
        return x.new_tensor(features=self.norm(x.F))


class LayerNorm(nn.Module):
    """Implements LayerNorm for point cloud tensor."""

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)
    
    def forward(self, x: PointTensor) -> PointTensor:
        return x.new_tensor(features=self.norm(x.F))
