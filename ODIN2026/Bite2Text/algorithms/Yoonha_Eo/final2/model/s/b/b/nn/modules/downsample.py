import torch.nn as nn

from teethland import PointTensor
from teethland.nn.modules.linear import Linear
from teethland.nn.modules.normalization import LayerNorm
from teethland.nn.modules.pooling import GroupedMaxPool


class DownsampleBlock(nn.Module):
    """Implements point cloud downsampling with neural network layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        norm_layer: nn.Module=LayerNorm,
        k: int=16,
    ):
        super().__init__()
        
        self.norm = norm_layer(in_channels)
        self.linear = Linear(in_channels, out_channels)
        self.grouped_pool = GroupedMaxPool(k)
        
        self.ratio = ratio
        self.k = k

    def forward(self, x: PointTensor) -> PointTensor:
        x_down = x.downsample(self.ratio)
        grouped_x_down = x.queryandgroup(x_down, self.k)
        grouped_x_down = self.norm(grouped_x_down)
        grouped_x_down = self.linear(grouped_x_down)
        x_down = self.grouped_pool(grouped_x_down)
        
        return x_down
