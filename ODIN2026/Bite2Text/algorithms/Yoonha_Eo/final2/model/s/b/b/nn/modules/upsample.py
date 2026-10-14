import torch.nn as nn

from teethland import PointTensor
from teethland.nn.modules.linear import Linear
from teethland.nn.modules.normalization import LayerNorm


class UpsampleBlock(nn.Module):
    """Implements point cloud upsampling with neural network layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: nn.Module=LayerNorm,
        k: int=3,
    ):
        super().__init__()

        self.norm = norm_layer(in_channels)
        self.linear = Linear(in_channels, out_channels, bias=True)

        self.norm_up = norm_layer(out_channels)
        self.linear_up = Linear(out_channels, out_channels, bias=True)

        self.k = k

    def forward(
        self,
        x: PointTensor,
        x_up: PointTensor,
    ) -> PointTensor:
        x = self.norm(x)
        x = self.linear(x)

        x_up = self.norm_up(x_up)
        x_up = self.linear_up(x_up)
        
        return x.interpolate(x_up, self.k) + x_up
