import torch.nn as nn

from teethland import PointTensor


class Linear(nn.Module):
    """Implements linear layer for point cloud tensor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool=False,
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_channels, out_channels, bias)
        
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: PointTensor) -> PointTensor:
        return x.new_tensor(features=self.linear(x.F))
