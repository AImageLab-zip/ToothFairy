import torch.nn as nn

from teethland import PointTensor
from teethland.nn.modules.activation import GELU, LeakyReLU
from teethland.nn.modules.linear import Linear
from teethland.nn.modules.normalization import BatchNorm


class MLP(nn.Module):
    """Implements two-layer perceptron with non-linear activation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation_layer: nn.Module=GELU,
    ):
        super().__init__()

        self.linear1 = Linear(in_channels, hidden_channels, bias=True)
        self.activation = activation_layer()
        self.linear2 = Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x: PointTensor) -> PointTensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class FCLayer(nn.Module):
    """Implements linear layer with BatchNorm and leaky ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: nn.Module=BatchNorm,
        activation_layer: nn.Module=LeakyReLU,
    ):
        super().__init__()

        self.linear = Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.activation = activation_layer()

    def forward(self, x: PointTensor) -> PointTensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)

        return x
