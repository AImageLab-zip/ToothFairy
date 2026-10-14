import torch.nn as nn

from teethland import PointTensor


class ReLU(nn.Module):
    """Implements rectified linear unit activation function."""

    def __init__(self):
        super().__init__()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: PointTensor) -> PointTensor:
        return x.new_tensor(features=self.activation(x.F))


class GELU(nn.Module):
    """Implements Gaussian error linear unit activation function."""

    def __init__(self):
        super().__init__()

        self.activation = nn.GELU()

    def forward(self, x: PointTensor) -> PointTensor:
        return x.new_tensor(features=self.activation(x.F))


class LeakyReLU(nn.Module):
    """Implements leaky rectified linear unit activation function."""

    def __init__(
        self,
        negative_slope: float=0.2,
    ):
        super().__init__()

        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        

    def forward(self, x: PointTensor) -> PointTensor:
        return x.new_tensor(features=self.activation(x.F))
