import torch.nn as nn
from timm.models.layers import DropPath as DropPathLayer

from teethland import PointTensor


class DropPath(nn.Module):
    """Implements Dropout for model paths, used with residual connections."""

    def __init__(self, prob: float):
        super().__init__()

        self.drop_path = DropPathLayer(prob)

    def forward(self, x: PointTensor) -> PointTensor:
        return x.new_tensor(features=self.drop_path(x.F))
