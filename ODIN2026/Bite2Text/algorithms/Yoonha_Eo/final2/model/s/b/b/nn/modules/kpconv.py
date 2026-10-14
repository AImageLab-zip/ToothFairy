import torch.nn as nn

from teethland import PointTensor
from teethland.nn.modules.activation import LeakyReLU
from teethland.nn.modules.feedforward import FCLayer
from teethland.nn.modules.normalization import BatchNorm
from teethland.nn.modules.torch_points3d import KPConvLayer


class KPConvResidualBlock(nn.Module):
    """Implements residual block with Kernel Point Convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        point_influence: float,
        ball_radius: float,
    ):
        super().__init__()

        self.fc1 = FCLayer(in_channels, out_channels // 4)
        self.kpconv = KPConv(
            in_channels=out_channels // 4,
            out_channels=out_channels // 4,
            point_influence=point_influence,
            ball_radius=ball_radius,
            norm_layer=nn.Identity,
            activation_layer=nn.Identity,
        )
        self.fc2 = FCLayer(out_channels // 4, out_channels)

        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = FCLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=nn.Identity,
            )

    def forward(self, x: PointTensor) -> PointTensor:
        x_res = self.fc1(x)
        x_res = self.kpconv(x_res)
        x_res = self.fc2(x_res)

        return self.projection(x) + x_res


class KPConv(nn.Module):
    """Implements Kernel Point Convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        point_influence: float,
        ball_radius: float,
        norm_layer: nn.Module=BatchNorm,
        activation_layer: nn.Module=LeakyReLU,
        k: int=34,
    ):
        super().__init__()

        self.kpconv = KPConvLayer(
            num_inputs=in_channels,
            num_outputs=out_channels,
            point_influence=point_influence,
        )
        self.norm = norm_layer(out_channels)
        self.activation = activation_layer()
        self.k = k
        self.ball_radius = ball_radius

    def forward(self, x: PointTensor) -> PointTensor:
        neighbor_idxs, _ = x.neighbors(
            k=self.k, cache=True, method='ball', radius=self.ball_radius,
        )

        x = x.new_tensor(
            features=self.kpconv(
                query_points=x.C,
                support_points=x.C,
                # clone necessary due to in-place operation
                neighbors=neighbor_idxs.clone(),                  
                x=x.F,
            ),
        )
        x = self.norm(x)
        x = self.activation(x)

        return x
