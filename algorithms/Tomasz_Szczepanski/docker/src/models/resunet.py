import warnings
import torch
import torch.nn as nn
import torch.nn.functional as f

from typing import Optional, Sequence, Tuple, Union
from monai.networks.layers.factories import Act, Norm
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.blocks import MaxAvgPool

if __name__ == "__main__":
    from resnet import get_outplanes, resnet
else:
    from .resnet import get_outplanes, resnet


class ResUNet(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int] = 3,
            up_kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 0,
            act: Union[Tuple, str] = Act.RELU,
            norm: Union[Tuple, str] = Norm.BATCH,
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            backbone_name: str = 'resnet18',
            dimensions: Optional[int] = None,
            big_decoder: bool = False
    ) -> None:

        super().__init__()

        backbone = resnet(
            norm=norm,
            act=act,
            resnet_type=backbone_name,
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=out_channels,
            conv1_t_size=3,
            conv1_t_stride=2,
            no_max_pool=True
        )

        backbone_layers = backbone.get_encoder_layers()
        up_channels = get_outplanes(backbone_name)
        strides = [2, 1, 2, 2]

        if len(up_channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(up_channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.big_decoder = big_decoder
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_channels = up_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.encoder_layers = backbone_layers

        self.encoder_blocks = []
        self.decoder_blocks = []

        # self.pooling_concat = MaxAvgPool(spatial_dims=3, kernel_size=16, stride=1)

        def _create_block(
                outc: int, up_channels: Sequence[int], strides: Sequence[int], is_top: bool
        ):  # -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            # Feature map sizes
            s = strides[0]
            in_down = up_channels[0]
            up_conv = up_channels[1]
            concat = up_conv + in_down

            # subblock: nn.Module

            if len(up_channels) > 2:
                # continue recursion down
                subblock = _create_block(up_conv, up_channels[1:], strides[1:], False)
            else:
                # get encoder layer for downsampling path
                subblock = self.encoder_layers.pop()
                self.encoder_blocks.append(subblock)

            # get encoder layer for downsampling path
            down = self.encoder_layers.pop()
            # create layer in upsampling path
            up = self._get_up_layer(concat, outc, s, is_top)
            self.encoder_blocks.append(down)
            self.decoder_blocks.append(up)
            # return self._get_connection_block(down, up, subblock)

        self.model = _create_block(self.out_channels, self.up_channels, self.strides, True)
        # reverse recursively built encoder
        self.encoder_blocks.reverse()
        # assign to nn.Module
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # add residual connection not to destroy features behind - allow for identity
        # self.num_res_units=1

        # self.edt_decoder = self.decoder_blocks[-1]
        # # self.seeds_decoder = self._get_up_layer(up_channels[0]+up_channels[1], 1, 2, True)
        # self.segmentation_decoder = nn.Sequential(
        #     self._get_up_layer(up_channels[1]+up_channels[2], up_channels[1], 1, False),
        #     self._get_up_layer(up_channels[0]+up_channels[1], 1, 2, True))
        # self.direction_decoder = self._get_up_layer(up_channels[0]+up_channels[1], 3, 2, True)
        print('Finished network setup.')
        # classification
        # self.avgpool = nn.AvgPool3d(16,1,0)
        # self.fc = nn.Linear(512, 32)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock, mode="cat"), up_path)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        if self.act == 'relu':
            self.act = (self.act, {"inplace": False})

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        if is_top and self.num_res_units == 0 and self.big_decoder:
            up_conv = Convolution(
                self.dimensions,
                in_channels,
                in_channels // 8,
                strides=strides,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=False,
                is_transposed=True,
                adn_ordering=self.adn_ordering,
            )
            conv = Convolution(
                self.dimensions,
                in_channels // 8,
                in_channels // 16,
                strides=1,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=False,
                is_transposed=True,
                adn_ordering=self.adn_ordering,
            )
            conv_out = Convolution(
                self.dimensions,
                in_channels // 16,
                out_channels,
                strides=1,
                kernel_size=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=True,
                is_transposed=True,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(up_conv, conv, conv_out)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = self.model(x)

        # encoder forward
        features = []
        for layer_id, layer in enumerate(self.encoder_blocks):
            x = layer(x)
            if layer_id < len(self.encoder_blocks) - 1:
                features.append(x)
        features.reverse()

        # decoder forward
        for level, (layer, feature) in enumerate(zip(self.decoder_blocks, features)):
            # last block before output
            # if level == 2:
            #     seg = self.segmentation_decoder[0](torch.cat([x, feature], dim=1))
            # #top decoder level - network output
            # if level == 3:
            #     edt = self.edt_decoder(torch.cat([x, feature], dim=1))
            #     # seeds = self.seeds_decoder(torch.cat([x, feature], dim=1))
            #     # edt_direction = self.direction_decoder(torch.cat([x, feature], dim=1))
            #     seg = self.segmentation_decoder[1](torch.cat([seg, feature], dim=1))
            # else:
            x = torch.cat([x, feature], dim=1)
            x = layer(x)

        # BxCxHxWxD - norm over channel
        # edt_direction = f.normalize(edt_direction, p=2.0, dim=1, eps=1e-12)
        # classification
        # y=self.fc(self.avgpool(features[0]).view(x.size(0), -1))
        # return edt, seg, seeds, edt_direction #, y

        seg_out = torch.sigmoid(x)
        # edt_out = torch.sigmoid(edt)
        return seg_out


if __name__ == "__main__":

    import time
    import numpy as np

    backbone_name = 'resnet18'
    device = "cuda:1"
    model = ResUNet(spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    act='relu',
                    norm='batch',
                    bias=False,
                    backbone_name=backbone_name,
                    big_decoder=True).to(device)
    a = 128
    input = torch.rand(1, 1, a, a, a).to(device)
    print(f"Model input: {input.shape}, encoder name: {backbone_name}, device: {device}.\n")

    time_acc = []
    memory_acc = []
    print("Running benchmark...")
    for i in range(32):
        start = time.time()
        output = model(input)
        t = (time.time() - start) * 1000
        if i > 8:
            time_acc.append(t)
            memory_acc.append(torch.cuda.memory_allocated(device) / 1024 ** 3)

    print(f"Forward pass avg. time: {np.array(time_acc).mean():.3f} ms")
    print(f" - Allocated gpu avg. memory: {np.array(memory_acc).mean():.1f} GB")


