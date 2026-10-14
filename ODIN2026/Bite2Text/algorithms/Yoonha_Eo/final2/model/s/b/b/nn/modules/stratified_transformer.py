from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from teethland import PointTensor
from teethland.nn.modules.attention import StratifiedCRPEAttention
from teethland.nn.modules.downsample import DownsampleBlock
from teethland.nn.modules.dropout import DropPath
from teethland.nn.modules.feedforward import MLP
from teethland.nn.modules.kpconv import KPConv, KPConvResidualBlock
from teethland.nn.modules.linear import Linear
from teethland.nn.modules.normalization import LayerNorm
from teethland.nn.modules.upsample import UpsampleBlock


class StratifiedTransformer(nn.Module):
    """
    Implements stratified transformer from 'Stratified Transformer for 3D Point
    Cloud Segmentation', accepted at CVPR 2022.

    [1]: https://arxiv.org/abs/2203.14508
    [2]: https://github.com/dvlab-research/Stratified-Transformer
    """

    def __init__(
        self,
        in_channels: int,
        channels_list: List[int],
        out_channels: Union[List[int], int],
        depths: List[int],
        heads_list: List[int],
        window_sizes: List[int],
        point_embedding: Dict[str, Any],
        stratified_union: bool,
        downsample_ratio: float,
        max_drop_path_prob: float,
        stratified_downsample_ratio: float,
        crpe_bins: int,
        transformer_lr_ratio: float,
        **kwargs,
    ):
        super().__init__()

        channels_list = [in_channels, *channels_list]
        self.point_embedding = self.init_point_embedding(
            in_channels=channels_list[0],
            out_channels=channels_list[1],
            **{k: v for k, v in point_embedding.items() if k != 'use'},
        ) if point_embedding['use'] else nn.Identity()
        
        self.transformers = self.init_transformers(
            channels_list=channels_list[2 * point_embedding['use']:],
            depths=depths,
            heads_list=heads_list,
            window_sizes=window_sizes,
            stratified_union=stratified_union,            
            max_drop_path_prob=max_drop_path_prob,
            stratified_downsample_ratio=stratified_downsample_ratio,            
            crpe_bins=crpe_bins,
        )

        self.downsample_blocks = self.init_downsample_blocks(
            channels_list[point_embedding['use']:], downsample_ratio,
        )

        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        self.upsample_blocks, self.heads = nn.ModuleList(), nn.ModuleList()
        for channels in out_channels:
            self.upsample_blocks.append(
                self.init_upsample_blocks(
                    channels_list[point_embedding['use']:],
                ),
            )
            self.heads.append(Linear(
                in_channels=channels_list[point_embedding['use']],
                out_channels=channels,
                bias=True,
            ) if channels is not None else nn.Identity())

        self.enc_channels = channels_list[-1]
        self.out_channels = [
            channels_list[point_embedding['use']] if chs is None else chs
            for chs in out_channels
        ]
        self.use_point_embedding = point_embedding['use']
        self.transformer_lr_ratio = transformer_lr_ratio

    def init_point_embedding(
        self,
        in_channels: int,
        out_channels: int,
        kpconv_point_influence: float,
        kpconv_ball_radius: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            KPConv(
                in_channels=in_channels,
                out_channels=out_channels,
                point_influence=kpconv_point_influence,
                ball_radius=kpconv_ball_radius,
            ),
            KPConvResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                point_influence=kpconv_point_influence,
                ball_radius=kpconv_ball_radius,
            ),
        )

    def init_transformers(
        self,
        channels_list: List[int],
        depths: List[int],
        heads_list: List[int],
        window_sizes: List[float],
        stratified_union: bool,
        max_drop_path_prob: float,
        stratified_downsample_ratio: float,
        crpe_bins: int,
    ) -> nn.ModuleList:    
        drop_path_probs = torch.linspace(0, max_drop_path_prob, sum(depths))
        current_block = 0

        transformers = nn.ModuleList()
        for i in range(len(channels_list)):
            transformer = nn.Sequential()
            for block_idx in range(depths[i]):
                transformer.append(StratifiedTransformerBlock(
                    channels=channels_list[i],
                    heads=heads_list[i],
                    window_size=window_sizes[i],
                    union=stratified_union,
                    shifted=block_idx % 2 == 1,
                    drop_path_prob=drop_path_probs[current_block].item(),
                    downsample_ratio=stratified_downsample_ratio,
                    crpe_bins=crpe_bins,
                ))

                current_block += 1

            transformers.append(transformer)

        return transformers

    def init_downsample_blocks(
        self,
        channels_list: List[int],
        downsample_ratio: float,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            blocks.append(DownsampleBlock(
                in_channels=channels_list[i],
                out_channels=channels_list[i + 1],
                ratio=downsample_ratio,
            ))

        return blocks

    def init_upsample_blocks(
        self,
        channels_list: List[int],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1, 0, -1):
            blocks.append(UpsampleBlock(
                in_channels=channels_list[i],
                out_channels=channels_list[i - 1],
            ))
        
        return blocks

    def param_groups(
        self,
        lr: float,
    ) -> List[Dict[str, Any]]:
        return [
            {
                'params': self.point_embedding.parameters(),
                'lr': lr,
                'name': 'point_embedding',
            },
            {
                'params': self.transformers.parameters(),
                'lr': lr * self.transformer_lr_ratio,
                'name': 'transformers',
            },
            {
                'params': self.downsample_blocks.parameters(),
                'lr': lr,
                'name': 'downsample_blocks',
            },
            {
                'params': self.upsample_blocks.parameters(),
                'lr': lr,
                'name': 'upsample_blocks',
            },
            {
                'params': self.heads.parameters(),
                'lr': lr,
                'name': 'heads',
            },
        ]

    def forward(self, x: PointTensor) -> Union[
        Tuple[PointTensor, PointTensor],
        Tuple[PointTensor, List[PointTensor]],
    ]:
        if self.use_point_embedding:
            # local aggregation to bootstrap attention mechanism
            x = self.point_embedding(x)
        else:
            # first apply transformer before downsampling
            transformer = self.transformers[0]
            x = transformer(x)
        
        # apply downsampling blocks and transformers to encode input
        x_ups = []
        blocks = (
            self.downsample_blocks,
            self.transformers[not self.use_point_embedding:],
        )
        for downsample_block, transformer in zip(*blocks):
            x_ups.insert(0, x)
            x = downsample_block(x)
            x = transformer(x)

        # save encoded input for further processing
        encoding = x

        # apply upsampling blocks with skip connections and heads
        outs = []
        for upsample_blocks, head in zip(self.upsample_blocks, self.heads):
            x = encoding
            for upsample_block, x_up in zip(upsample_blocks, x_ups):
                x = upsample_block(x, x_up)
            x = head(x)

            outs.append(x)

        if len(outs) == 1:
            return encoding, outs[0]

        return encoding, outs


class StratifiedTransformerBlock(nn.Module):
    """Block with stratified window-based multi-head self-attention (SW-MSA)."""

    def __init__(
        self,
        channels: int,
        heads: int,
        window_size: float,
        union: bool,
        shifted: bool,
        drop_path_prob: float,
        downsample_ratio: float,
        crpe_bins: int,
    ):
        super().__init__()

        self.norm1 = LayerNorm(channels)
        self.attention = StratifiedCRPEAttention.factory(
            in_channels=channels,
            out_channels=channels,
            heads=heads,
            window_size=window_size,
            union=union,
            shifted=shifted,
            downsample_ratio=downsample_ratio,
            crpe_bins=crpe_bins,
        )

        self.norm2 = LayerNorm(channels)
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=4 * channels,
            out_channels=channels,
        )
        
        if drop_path_prob:
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x: PointTensor) -> PointTensor:
        x_res = self.norm1(x)
        x_res = self.attention(x_res)
        x = x + self.drop_path(x_res)

        x_res = self.norm2(x)
        x_res = self.mlp(x_res)
        x = x + self.drop_path(x_res)

        return x
