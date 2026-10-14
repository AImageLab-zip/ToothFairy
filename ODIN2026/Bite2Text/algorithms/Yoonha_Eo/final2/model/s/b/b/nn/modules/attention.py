import math
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
from torchtyping import TensorType

from teethland import PointTensor
from teethland.nn.modules.linear import Linear
import teethland.nn.functional as F

from pointops import stratifiedQueryKeyPairs


class CRPEAttention(nn.Module):
    """Implements attention with contextual relative position encoding."""

    def __init__(
        self,
        in_channels: int,
        heads: int,
        head_channels: int,
        bins: int,
        bin_size: float,
    ):
        super().__init__()

        self.qkv = Linear(in_channels, 3 * heads * head_channels, bias=True)

        self.query_rel_xyz_tables = torch.empty(3, bins, heads, head_channels)
        self.query_rel_xyz_tables = nn.Parameter(self.query_rel_xyz_tables)
        nn.init.trunc_normal_(self.query_rel_xyz_tables, std=0.02)

        self.key_rel_xyz_tables = torch.empty(3, bins, heads, head_channels)
        self.key_rel_xyz_tables = nn.Parameter(self.key_rel_xyz_tables)
        nn.init.trunc_normal_(self.key_rel_xyz_tables, std=0.02)

        self.value_rel_xyz_tables = torch.empty(3, bins, heads, head_channels)
        self.value_rel_xyz_tables = nn.Parameter(self.value_rel_xyz_tables)
        nn.init.trunc_normal_(self.value_rel_xyz_tables, std=0.02)        

        self.heads = heads
        self.head_channels = head_channels
        self.query_scale = 1 / math.sqrt(head_channels)
        self.bins = bins
        self.bin_size = bin_size

    def relative_xyz_table_indices_cache_key(
        self,
        x: PointTensor,
        qk_pair_idxs: TensorType[2, 'M', torch.int64],
    ) -> str:
        try:
            key = f'{x.cache.key(qk_pair_idxs)}__'
        except ValueError:
            key = ''

        return key + '__'.join([
            'rel_xyz_table_idxs',
            f'bins={self.bins}',
            f'bin_size={self.bin_size:.4f}',
        ])

    def relative_xyz_table_indices(
        self,
        x: PointTensor,
        qk_pair_idxs: TensorType[2, 'M', torch.int64],
    ) -> TensorType["M", 3, torch.int32]:
        cache_key = self.relative_xyz_table_indices_cache_key(x, qk_pair_idxs)
        if cache_key in x.cache:
            return x.cache[cache_key]

        query_idxs, key_idxs = qk_pair_idxs
        rel_xyzs = x.C[query_idxs] - x.C[key_idxs]
        rel_xyzs /= self.bin_size
        rel_xyzs += self.bins / 2
        rel_xyz_table_idxs = torch.clip(
            input=rel_xyzs.int(),
            min=0,
            max=self.bins - 1,
        )

        x.cache[cache_key] = rel_xyz_table_idxs

        return rel_xyz_table_idxs

    def forward(
        self,
        x: PointTensor,
        qk_pair_idxs: TensorType[2, 'M', torch.int64],
    ) -> PointTensor:
        # compute queries, keys, and values
        qkv = self.qkv(x).F
        qkv = qkv.reshape(-1, 3, self.heads, self.head_channels)
        queries, keys, values = qkv.transpose(0, 1)

        # compute indices into cRPE look-up tables
        rel_xyz_table_idxs = self.relative_xyz_table_indices(x, qk_pair_idxs)

        # compute attention logits
        attention_logits = F.attention_logits_crpe(
            queries * self.query_scale,
            keys,
            qk_pair_idxs,
            self.query_rel_xyz_tables,
            self.key_rel_xyz_tables,
            rel_xyz_table_idxs
        )
        
        # compute attention distribution for each query and head
        attention_distrs = scatter_softmax(
            src=attention_logits,
            index=qk_pair_idxs[0],
            dim=0,
        )

        # aggregate values given attention distributions
        agg_values = F.aggregate_values_crpe(
            values,
            qk_pair_idxs,
            attention_distrs,
            self.value_rel_xyz_tables,
            rel_xyz_table_idxs,
        )

        # project aggregated values to final output
        agg_values = agg_values.reshape(-1, self.heads * self.head_channels)
        x = x.new_tensor(features=agg_values)

        return x


class StratifiedCRPEAttention(nn.Module):
    """Implements stratified window-based multi-head self-attention (SW-MSA)."""

    @staticmethod
    def factory(
        union: bool,
        **kwargs: Dict[str, Any],
    ):
        if union:
            return DenseAndSparseStratifiedCRPEAttention(**kwargs)
        else:
            return DenseOrSparseStratifiedCRPEAttention(**kwargs)
    
    def __init__(
        self,
        out_channels: int,
        heads: int,
        window_size: float,
        shifted: bool,
        downsample_ratio: float,
    ) -> None:
        super().__init__()
        
        head_channels = math.ceil(out_channels / heads)
        self.projection = Linear(heads * head_channels, out_channels, bias=True)

        self.head_channels = head_channels
        self.small_window_size = window_size
        self.large_window_size = 2 * window_size
        self.shifted = shifted
        self.downsample_ratio = downsample_ratio    

    def _cache_key(self, name: str) -> str:
        return '__'.join([
            name,
            f'small_window_size={self.small_window_size:.4f}',
            f'large_window_size={self.large_window_size:.4f}',
            f'shifted={self.shifted}',
            f'downsample_ratio={self.downsample_ratio:.4f}',
        ])

    def query_key_pair_indices(
        self,
        x: PointTensor,
    ) -> Tuple[
        TensorType[2, 'M', torch.int64],
        TensorType['1 or 2', torch.int32],
    ]:
        small_window_idxs = x.gridsample(
            voxel_size=self.small_window_size,
            start=-self.small_window_size / 2 if self.shifted else 0,
        )

        large_window_idxs = x.gridsample(
            voxel_size=self.large_window_size,
            start=-self.large_window_size / 2 if self.shifted else 0,
        )

        downsample_idxs = x.farthest_point_sampling(
            ratio=self.downsample_ratio,
        )

        qk_pair_idxs, qk_pair_counts = stratifiedQueryKeyPairs(
            small_window_idxs,
            large_window_idxs,
            downsample_idxs,
            union=isinstance(self, DenseAndSparseStratifiedCRPEAttention),
        )

        return qk_pair_idxs, qk_pair_counts


class DenseAndSparseStratifiedCRPEAttention(StratifiedCRPEAttention):
    """Implements stratified window-based multi-head self-attention (SW-MSA)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        window_size: float,
        shifted: bool,
        downsample_ratio: float,
        crpe_bins: int,
    ):
        super().__init__(
            out_channels=out_channels,
            heads=heads,
            window_size=window_size,
            shifted=shifted,
            downsample_ratio=downsample_ratio,
        )

        self.attention = CRPEAttention(
            in_channels=in_channels,
            heads=heads,
            head_channels=self.head_channels,
            bins=crpe_bins,
            bin_size=2 * self.large_window_size / crpe_bins,
        )

        self.cache_key = self._cache_key('qk_pair_idxs')

    def query_key_pair_indices(
        self,
        x: PointTensor,
    ) -> TensorType[2, 'M', torch.int64]:
        if self.cache_key in x.cache:
            return x.cache[self.cache_key]

        qk_pair_idxs, _ = super().query_key_pair_indices(x)

        x.cache[self.cache_key] = qk_pair_idxs

        return qk_pair_idxs

    def forward(
        self,
        x: PointTensor,
    ) -> PointTensor:
        qk_pair_idxs = self.query_key_pair_indices(x)

        x = self.attention(x, qk_pair_idxs)
        x = self.projection(x)

        return x


class DenseOrSparseStratifiedCRPEAttention(StratifiedCRPEAttention):
    """Implements stratified window-based multi-head self-attention (SW-MSA)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        window_size: float,
        shifted: bool,
        downsample_ratio: float,
        crpe_bins: int,
    ):
        super().__init__(
            out_channels=out_channels,
            heads=heads,
            window_size=window_size,
            shifted=shifted,
            downsample_ratio=downsample_ratio,
        )

        self.dense_attention = CRPEAttention(
            in_channels=in_channels,
            heads=math.ceil(heads / 2),
            head_channels=self.head_channels,
            bins=crpe_bins,
            bin_size=2 * self.small_window_size / crpe_bins,
        )
        self.sparse_attention = CRPEAttention(
            in_channels=in_channels,
            heads=math.floor(heads / 2),
            head_channels=self.head_channels,
            bins=crpe_bins,
            bin_size=2 * self.large_window_size / crpe_bins,
        )

        self.cache_keys = (
            self._cache_key('dense_qk_pair_idxs'),
            self._cache_key('sparse_qk_pair_idxs'),
        )

    def query_key_pair_indices(
        self,
        x: PointTensor,
    ) -> Tuple[
        TensorType[2, 'M', torch.int64],
        TensorType[2, 'M', torch.int64],
    ]:
        if all(k in x.cache for k in self.cache_keys):
            return tuple(x.cache[k] for k in self.cache_keys)

        qk_pair_idxs, qk_pair_counts = super().query_key_pair_indices(x)
        dense_qk_pair_idxs = qk_pair_idxs[:, :qk_pair_counts[0]]
        sparse_qk_pair_idxs = qk_pair_idxs[:, qk_pair_counts[0]:]

        x.cache[self.cache_keys[0]] = dense_qk_pair_idxs
        x.cache[self.cache_keys[1]] = sparse_qk_pair_idxs

        return dense_qk_pair_idxs, sparse_qk_pair_idxs

    def forward(
        self,
        x: PointTensor,
    ) -> PointTensor:
        dense_qk_pair_idxs, sparse_qk_pair_idxs = self.query_key_pair_indices(x)

        x_dense = self.dense_attention(x, dense_qk_pair_idxs)
        x_sparse = self.sparse_attention(x, sparse_qk_pair_idxs)

        x = x.new_tensor(
            features=torch.column_stack((x_dense.F, x_sparse.F)),
        )
        x = self.projection(x)

        return x
