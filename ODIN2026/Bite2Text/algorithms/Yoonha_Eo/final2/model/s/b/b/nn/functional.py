from typing import Tuple

import torch
from torch.autograd.function import Function, FunctionCtx
from torchtyping import TensorType

from pointops import (
    attentionLogitsCRPE_forward,
    attentionLogitsCRPE_backward,
    aggregateValuesCRPE_forward,
    aggregateValuesCRPE_backward,
)


class CRPEAttentionLogits(Function):
    """Computes attention logits using contextual relative position encoding."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        queries: TensorType['N', 'heads', 'head_channels', torch.float32],
        keys: TensorType['N', 'heads', 'head_channels', torch.float32],
        query_key_pair_idxs: TensorType[2, 'M', torch.int64],
        query_rel_xyz_tables: TensorType[
            3, 'bins', 'heads', 'head_channels', torch.float32,
        ],
        key_rel_xyz_tables: TensorType[
            3, 'bins', 'heads', 'head_channels', torch.float32,
        ],
        rel_xyz_table_idxs: TensorType['M', 3, torch.int32],
    ) -> TensorType['M', 'heads', torch.float32]:
        attention_logits = attentionLogitsCRPE_forward(
            queries,
            keys,
            query_key_pair_idxs,
            query_rel_xyz_tables,
            key_rel_xyz_tables,
            rel_xyz_table_idxs,
        )

        ctx.save_for_backward(
            queries,
            keys,
            query_key_pair_idxs,
            query_rel_xyz_tables,
            key_rel_xyz_tables,
            rel_xyz_table_idxs,
        )

        return attention_logits

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        attention_logits_grad: TensorType['M', 'heads', torch.float32],
    ) -> Tuple[
        TensorType['N', 'heads', 'head_channels', torch.float32],
        TensorType['N', 'heads', 'head_channels', torch.float32],
        None,
        TensorType[3, 'bins', 'heads', 'head_channels', torch.float32],
        TensorType[3, 'bins', 'heads', 'head_channels', torch.float32],
        None,
    ]:
        (
            queries_grad,
            keys_grad,
            query_rel_xyz_tables_grad,
            key_rel_xyz_tables_grad,
        ) = attentionLogitsCRPE_backward(
            *ctx.saved_tensors,
            attention_logits_grad,
        )

        return (
            queries_grad,
            keys_grad,
            None,
            query_rel_xyz_tables_grad,
            key_rel_xyz_tables_grad,
            None,
        )

attention_logits_crpe = CRPEAttentionLogits.apply


class CRPEAggregateValues(Function):
    """Aggregates values using contextual relative position encoding."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        values: TensorType['N', 'heads', 'head_channels', torch.float32],
        query_key_pair_idxs: TensorType[2, 'M', torch.int64],
        attention_distrs: TensorType['M', 'heads', torch.float32],
        values_rel_xyz_tables: TensorType[
            3, 'bins', 'heads', 'head_channels', torch.float32,
        ],
        rel_xyz_table_idxs: TensorType['M', 3, torch.int32],
    ) -> TensorType['N', 'heads', 'head_channels', torch.float32]:        
        _, query_key_counts = torch.unique_consecutive(
            query_key_pair_idxs[0], return_counts=True,
        )
        query_key_offsets = query_key_counts.cumsum(dim=-1, dtype=torch.int32)
        max_key_count = query_key_counts.amax().item()

        agg_values = aggregateValuesCRPE_forward(
            values,
            query_key_pair_idxs,
            query_key_offsets,
            max_key_count,
            attention_distrs,
            values_rel_xyz_tables,
            rel_xyz_table_idxs,
        )

        ctx.query_key_offsets = query_key_offsets
        ctx.max_key_count = max_key_count
        ctx.save_for_backward(
            values,
            query_key_pair_idxs,
            attention_distrs,
            values_rel_xyz_tables,
            rel_xyz_table_idxs,
        )

        return agg_values

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        agg_values_grad: TensorType[
            'N', 'heads', 'head_channels', torch.float32,
        ],
    ) -> Tuple[
        TensorType['N', 'heads', 'head_channels', torch.float32],
        None,
        TensorType['M', 'heads', torch.float32],
        TensorType[3, 'bins', 'heads', 'head_channels', torch.float32],
        None,
    ]:
        (
            values,
            query_key_pair_idxs,
            attention_distrs,
            values_rel_xyz_tables,
            rel_xyz_table_idxs,
        ) = ctx.saved_tensors
        
        (
            values_grad,
            attention_distrs_grad,
            values_rel_xyz_tables_grad,
        ) = aggregateValuesCRPE_backward(
            values,
            query_key_pair_idxs,
            ctx.query_key_offsets,
            ctx.max_key_count,
            attention_distrs,
            values_rel_xyz_tables,
            rel_xyz_table_idxs,
            agg_values_grad,
        )

        return (
            values_grad,
            None,
            attention_distrs_grad,
            values_rel_xyz_tables_grad,
            None,
        )

aggregate_values_crpe = CRPEAggregateValues.apply
