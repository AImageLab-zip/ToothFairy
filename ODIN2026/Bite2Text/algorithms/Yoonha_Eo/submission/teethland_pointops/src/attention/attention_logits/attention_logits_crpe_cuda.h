#ifndef ATTENTION_LOGITS_CRPE_CUDA_H
#define ATTENTION_LOGITS_CRPE_CUDA_H

#include <torch/extension.h>


void attentionLogitsCRPE_forward_launcher(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor attentionLogits
);


template <typename scalar_t>
__global__ void attentionLogitsCRPE_forward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> keys,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queryRelativeXYZTables,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keyRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionLogits
);


void attentionLogitsCRPE_backward_launcher(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor attentionLogitsGradient,
    torch::Tensor queriesGradient,
    torch::Tensor keysGradient,
    torch::Tensor queryRelativeXYZTablesGradient,
    torch::Tensor keyRelativeXYZTablesGradient
);


template <typename scalar_t>
__global__ void attentionLogitsCRPE_backward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> keys,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queryRelativeXYZTables,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keyRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionLogitsGradient,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queriesGradient,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> keysGradient,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queryRelativeXYZTablesGradient,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keyRelativeXYZTablesGradient
);


#endif
