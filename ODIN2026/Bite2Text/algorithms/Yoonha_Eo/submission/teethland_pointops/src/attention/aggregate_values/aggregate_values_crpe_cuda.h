#ifndef AGGREGATE_VALUES_CRPE_CUDA_H
#define AGGREGATE_VALUES_CRPE_CUDA_H

#include <torch/extension.h>


void aggregateValuesCRPE_forward_launcher(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor aggregatedValues
);


template <typename scalar_t>
__global__ void aggregateValuesCRPE_forward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> values,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> queryKeyOffsets,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionDistributions,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> valueRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aggregatedValues
);


void aggregateValuesCRPE_backward_launcher(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor aggregatedValuesGradient,
    torch::Tensor valuesGradient,
    torch::Tensor attentionDistributionsGradient,
    torch::Tensor valueRelativeXYZTablesGradient
);


template <typename scalar_t>
__global__ void aggregateValuesCRPE_backward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> values,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> queryKeyOffsets,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionDistributions,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> valueRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aggregatedValuesGradient,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> valuesGradient,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionDistributionsGradient,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> valueRelativeXYZTablesGradient
);


#endif
