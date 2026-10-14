#ifndef STRATIFIED_QK_PAIRS_CUDA_H
#define STRATIFIED_QK_PAIRS_CUDA_H

#include <torch/extension.h>


__global__ void stratifiedQueryKeyPairIndices_cuda(
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> smallWindowIndices,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> smallWindowArgIndices,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> largeWindowIndices,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> largeWindowDownsampleArgIndices,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> smallWindowPointOffsets,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> largeWindowDownsamplePointOffsets,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> smallWindowPairOffsets,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> largeWindowPairOffsets,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> downsamplePointIdxToPointIdx,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> windowPairCount,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices
);


#endif
