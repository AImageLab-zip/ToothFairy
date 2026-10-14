#ifndef BALL_QUERY_CUDA_H
#define BALL_QUERY_CUDA_H

#include <torch/extension.h>


__global__ void ballQuery_cuda(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> selfCoordinates,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> otherCoordinates,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> selfBatchOffsets,
    const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> otherBatchIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> neighborOffsets,
    const float radius,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sampleIndices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sampleDistances
);


#endif
