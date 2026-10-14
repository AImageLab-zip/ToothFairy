#ifndef FPS_CUDA_H
#define FPS_CUDA_H

#include <torch/extension.h>


__global__ void farthestPointSampling_cuda(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coordinates,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> batchOffsets,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sampleBatchOffsets,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> seeds,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> distancesToSample,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sampleIndices
);


#endif
