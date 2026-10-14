#ifndef KNN_QUERY_CUDA_H
#define KNN_QUERY_CUDA_H

#include <torch/extension.h>


template <typename scalar_t>
__device__ void swap(
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> scalars,
    const int idx1, const int idx2
);


__device__ void reheap(
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int startIdx, const int endIdx
);


__device__ void heapSort(    
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int startIdx, const int endIdx
);


__global__ void kNNQuery_cuda(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> selfCoordinates,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> otherCoordinates,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> selfBatchOffsets,
    const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> otherBatchIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> neighborOffsets,
    const bool sorted,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sampleIndices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sampleDistances
);


#endif
