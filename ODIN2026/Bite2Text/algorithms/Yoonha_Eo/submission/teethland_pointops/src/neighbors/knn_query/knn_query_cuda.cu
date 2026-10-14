#include "knn_query.h"


template <typename scalar_t>
__device__ void swap(
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> scalars,
    const int idx1, const int idx2
) {
    scalar_t tmp = scalars[idx1];
    scalars[idx1] = scalars[idx2];
    scalars[idx2] = tmp;
}


__device__ void reheap(
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int startIdx, const int endIdx
) {
    int root = startIdx, child = startIdx + 1;
    while (child < endIdx)
    {
        if (
            child + 1 < endIdx
            && values[child + 1] > values[child]
        )
            child++;

        if (values[root] > values[child])
            return;

        swap<int64_t>(indices, root, child);
        swap<float>(values, root, child);
        root = child;
        child = startIdx + ((root - startIdx) << 1) + 1;
    }
}


__device__ void heapSort(    
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> values,
    const int startIdx, const int endIdx
) {
    for (int i = endIdx - 1; i > startIdx; i--)
    {
        swap<int64_t>(indices, startIdx, i);
        swap<float>(values, startIdx, i);
        reheap(indices, values, startIdx, i);
    }
}


__global__ void kNNQuery_cuda(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> selfCoordinates,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> otherCoordinates,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> selfBatchOffsets,
    const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> otherBatchIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> neighborOffsets,
    const bool sorted,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sampleIndices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sampleDistances
) {
    const int otherIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (otherIdx >= otherCoordinates.size(0)) return;

    const int batchIdx = otherBatchIndices[otherIdx];
    const int selfStartIdx = batchIdx == 0 ? 0 : selfBatchOffsets[batchIdx - 1];
    const int selfEndIdx = selfBatchOffsets[batchIdx];
    const int neighborStartIdx = otherIdx == 0 ? 0 : neighborOffsets[otherIdx - 1];
    const int neighborEndIdx = neighborOffsets[otherIdx];

    const float x1 = otherCoordinates[otherIdx][0];
    const float y1 = otherCoordinates[otherIdx][1];
    const float z1 = otherCoordinates[otherIdx][2];

    float x2, y2, z2, v1v2d;
    for(int i = selfStartIdx; i < selfEndIdx; i++){
        x2 = selfCoordinates[i][0];
        y2 = selfCoordinates[i][1];
        z2 = selfCoordinates[i][2];
        
        v1v2d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        if (v1v2d < sampleDistances[neighborStartIdx]) {
            sampleDistances[neighborStartIdx] = v1v2d;
            sampleIndices[neighborStartIdx] = i;
            reheap(sampleIndices, sampleDistances, neighborStartIdx, neighborEndIdx);
        }
    }

    if (sorted) {
        heapSort(sampleIndices, sampleDistances, neighborStartIdx, neighborEndIdx);
    }
}
