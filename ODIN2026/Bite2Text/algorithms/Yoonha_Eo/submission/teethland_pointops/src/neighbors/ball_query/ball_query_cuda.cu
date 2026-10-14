#include "ball_query_cuda.h"


__global__ void ballQuery_cuda(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> selfCoordinates,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> otherCoordinates,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> selfBatchOffsets,
    const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> otherBatchIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> neighborOffsets,
    const float radius,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sampleIndices,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> sampleDistances
) {
    const int otherIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (otherIdx >= otherCoordinates.size(0)) return;

    const int batchIdx = otherBatchIndices[otherIdx];
    const int selfStartIdx = batchIdx == 0 ? 0 : selfBatchOffsets[batchIdx - 1];
    const int selfEndIdx = selfBatchOffsets[batchIdx];
    int neighborIdx = otherIdx == 0 ? 0 : neighborOffsets[otherIdx - 1];
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
        if (v1v2d <= radius) {
            sampleIndices[neighborIdx] = i;
            sampleDistances[neighborIdx] = v1v2d;
            
            if (++neighborIdx == neighborEndIdx) break;
        }
    }
}
