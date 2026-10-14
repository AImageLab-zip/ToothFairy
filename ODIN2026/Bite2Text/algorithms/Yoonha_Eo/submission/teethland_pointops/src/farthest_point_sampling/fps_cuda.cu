#include "fps_cuda.h"


__global__ void farthestPointSampling_cuda(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coordinates,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> batchOffsets,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sampleBatchOffsets,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> seeds,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> distancesToSample,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> sampleIndices
) {
    extern __shared__ int s[];
    int *threadVertexIndices = s;
    float *threadVertexDistances = (float*)&threadVertexIndices[blockDim.x];


    const int stride = blockDim.x;
    const int ti = threadIdx.x;

    const int batchIdx = blockIdx.x;
    const int startIdx = batchIdx == 0 ? 0 : batchOffsets[batchIdx - 1];
    const int endIdx = batchOffsets[batchIdx];
    const int sampleStartIdx = batchIdx == 0 ? 0 : sampleBatchOffsets[batchIdx - 1];
    const int sampleEndIdx = sampleBatchOffsets[batchIdx];


    if (ti == 0) {
        sampleIndices[sampleStartIdx] = seeds[batchIdx];
    }

    int v2i, pow_2;
    float x1, y1, z1, x2, y2, z2, v1v2d;
    for (int j = sampleStartIdx + 1; j < sampleEndIdx; j++)
    {
        __syncthreads();

        x1 = coordinates[sampleIndices[j - 1]][0];
        y1 = coordinates[sampleIndices[j - 1]][1];
        z1 = coordinates[sampleIndices[j - 1]][2];

        threadVertexDistances[ti] = -1;
        for (v2i = startIdx + ti; v2i < endIdx; v2i += stride)
        {
            x2 = coordinates[v2i][0];
            y2 = coordinates[v2i][1];
            z2 = coordinates[v2i][2];

            v1v2d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

            distancesToSample[v2i] = min(v1v2d, distancesToSample[v2i]);
            if (distancesToSample[v2i] > threadVertexDistances[ti]) {
                threadVertexIndices[ti] = v2i;
                threadVertexDistances[ti] = distancesToSample[v2i];
            }
        }


        for (pow_2 = stride >> 1; pow_2 > 0; pow_2 = pow_2 >> 1)
        {
            __syncthreads();
            
            if (
                ti < pow_2
                && threadVertexDistances[ti] < threadVertexDistances[ti + pow_2]
            ) {
                threadVertexIndices[ti] = threadVertexIndices[ti + pow_2];
                threadVertexDistances[ti] = threadVertexDistances[ti + pow_2];
            }
        }

        if (ti == 0) {
            sampleIndices[j] = threadVertexIndices[0];
        }
    }
}
