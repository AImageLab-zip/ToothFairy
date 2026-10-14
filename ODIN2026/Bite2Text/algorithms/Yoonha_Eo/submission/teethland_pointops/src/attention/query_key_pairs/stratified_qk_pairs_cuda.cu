#include "stratified_qk_pairs_cuda.h"


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
) {
    const int stride = blockDim.x;
    const int ti = threadIdx.x;  // thread index
    const int smallWindowCount = smallWindowPointOffsets.size(0);
    const int totalSmallWindowPairCount = smallWindowPairOffsets[smallWindowCount - 1];
    

    int qi = blockIdx.x;  // index of query point in original point cloud
    int wi = smallWindowIndices[qi];
    int startPointIdx = wi == 0 ? 0 : smallWindowPointOffsets[wi - 1];
    int endPointIdx = smallWindowPointOffsets[wi];
    int startPairIdx = wi == 0 ? 0 : smallWindowPairOffsets[wi - 1];

    int wki;  // index of key point within window
    int ki;  // index of key point in original point cloud
    int qki;  // index of query-key pair
    for (wki = startPointIdx + ti; wki < endPointIdx; wki += stride) {
        ki = smallWindowArgIndices[wki];

        qki = startPairIdx + atomicAdd(&windowPairCount[wi], 1);
        queryKeyPairIndices[0][qki] = qi;
        queryKeyPairIndices[1][qki] = ki;
    }


    wi = largeWindowIndices[qi];
    startPointIdx = wi == 0 ? 0 : largeWindowDownsamplePointOffsets[wi - 1];
    endPointIdx = largeWindowDownsamplePointOffsets[wi];
    startPairIdx = wi == 0 ? 0 : largeWindowPairOffsets[wi - 1];

    // get unique indices
    wi += smallWindowCount;
    startPairIdx += totalSmallWindowPairCount;

    for (wki = startPointIdx + ti; wki < endPointIdx; wki += stride) {
        ki = largeWindowDownsampleArgIndices[wki];
        ki = downsamplePointIdxToPointIdx[ki];

        qki = startPairIdx + atomicAdd(&windowPairCount[wi], 1);
        queryKeyPairIndices[0][qki] = qi;
        queryKeyPairIndices[1][qki] = ki;
    }
}
