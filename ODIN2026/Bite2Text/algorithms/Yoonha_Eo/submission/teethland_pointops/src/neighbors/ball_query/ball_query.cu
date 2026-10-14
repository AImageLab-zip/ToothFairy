#include "../../cuda_utils.h"
#include "ball_query.h"
#include "ball_query_cuda.h"


std::tuple<torch::Tensor, torch::Tensor> ballQuery(
    torch::Tensor selfCoordinates,
    torch::Tensor selfBatchCounts,
    torch::Tensor otherCoordinates,
    torch::Tensor otherBatchIndices,
    torch::Tensor ks,
    const float radius
) {
    torch::Tensor selfBatchOffsets = selfBatchCounts.cumsum(/*dim=*/-1, /*dtype=*/selfBatchCounts.scalar_type());
    torch::Tensor neighborOffsets = ks.cumsum(/*dim=*/-1, /*dtype=*/ks.scalar_type());
    const int totalNeighbors = neighborOffsets[-1].item<int>();
    
    torch::Tensor sampleIndices = torch::full(totalNeighbors, -1, torch::device(selfCoordinates.device()).dtype(torch::kInt64));
    torch::Tensor sampleDistances = torch::empty(totalNeighbors, selfCoordinates.device());

    const int blocks = DIVUP(otherCoordinates.size(0), THREADS_PER_BLOCK);
    ballQuery_cuda<<<blocks, THREADS_PER_BLOCK>>>(
        selfCoordinates.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        otherCoordinates.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        selfBatchOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        otherBatchIndices.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
        neighborOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        radius * radius,
        sampleIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        sampleDistances.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
    );

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return std::make_tuple(sampleIndices, sampleDistances);
}
