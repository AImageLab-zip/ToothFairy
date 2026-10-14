#include "../../cuda_utils.h"
#include "knn_query.h"
#include "knn_query_cuda.h"


std::tuple<torch::Tensor, torch::Tensor> kNNQuery(
    torch::Tensor selfCoordinates,
    torch::Tensor selfBatchCounts,
    torch::Tensor otherCoordinates,
    torch::Tensor otherBatchIndices,
    torch::Tensor ks,
    const bool sorted
) {
    torch::Tensor selfBatchOffsets = selfBatchCounts.cumsum(/*dim=*/-1, /*dtype=*/selfBatchCounts.scalar_type());
    torch::Tensor neighborOffsets = ks.cumsum(/*dim=*/-1, /*dtype=*/ks.scalar_type());
    const int totalNeighbors = neighborOffsets[-1].item<int>();

    torch::Tensor sampleIndices = torch::empty(totalNeighbors, torch::device(otherCoordinates.device()).dtype(torch::kInt64));
    torch::Tensor sampleDistances = torch::full(totalNeighbors, 1e10, otherCoordinates.device());

    const int blocks = DIVUP(otherCoordinates.size(0), THREADS_PER_BLOCK);
    kNNQuery_cuda<<<blocks, THREADS_PER_BLOCK>>>(
        selfCoordinates.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        otherCoordinates.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        selfBatchOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        otherBatchIndices.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
        neighborOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        sorted,
        sampleIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        sampleDistances.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
    );

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return std::make_tuple(sampleIndices, sampleDistances);
}
