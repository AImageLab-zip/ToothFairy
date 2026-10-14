#include "../cuda_utils.h"
#include "fps.h"
#include "fps_cuda.h"


std::tuple<torch::Tensor, torch::Tensor> farthestPointSampling(
    torch::Tensor coordinates,
    torch::Tensor batchCounts,
    const float ratio,
    const int maxPoints
) {
    torch::Tensor batchOffsets = batchCounts.cumsum(/*dim=*/-1, /*dtype=*/batchCounts.scalar_type());
    torch::Tensor sampleBatchCounts = torch::ceil(batchCounts * ratio).to(torch::kInt32);
    sampleBatchCounts = torch::minimum(sampleBatchCounts, torch::full({}, maxPoints, coordinates.device()));
    torch::Tensor sampleBatchOffsets = sampleBatchCounts.cumsum(/*dim=*/-1, /*dtype=*/sampleBatchCounts.scalar_type());
    torch::Tensor seeds = batchOffsets - 1 - (batchCounts * torch::rand_like(batchCounts, torch::kFloat32)).to(torch::kInt32);

    torch::Tensor distancesToSample = torch::full(coordinates.size(0), 1e10, coordinates.device());
    torch::Tensor sampleIndices = torch::empty(sampleBatchOffsets[-1].item<int>(), torch::device(coordinates.device()).dtype(torch::kInt64));

    // The FPS kernel's register footprint cannot launch at 1024 threads on
    // Turing (T4). A 256-thread power-of-two block performs the same reduction
    // while remaining compatible with both T4 and Ampere GPUs.
    const int threads = std::min(optThreads(coordinates.size(0)), THREADS_PER_BLOCK);
    farthestPointSampling_cuda<<<batchCounts.size(0), threads, threads * (sizeof(int) + sizeof(float))>>>(
        coordinates.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        batchOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        sampleBatchOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        seeds.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        distancesToSample.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        sampleIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>()
    );

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return std::make_tuple(sampleIndices, sampleBatchCounts);
}
