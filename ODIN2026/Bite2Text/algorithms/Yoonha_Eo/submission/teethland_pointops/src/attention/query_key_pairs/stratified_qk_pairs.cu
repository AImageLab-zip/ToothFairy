#include "../../cuda_utils.h"
#include "stratified_qk_pairs.h"
#include "stratified_qk_pairs_cuda.h"


std::tuple<torch::Tensor, torch::Tensor> stratifiedQueryKeyPairs(
    torch::Tensor smallWindowIndices,
    torch::Tensor largeWindowIndices,
    torch::Tensor downsampleIndices,
    const bool mergeDenseAndSparse
) {
    const int numPoints = smallWindowIndices.size(0);

    if (numPoints == 0) {
        torch::Tensor queryKeyPairIndices = torch::empty({2, 0}, torch::device(smallWindowIndices.device()).dtype(torch::kInt64));
        torch::Tensor pairCounts = torch::zeros(2 - mergeDenseAndSparse, torch::device(smallWindowIndices.device()).dtype(torch::kInt32));

        return std::make_tuple(queryKeyPairIndices, pairCounts);
    }


    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> smallTensors = torch::_unique2(
        smallWindowIndices, /*sorted=*/false, /*return_inverse=*/true, /*return_counts=*/true
    );
    const int smallWindowCount = std::get<0>(smallTensors).size(0);
    smallWindowIndices = std::get<1>(smallTensors);
    torch::Tensor smallWindowArgIndices = smallWindowIndices.argsort();
    torch::Tensor smallWindowPointCounts = std::get<2>(smallTensors);
    const int maxSmallWindowPointCount = smallWindowPointCounts.amax().item<int>();
    torch::Tensor smallWindowPointOffsets = smallWindowPointCounts.cumsum(/*dim=*/-1);
    torch::Tensor smallWindowPairOffsets = smallWindowPointCounts.square().cumsum(/*dim=*/-1);
    const int totalSmallWindowPairCount = smallWindowPairOffsets[-1].item<int>();


    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> largeTensors = torch::_unique2(
        largeWindowIndices, /*sorted=*/false, /*return_inverse=*/true, /*return_counts=*/true
    );
    const int largeWindowCount = std::get<0>(largeTensors).size(0);
    largeWindowIndices = std::get<1>(largeTensors);
    torch::Tensor largeWindowDownsampleIndices = largeWindowIndices.index({downsampleIndices});
    torch::Tensor largeWindowDownsampleArgIndices = largeWindowDownsampleIndices.argsort();
    torch::Tensor largeWindowPointCounts = std::get<2>(largeTensors);
    torch::Tensor largeWindowDownsamplePointCounts = torch::bincount(largeWindowDownsampleIndices, /*weights=*/{}, /*minlength=*/largeWindowCount);    
    const int maxLargeWindowDownsamplePointCount = largeWindowDownsamplePointCounts.amax().item<int>();
    torch::Tensor largeWindowDownsamplePointOffsets = largeWindowDownsamplePointCounts.cumsum(/*dim=*/-1);
    torch::Tensor largeWindowPairOffsets = (largeWindowPointCounts * largeWindowDownsamplePointCounts).cumsum(/*dim=*/-1);
    const int totalLargeWindowPairCount = largeWindowPairOffsets[-1].item<int>();


    const int totalWindowCount = smallWindowCount + largeWindowCount;
    torch::Tensor windowPairCount = torch::zeros(totalWindowCount, torch::device(smallWindowIndices.device()).dtype(torch::kInt32));
    const int totalPairCount = totalSmallWindowPairCount + totalLargeWindowPairCount;
    torch::Tensor queryKeyPairIndices = torch::empty({2, totalPairCount}, torch::device(smallWindowIndices.device()).dtype(torch::kInt64));


    const int maxWindowPointCount = max(maxSmallWindowPointCount, maxLargeWindowDownsamplePointCount);
    const int threads = optThreads(maxWindowPointCount);
    stratifiedQueryKeyPairIndices_cuda<<<numPoints, threads>>>(
        smallWindowIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        smallWindowArgIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        largeWindowIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        largeWindowDownsampleArgIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        smallWindowPointOffsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        largeWindowDownsamplePointOffsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        smallWindowPairOffsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        largeWindowPairOffsets.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        downsampleIndices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        windowPairCount.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        queryKeyPairIndices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>()
    );

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    torch::Tensor pairCounts;
    if (mergeDenseAndSparse) {
        queryKeyPairIndices = std::get<0>(torch::unique_dim(queryKeyPairIndices, /*dim=*/1, /*sorted=*/true));
        pairCounts = torch::full(1, queryKeyPairIndices.size(1));
    } else {
        torch::Tensor smallWindowQueryKeyPairIndices = queryKeyPairIndices.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, totalSmallWindowPairCount)});
        smallWindowQueryKeyPairIndices = std::get<0>(torch::unique_dim(smallWindowQueryKeyPairIndices, /*dim=*/1, /*sorted=*/true));
        
        torch::Tensor largeWindowQueryKeyPairIndices = queryKeyPairIndices.index({torch::indexing::Slice(), torch::indexing::Slice(totalSmallWindowPairCount, torch::indexing::None)});
        torch::Tensor diagonal = torch::arange(smallWindowIndices.size(0), torch::device(smallWindowIndices.device()).dtype(torch::kInt64));
        largeWindowQueryKeyPairIndices = torch::column_stack({largeWindowQueryKeyPairIndices, diagonal.expand({2, diagonal.size(0)})});
        largeWindowQueryKeyPairIndices = std::get<0>(torch::unique_dim(largeWindowQueryKeyPairIndices, /*dim=*/1, /*sorted=*/true));

        queryKeyPairIndices = torch::column_stack({smallWindowQueryKeyPairIndices, largeWindowQueryKeyPairIndices});
        int64_t data[] = {smallWindowQueryKeyPairIndices.size(1), largeWindowQueryKeyPairIndices.size(1)};
        pairCounts = torch::from_blob(data, {2}, torch::kInt64);
    }
    pairCounts = pairCounts.to(smallWindowIndices.device(), torch::kInt32);

    return std::make_tuple(queryKeyPairIndices, pairCounts);
}
