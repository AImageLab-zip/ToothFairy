#ifndef KNN_QUERY_H
#define KNN_QUERY_H

#include <tuple>

#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor> kNNQuery(
    torch::Tensor selfCoordinates,
    torch::Tensor selfBatchCounts,
    torch::Tensor otherCoordinates,
    torch::Tensor otherBatchIndices,
    torch::Tensor ks,
    const bool sorted = true
);


#endif
