#ifndef BALL_QUERY_H
#define BALL_QUERY_H

#include <tuple>

#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor> ballQuery(
    torch::Tensor selfCoordinates,
    torch::Tensor selfBatchCounts,
    torch::Tensor otherCoordinates,
    torch::Tensor otherBatchIndices,
    torch::Tensor ks,
    const float radius
);

#endif