#ifndef AGGREGATE_VALUES_CRPE_H
#define AGGREGATE_VALUES_CRPE_H

#include <tuple>

#include <torch/extension.h>


torch::Tensor aggregateValuesCRPE_forward(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> aggregateValuesCRPE_backward(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor aggregatedValuesGradient
);


#endif
