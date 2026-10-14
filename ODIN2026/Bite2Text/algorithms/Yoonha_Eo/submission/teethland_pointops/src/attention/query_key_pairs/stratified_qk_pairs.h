#ifndef STRATIFIED_QK_PAIRS_H
#define STRATIFIED_QK_PAIRS_H

#include <tuple>

#include <torch/extension.h>


std::tuple<torch::Tensor, torch::Tensor> stratifiedQueryKeyPairs(
    torch::Tensor smallWindowIndices,
    torch::Tensor largeWindowIndices,
    torch::Tensor downsampleIndices,
    const bool mergeDenseAndSparse = true
);


#endif
