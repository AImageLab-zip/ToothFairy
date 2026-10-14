#ifndef ATTENTION_LOGITS_CRPE_H
#define ATTENTION_LOGITS_CRPE_H

#include <vector>

#include <torch/extension.h>


torch::Tensor attentionLogitsCRPE_forward(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> attentionLogitsCRPE_backward(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor attentionLogitsGradient
);


#endif
