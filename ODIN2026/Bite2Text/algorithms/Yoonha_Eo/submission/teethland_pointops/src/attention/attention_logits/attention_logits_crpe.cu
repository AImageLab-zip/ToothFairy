#include "../../cuda_utils.h"
#include "attention_logits_crpe.h"
#include "attention_logits_crpe_cuda.h"


torch::Tensor attentionLogitsCRPE_forward(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices
) {
    const int numHeads = queries.size(1);
    const int numQueryKeyPairs = queryKeyPairIndices.size(1);

    torch::Tensor attentionLogits = torch::zeros({numQueryKeyPairs, numHeads}, queries.options());

    if (numQueryKeyPairs > 0) {
        attentionLogitsCRPE_forward_launcher(
            queries,
            keys,
            queryKeyPairIndices,
            queryRelativeXYZTables,
            keyRelativeXYZTables,
            relativeXYZTableIndices,
            attentionLogits
        );
    }

    return attentionLogits;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> attentionLogitsCRPE_backward(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor attentionLogitsGradient
) {
    const int numHeads = queries.size(1);
    const int numQueryKeyPairs = queryKeyPairIndices.size(1);

    torch::Tensor queriesGradient = torch::zeros_like(queries);
    torch::Tensor keysGradient = torch::zeros_like(keys);
    torch::Tensor queryRelativeXYZTablesGradient = torch::zeros_like(queryRelativeXYZTables);
    torch::Tensor keyRelativeXYZTablesGradient = torch::zeros_like(keyRelativeXYZTables);

    if (numQueryKeyPairs > 0) {
        attentionLogitsCRPE_backward_launcher(
            queries,
            keys,
            queryKeyPairIndices,
            queryRelativeXYZTables,
            keyRelativeXYZTables,
            relativeXYZTableIndices,
            attentionLogitsGradient,
            queriesGradient,
            keysGradient,
            queryRelativeXYZTablesGradient,
            keyRelativeXYZTablesGradient
        );
    }

    return std::make_tuple(
        queriesGradient,
        keysGradient,
        queryRelativeXYZTablesGradient,
        keyRelativeXYZTablesGradient
    );
}
