#include "aggregate_values_crpe.h"
#include "aggregate_values_crpe_cuda.h"

#include <tuple>


torch::Tensor aggregateValuesCRPE_forward(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices
) {
    torch::Tensor aggregatedValues = torch::empty_like(values);

    if (queryKeyPairIndices.size(1) > 0) {
        aggregateValuesCRPE_forward_launcher(
            values,
            queryKeyPairIndices,
            queryKeyOffsets,
            maxKeyCount,
            attentionDistributions,
            valueRelativeXYZTables,
            relativeXYZTableIndices,
            aggregatedValues
        );
    }

    return aggregatedValues;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> aggregateValuesCRPE_backward(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor aggregatedValuesGradient
) {
    torch::Tensor valuesGradient = torch::zeros_like(values);
    torch::Tensor attentionDistributionsGradient = torch::zeros_like(attentionDistributions);
    torch::Tensor valueRelativeXYZTablesGradient = torch::zeros_like(valueRelativeXYZTables);
    
    if (queryKeyPairIndices.size(1) > 0) {
        aggregateValuesCRPE_backward_launcher(
            values,
            queryKeyPairIndices,
            queryKeyOffsets,
            maxKeyCount,
            attentionDistributions,
            valueRelativeXYZTables,
            relativeXYZTableIndices,
            aggregatedValuesGradient,
            valuesGradient,
            attentionDistributionsGradient,
            valueRelativeXYZTablesGradient
        );
    }

    return std::make_tuple(
        valuesGradient,
        attentionDistributionsGradient,
        valueRelativeXYZTablesGradient
    );
}
