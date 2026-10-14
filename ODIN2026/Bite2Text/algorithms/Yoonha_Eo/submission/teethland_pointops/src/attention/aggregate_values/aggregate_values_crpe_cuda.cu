#include <ATen/cuda/Atomic.cuh>

#include "../../cuda_utils.h"
#include "aggregate_values_crpe_cuda.h"


void aggregateValuesCRPE_forward_launcher(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor aggregatedValues
) {
    const int numValues = values.size(0);
    const int numHeads = values.size(1);
    const int numHeadChannels = values.size(2);
    const dim3 blocks(
        numValues,
        numHeads,
        numHeadChannels
    );
    const int threads = optThreads(maxKeyCount);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "aggregateValuesCRPE_forward", ([&] {
        aggregateValuesCRPE_forward_cuda<scalar_t><<<blocks, threads, threads * sizeof(double)>>>(
            values.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            queryKeyPairIndices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            queryKeyOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            attentionDistributions.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            valueRelativeXYZTables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            relativeXYZTableIndices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            aggregatedValues.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }));

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename scalar_t>
__global__ void aggregateValuesCRPE_forward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> values,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> queryKeyOffsets,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionDistributions,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> valueRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aggregatedValues
) {
    extern __shared__ double s[];
    double *aggregatedValue_hci = s;


    const int stride = blockDim.x;
    const int ti = threadIdx.x;

    const int qi = blockIdx.x;  // query index
    const int hi = blockIdx.y;  // head index
    const int hci = blockIdx.z;  // head channel index

    const int startIdx = qi == 0 ? 0 : queryKeyOffsets[qi - 1];
    const int endIdx = queryKeyOffsets[qi];


    int qki;  // query-key pair index
    int ki;  // key index
    int relXTableIdx, relYTableIdx, relZTableIdx;
    scalar_t rpe_hci;  // element of relative position encoding at index hci
    aggregatedValue_hci[ti] = 0;
    for (qki = startIdx + ti; qki < endIdx; qki += stride) {
        relXTableIdx = relativeXYZTableIndices[qki][0];
        relYTableIdx = relativeXYZTableIndices[qki][1];
        relZTableIdx = relativeXYZTableIndices[qki][2];

        rpe_hci = valueRelativeXYZTables[0][relXTableIdx][hi][hci];
        rpe_hci += valueRelativeXYZTables[1][relYTableIdx][hi][hci];
        rpe_hci += valueRelativeXYZTables[2][relZTableIdx][hi][hci];

        ki = queryKeyPairIndices[1][qki];
        aggregatedValue_hci[ti] += attentionDistributions[qki][hi] * (values[ki][hi][hci] + rpe_hci);
    }


    for (int pow_2 = stride >> 1; pow_2 > 0; pow_2 = pow_2 >> 1)
    {
        __syncthreads();
        
        if (ti < pow_2) {
            aggregatedValue_hci[ti] += aggregatedValue_hci[ti + pow_2];
        }
    }

    if (ti == 0) {
        aggregatedValues[qi][hi][hci] = aggregatedValue_hci[0];
    }
}


void aggregateValuesCRPE_backward_launcher(
    torch::Tensor values,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryKeyOffsets,
    const int maxKeyCount,
    torch::Tensor attentionDistributions,
    torch::Tensor valueRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor aggregatedValuesGradient,
    torch::Tensor valuesGradient,
    torch::Tensor attentionDistributionsGradient,
    torch::Tensor valueRelativeXYZTablesGradient
) {
    const int numValues = values.size(0);
    const int numHeads = values.size(1);
    const int numHeadChannels = values.size(2);
    const dim3 blocks(
        numValues,
        numHeads,
        numHeadChannels
    );
    const int threads = optThreads(maxKeyCount);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "aggregateValuesCRPE_backward", ([&] {
        aggregateValuesCRPE_backward_cuda<scalar_t><<<blocks, threads>>>(
            values.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            queryKeyPairIndices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            queryKeyOffsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            attentionDistributions.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            valueRelativeXYZTables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            relativeXYZTableIndices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            aggregatedValuesGradient.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            valuesGradient.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            attentionDistributionsGradient.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            valueRelativeXYZTablesGradient.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif 
}


template <typename scalar_t>
__global__ void aggregateValuesCRPE_backward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> values,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> queryKeyOffsets,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionDistributions,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> valueRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aggregatedValuesGradient,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> valuesGradient,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionDistributionsGradient,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> valueRelativeXYZTablesGradient
) {
    const int stride = blockDim.x;
    const int ti = threadIdx.x;

    const int qi = blockIdx.x;  // query index
    const int hi = blockIdx.y;  // head index
    const int hci = blockIdx.z;  // head channel index

    const int startIdx = qi == 0 ? 0 : queryKeyOffsets[qi - 1];
    const int endIdx = queryKeyOffsets[qi];    

    const scalar_t dLdy = aggregatedValuesGradient[qi][hi][hci];  // partial derivative of loss wrt output

    int qki;  // query-key pair index
    int ki;  // key index
    int relXTableIdx, relYTableIdx, relZTableIdx;
    scalar_t dydx;  // partial derivative of output wrt input
    scalar_t dLdx;  // partial derivative of loss wrt input
    for (qki = startIdx + ti; qki < endIdx; qki += stride) {
        relXTableIdx = relativeXYZTableIndices[qki][0];
        relYTableIdx = relativeXYZTableIndices[qki][1];
        relZTableIdx = relativeXYZTableIndices[qki][2];

        ki = queryKeyPairIndices[1][qki];
        gpuAtomicAdd(&valuesGradient[ki][hi][hci], dLdy * attentionDistributions[qki][hi]);

        dydx = values[ki][hi][hci];
        dydx += valueRelativeXYZTables[0][relXTableIdx][hi][hci];
        dydx += valueRelativeXYZTables[1][relYTableIdx][hi][hci];
        dydx += valueRelativeXYZTables[2][relZTableIdx][hi][hci];
        gpuAtomicAdd(&attentionDistributionsGradient[qki][hi], dLdy * dydx);

        dLdx = dLdy * attentionDistributions[qki][hi];
        gpuAtomicAdd(&valueRelativeXYZTablesGradient[0][relXTableIdx][hi][hci], dLdx);
        gpuAtomicAdd(&valueRelativeXYZTablesGradient[1][relYTableIdx][hi][hci], dLdx);
        gpuAtomicAdd(&valueRelativeXYZTablesGradient[2][relZTableIdx][hi][hci], dLdx);
    }
}
