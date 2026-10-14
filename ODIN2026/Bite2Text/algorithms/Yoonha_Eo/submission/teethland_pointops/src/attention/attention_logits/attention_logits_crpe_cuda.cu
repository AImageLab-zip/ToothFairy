#include <ATen/cuda/Atomic.cuh>

#include "../../cuda_utils.h"
#include "attention_logits_crpe_cuda.h"


void attentionLogitsCRPE_forward_launcher(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor attentionLogits
) {
    const dim3 blocks(
        DIVUP(attentionLogits.size(0), THREADS_PER_BLOCK),
        attentionLogits.size(1)
    );
    const int threads = THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.scalar_type(), "attentionLogitsCRPE_forward", ([&] {
        attentionLogitsCRPE_forward_cuda<scalar_t><<<blocks, threads>>>(
            queries.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            queryKeyPairIndices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            queryRelativeXYZTables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            keyRelativeXYZTables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            relativeXYZTableIndices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            attentionLogits.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename scalar_t>
__global__ void attentionLogitsCRPE_forward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> keys,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queryRelativeXYZTables,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keyRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionLogits
) {
    const int qki = blockDim.x * blockIdx.x + threadIdx.x;  // query-key pair index
    if (qki >= queryKeyPairIndices.size(1)) return;

    const int qi = queryKeyPairIndices[0][qki];  // query index
    const int ki = queryKeyPairIndices[1][qki];  // key index
    const int hi = blockIdx.y;  // head index
    const int numHeadChannels = queries.size(2);

    const int relXTableIdx = relativeXYZTableIndices[qki][0];
    const int relYTableIdx = relativeXYZTableIndices[qki][1];
    const int relZTableIdx = relativeXYZTableIndices[qki][2];

    int hci;  // head channel index
    scalar_t rpe_hci;  // element of relative position encoding at index hci
    for (hci = 0; hci < numHeadChannels; hci++) {
        // dot product between query and key
        attentionLogits[qki][hi] += queries[qi][hi][hci] * keys[ki][hi][hci];
        
        // relative position bias with query context
        rpe_hci = queryRelativeXYZTables[0][relXTableIdx][hi][hci];
        rpe_hci += queryRelativeXYZTables[1][relYTableIdx][hi][hci];
        rpe_hci += queryRelativeXYZTables[2][relZTableIdx][hi][hci];
        attentionLogits[qki][hi] += queries[qi][hi][hci] * rpe_hci;
        
        // relative position bias with key context
        rpe_hci = keyRelativeXYZTables[0][relXTableIdx][hi][hci];
        rpe_hci += keyRelativeXYZTables[1][relYTableIdx][hi][hci];
        rpe_hci += keyRelativeXYZTables[2][relZTableIdx][hi][hci];
        attentionLogits[qki][hi] += keys[ki][hi][hci] * rpe_hci;
    }
}


void attentionLogitsCRPE_backward_launcher(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor queryKeyPairIndices,
    torch::Tensor queryRelativeXYZTables,
    torch::Tensor keyRelativeXYZTables,
    torch::Tensor relativeXYZTableIndices,
    torch::Tensor attentionLogitsGradient,
    torch::Tensor queriesGradient,
    torch::Tensor keysGradient,
    torch::Tensor queryRelativeXYZTablesGradient,
    torch::Tensor keyRelativeXYZTablesGradient
) {
    const dim3 blocks(
        DIVUP(attentionLogitsGradient.size(0), THREADS_PER_BLOCK),
        attentionLogitsGradient.size(1)
    );
    const int threads = THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.scalar_type(), "attentionLogitsCRPE_backward", ([&] {
        attentionLogitsCRPE_backward_cuda<scalar_t><<<blocks, threads>>>(
            queries.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            queryKeyPairIndices.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            queryRelativeXYZTables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            keyRelativeXYZTables.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            relativeXYZTableIndices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            attentionLogitsGradient.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            queriesGradient.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            keysGradient.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            queryRelativeXYZTablesGradient.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            keyRelativeXYZTablesGradient.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()        
        );
    }));

#if DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename scalar_t>
__global__ void attentionLogitsCRPE_backward_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> keys,
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> queryKeyPairIndices,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queryRelativeXYZTables,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keyRelativeXYZTables,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> relativeXYZTableIndices,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> attentionLogitsGradient,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queriesGradient,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> keysGradient,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queryRelativeXYZTablesGradient,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keyRelativeXYZTablesGradient
) {
    const int qki = blockDim.x * blockIdx.x + threadIdx.x;  // query-key pair index
    if (qki >= queryKeyPairIndices.size(1)) return;

    const int qi = queryKeyPairIndices[0][qki];  // query index
    const int ki = queryKeyPairIndices[1][qki];  // key index
    const int hi = blockIdx.y;  // head index
    const int numHeadChannels = queries.size(2);

    const int relXTableIdx = relativeXYZTableIndices[qki][0];
    const int relYTableIdx = relativeXYZTableIndices[qki][1];
    const int relZTableIdx = relativeXYZTableIndices[qki][2];

    const scalar_t dLdy = attentionLogitsGradient[qki][hi];  // partial derivative of loss wrt output
 
    int hci;  // head channel index
    scalar_t dydx;  // partial derivative of output wrt input
    scalar_t dLdx;  // partial derivative of loss wrt input
    for (hci = 0; hci < numHeadChannels; hci++) {
        dydx = keys[ki][hi][hci];
        dydx += queryRelativeXYZTables[0][relXTableIdx][hi][hci];
        dydx += queryRelativeXYZTables[1][relYTableIdx][hi][hci];
        dydx += queryRelativeXYZTables[2][relZTableIdx][hi][hci];
        gpuAtomicAdd(&queriesGradient[qi][hi][hci], dLdy * dydx);

        dydx = queries[qi][hi][hci];
        dydx += keyRelativeXYZTables[0][relXTableIdx][hi][hci];
        dydx += keyRelativeXYZTables[1][relYTableIdx][hi][hci];
        dydx += keyRelativeXYZTables[2][relZTableIdx][hi][hci];
        gpuAtomicAdd(&keysGradient[ki][hi][hci], dLdy * dydx);

        dLdx = dLdy * queries[qi][hi][hci];
        gpuAtomicAdd(&queryRelativeXYZTablesGradient[0][relXTableIdx][hi][hci], dLdx);
        gpuAtomicAdd(&queryRelativeXYZTablesGradient[1][relYTableIdx][hi][hci], dLdx);
        gpuAtomicAdd(&queryRelativeXYZTablesGradient[2][relZTableIdx][hi][hci], dLdx);
        
        dLdx = dLdy * keys[ki][hi][hci];
        gpuAtomicAdd(&keyRelativeXYZTablesGradient[0][relXTableIdx][hi][hci], dLdx);
        gpuAtomicAdd(&keyRelativeXYZTablesGradient[1][relYTableIdx][hi][hci], dLdx);
        gpuAtomicAdd(&keyRelativeXYZTablesGradient[2][relZTableIdx][hi][hci], dLdx);
    }
}
