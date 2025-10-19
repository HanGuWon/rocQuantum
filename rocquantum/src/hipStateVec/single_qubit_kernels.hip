#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h"

namespace {

__device__ inline rocComplex rocq_make_complex(real_t real, real_t imag) {
#ifdef ROCQ_PRECISION_DOUBLE
    return rocComplex{real, imag};
#else
    return rocComplex{static_cast<float>(real), static_cast<float>(imag)};
#endif
}

__device__ inline rocComplex rocq_complex_add(const rocComplex& a, const rocComplex& b) {
    return rocq_make_complex(a.x + b.x, a.y + b.y);
}

__device__ inline rocComplex rocq_complex_mul(const rocComplex& a, const rocComplex& b) {
    return rocq_make_complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

} // namespace

/**
 * Applies a general 2x2 matrix to the specified target qubit across all batches.
 * The matrix elements are provided explicitly to avoid additional global memory loads.
 */
__global__ void apply_single_qubit_matrix_kernel(rocComplex* state,
                                                 unsigned numQubits,
                                                 unsigned targetQubit,
                                                 rocComplex m00,
                                                 rocComplex m01,
                                                 rocComplex m10,
                                                 rocComplex m11,
                                                 size_t batchSize) {
    const size_t thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;

    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t pairs_per_state = num_elements_per_state >> 1; // 2^(n-1)
    const size_t total_pairs = batchSize * pairs_per_state;

    if (pairs_per_state == 0) {
        return;
    }

    const size_t bit_stride = 1ULL << targetQubit;

    for (size_t pair_idx = thread_id; pair_idx < total_pairs; pair_idx += total_threads) {
        const size_t batch_idx = pair_idx / pairs_per_state;
        const size_t pair_idx_in_batch = pair_idx % pairs_per_state;

        const size_t idx0_local = (pair_idx_in_batch & ~(bit_stride - 1ULL)) * 2ULL
                                + (pair_idx_in_batch & (bit_stride - 1ULL));
        const size_t idx1_local = idx0_local + bit_stride;

        const size_t batch_offset = batch_idx * num_elements_per_state;
        const size_t idx0 = batch_offset + idx0_local;
        const size_t idx1 = batch_offset + idx1_local;

        const rocComplex amp0 = state[idx0];
        const rocComplex amp1 = state[idx1];

        const rocComplex out0 = rocq_complex_add(rocq_complex_mul(m00, amp0),
                                                 rocq_complex_mul(m01, amp1));
        const rocComplex out1 = rocq_complex_add(rocq_complex_mul(m10, amp0),
                                                 rocq_complex_mul(m11, amp1));

        state[idx0] = out0;
        state[idx1] = out1;
    }
}

/**
 * Applies a 2x2 matrix conditioned on a single control qubit being set to |1>.
 * The matrix acts on the target qubit within the subspace defined by the control.
 */
__global__ void apply_controlled_single_qubit_matrix_kernel(rocComplex* state,
                                                            unsigned numQubits,
                                                            unsigned controlQubit,
                                                            unsigned targetQubit,
                                                            rocComplex m00,
                                                            rocComplex m01,
                                                            rocComplex m10,
                                                            rocComplex m11,
                                                            size_t batchSize) {
    const size_t thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;

    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t pairs_per_state = num_elements_per_state >> 1; // 2^(n-1)
    const size_t total_pairs = batchSize * pairs_per_state;

    if (pairs_per_state == 0) {
        return;
    }

    const size_t target_stride = 1ULL << targetQubit;
    const size_t control_mask = 1ULL << controlQubit;

    for (size_t pair_idx = thread_id; pair_idx < total_pairs; pair_idx += total_threads) {
        const size_t batch_idx = pair_idx / pairs_per_state;
        const size_t pair_idx_in_batch = pair_idx % pairs_per_state;

        const size_t idx0_local = (pair_idx_in_batch & ~(target_stride - 1ULL)) * 2ULL
                                + (pair_idx_in_batch & (target_stride - 1ULL));
        const size_t idx1_local = idx0_local + target_stride;

        if ((idx0_local & control_mask) == 0ULL) {
            continue; // Control qubit not set to 1 in this branch.
        }

        const size_t batch_offset = batch_idx * num_elements_per_state;
        const size_t idx0 = batch_offset + idx0_local;
        const size_t idx1 = batch_offset + idx1_local;

        const rocComplex amp0 = state[idx0];
        const rocComplex amp1 = state[idx1];

        const rocComplex out0 = rocq_complex_add(rocq_complex_mul(m00, amp0),
                                                 rocq_complex_mul(m01, amp1));
        const rocComplex out1 = rocq_complex_add(rocq_complex_mul(m10, amp0),
                                                 rocq_complex_mul(m11, amp1));

        state[idx0] = out0;
        state[idx1] = out1;
    }
}
