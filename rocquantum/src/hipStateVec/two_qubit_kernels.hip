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

} // namespace

// CNOT gate: flip target qubit when control qubit is |1>
__global__ void apply_CNOT_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned controlQubit,
                                  unsigned targetQubit,
                                  size_t batchSize) {
    const size_t thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;

    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t total_states = batchSize * num_elements_per_state;

    const size_t control_mask = 1ULL << controlQubit;
    const size_t target_mask = 1ULL << targetQubit;

    for (size_t linear_idx = thread_id; linear_idx < total_states; linear_idx += total_threads) {
        const size_t batch_idx = linear_idx / num_elements_per_state;
        const size_t basis_state = linear_idx % num_elements_per_state;

        if ((basis_state & control_mask) == 0ULL) {
            continue;
        }
        if ((basis_state & target_mask) != 0ULL) {
            continue; // Only process the |0> branch for the target to avoid double swaps.
        }

        const size_t partner_state = basis_state | target_mask;
        const size_t batch_offset = batch_idx * num_elements_per_state;
        const size_t idx0 = batch_offset + basis_state;
        const size_t idx1 = batch_offset + partner_state;

        const rocComplex temp = state[idx0];
        state[idx0] = state[idx1];
        state[idx1] = temp;
    }
}

// Controlled-Z: add a phase of -1 when both qubits are |1>
__global__ void apply_CZ_kernel(rocComplex* state,
                                unsigned numQubits,
                                unsigned controlQubit,
                                unsigned targetQubit,
                                size_t batchSize) {
    const size_t thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;

    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t total_states = batchSize * num_elements_per_state;

    const size_t control_mask = 1ULL << controlQubit;
    const size_t target_mask = 1ULL << targetQubit;

    for (size_t linear_idx = thread_id; linear_idx < total_states; linear_idx += total_threads) {
        const size_t batch_idx = linear_idx / num_elements_per_state;
        const size_t basis_state = linear_idx % num_elements_per_state;

        if ((basis_state & control_mask) == 0ULL || (basis_state & target_mask) == 0ULL) {
            continue;
        }

        const size_t idx = batch_idx * num_elements_per_state + basis_state;
        state[idx].x = -state[idx].x;
        state[idx].y = -state[idx].y;
    }
}

// SWAP gate between two qubits (no controls)
__global__ void apply_SWAP_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned qubitA,
                                  unsigned qubitB,
                                  size_t batchSize) {
    const size_t thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;

    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t groups_per_state = num_elements_per_state >> 2; // 2^(n-2) groups of 4 amplitudes
    const size_t total_groups = batchSize * groups_per_state;

    if (groups_per_state == 0) {
        return;
    }

    const unsigned t1 = (qubitA < qubitB) ? qubitA : qubitB;
    const unsigned t2 = (qubitA < qubitB) ? qubitB : qubitA;
    const size_t mask_t1 = 1ULL << t1;
    const size_t mask_t2 = 1ULL << t2;

    for (size_t group_idx = thread_id; group_idx < total_groups; group_idx += total_threads) {
        const size_t batch_idx = group_idx / groups_per_state;
        const size_t group_in_batch = group_idx % groups_per_state;

        size_t base_state = 0;
        size_t source_bits = group_in_batch;
        for (unsigned qubit = 0; qubit < numQubits; ++qubit) {
            if (qubit == t1 || qubit == t2) {
                continue;
            }
            if (source_bits & 1ULL) {
                base_state |= (1ULL << qubit);
            }
            source_bits >>= 1ULL;
        }

        const size_t idx01 = base_state | mask_t2;        // |01>
        const size_t idx10 = base_state | mask_t1;        // |10>
        const size_t batch_offset = batch_idx * num_elements_per_state;

        const size_t index01 = batch_offset + idx01;
        const size_t index10 = batch_offset + idx10;

        const rocComplex temp = state[index01];
        state[index01] = state[index10];
        state[index10] = temp;
    }
}
