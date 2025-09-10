#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h"

// --- KERNEL IMPLEMENTATIONS (ALL MODIFIED FOR BATCHING) ---

// Generic kernel for applying an arbitrary 1-qubit unitary matrix
__global__ void apply_single_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, size_t batchSize)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = (size_t)gridDim.x * blockDim.x;
    
    size_t num_elements_per_state = 1ULL << numQubits;
    size_t total_pairs = batchSize * (num_elements_per_state / 2);

    for (size_t i = tid; i < total_pairs; i += total_threads) {
        size_t batch_idx = i / (num_elements_per_state / 2);
        size_t pair_idx_in_batch = i % (num_elements_per_state / 2);

        size_t k = 1ULL << targetQubit;
        size_t idx0_local = (pair_idx_in_batch & ~(k - 1)) * 2 + (pair_idx_in_batch & (k - 1));
        size_t idx1_local = idx0_local + k;

        size_t batch_offset = batch_idx * num_elements_per_state;
        rocComplex amp0 = state[batch_offset + idx0_local];
        rocComplex amp1 = state[batch_offset + idx1_local];

        // Apply U from constant memory
        // ... (matrix multiplication logic) ...
    }
}

// Hadamard Gate
__global__ void apply_H_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, size_t batchSize) {
    const real_t inv_sqrt2 = 1.0 / sqrt(2.0);
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = (size_t)gridDim.x * blockDim.x;

    size_t num_elements_per_state = 1ULL << numQubits;
    size_t total_pairs = batchSize * (num_elements_per_state / 2);

    for (size_t i = tid; i < total_pairs; i += total_threads) {
        size_t batch_idx = i / (num_elements_per_state / 2);
        size_t pair_idx_in_batch = i % (num_elements_per_state / 2);

        size_t k = 1ULL << targetQubit;
        size_t idx0_local = (pair_idx_in_batch & ~(k - 1)) * 2 + (pair_idx_in_batch & (k - 1));
        size_t idx1_local = idx0_local + k;

        size_t batch_offset = batch_idx * num_elements_per_state;
        rocComplex amp0_old = state[batch_offset + idx0_local];
        rocComplex amp1_old = state[batch_offset + idx1_local];

        state[batch_offset + idx0_local].x = (amp0_old.x + amp1_old.x) * inv_sqrt2;
        state[batch_offset + idx0_local].y = (amp0_old.y + amp1_old.y) * inv_sqrt2;
        state[batch_offset + idx1_local].x = (amp0_old.x - amp1_old.x) * inv_sqrt2;
        state[batch_offset + idx1_local].y = (amp0_old.y - amp1_old.y) * inv_sqrt2;
    }
}

// ... (The same batching pattern is applied to X, Y, Z, S, T, Sdg, Tdg, RX, RY, RZ kernels) ...