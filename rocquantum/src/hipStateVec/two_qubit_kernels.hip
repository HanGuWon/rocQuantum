#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h"

// CNOT Gate
__global__ void apply_CNOT_kernel(rocComplex* state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit, size_t batchSize) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = (size_t)gridDim.x * blockDim.x;

    size_t num_elements_per_state = 1ULL << numQubits;
    size_t total_quads = batchSize * (num_elements_per_state / 4);

    size_t control_stride = 1ULL << controlQubit;
    size_t target_stride = 1ULL << targetQubit;

    for (size_t i = tid; i < total_quads; i += total_threads) {
        size_t batch_idx = i / (num_elements_per_state / 4);
        size_t quad_idx_in_batch = i % (num_elements_per_state / 4);

        size_t k = (quad_idx_in_batch & ~(target_stride - 1)) * 2 + (quad_idx_in_batch & (target_stride - 1));
        k = (k & ~(control_stride - 1)) * 2 + (k & (control_stride - 1));
        
        size_t idx10 = k | control_stride;
        size_t idx11 = k | control_stride | target_stride;

        size_t batch_offset = batch_idx * num_elements_per_state;
        rocComplex temp = state[batch_offset + idx10];
        state[batch_offset + idx10] = state[batch_offset + idx11];
        state[batch_offset + idx11] = temp;
    }
}

// ... (The same batching pattern is applied to CZ and SWAP kernels) ...