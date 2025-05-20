#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h" // For rocComplex
#include <hiprand/hiprand.h> // For basic random number generation

// Kernel to calculate the sum of squared magnitudes of amplitudes where targetQubit is 0
// This is prob0. prob1 = 1.0 - prob0.
// This kernel performs a reduction. It's a simplified version.
// For high performance, a multi-stage reduction (e.g., using shared memory) is better.
__global__ void calculate_prob0_kernel(const rocComplex* state,
                                       unsigned numQubits,
                                       unsigned targetQubit,
                                       double* d_prob0_sum) { // Device pointer for sum
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit; // Stride related to targetQubit

    // Each thread processes one element and adds to a shared sum if it's a '0' state for targetQubit
    // This is a naive sum, not a proper parallel reduction.
    // A proper reduction would be needed for performance.
    // For now, this illustrates the logic path. THIS WILL BE SLOW AND INCORRECT FOR PARALLEL EXECUTION.
    // A real implementation would use shared memory for block-level reduction, then atomicAdd for global sum.
    // Or, better, use rocThrust/hipThrust for reduction.

    // THIS IS A PLACEHOLDER KERNEL LOGIC FOR PROBABILITY CALCULATION
    // DO NOT USE AS IS FOR A PRODUCTION SYSTEM DUE TO RACE CONDITIONS AND INCORRECT PARALLEL SUMMATION
    if (threadIdx.x == 0 && blockIdx.x == 0) { // HACK: Only one thread does the sum for now to avoid race
        double p0 = 0.0;
        for (size_t i = 0; i < N; ++i) {
            // Check if the targetQubit bit is 0 for state i
            // (i / k) % 2 == 0  <-- this checks if the bit at targetQubit position is 0
            if (((i >> targetQubit) & 1) == 0) {
                p0 += (double)state[i].x * state[i].x + (double)state[i].y * state[i].y;
            }
        }
        *d_prob0_sum = p0;
    }
}


// Kernel to collapse the state vector based on measurement outcome
// If outcome is 0, zero out states where targetQubit is 1.
// If outcome is 1, zero out states where targetQubit is 0.
__global__ void collapse_state_kernel(rocComplex* state,
                                      unsigned numQubits,
                                      unsigned targetQubit,
                                      int measuredOutcome) {
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        bool targetQubitIsOne = ((tid >> targetQubit) & 1);
        if (measuredOutcome == 0 && targetQubitIsOne) { // Measured 0, so zero out states where qubit is 1
            state[tid] = make_hipFloatComplex(0.0f, 0.0f);
        } else if (measuredOutcome == 1 && !targetQubitIsOne) { // Measured 1, so zero out states where qubit is 0
            state[tid] = make_hipFloatComplex(0.0f, 0.0f);
        }
    }
}

// Kernel to re-normalize the state vector
// sum_sq_mag = sum(|amp_i|^2) over remaining non-zero amplitudes
// norm_factor = 1.0 / sqrt(sum_sq_mag)
// amp_i_new = amp_i_old * norm_factor
// This also needs a proper parallel reduction for sum_sq_mag.
__global__ void renormalize_state_kernel(rocComplex* state,
                                         unsigned numQubits,
                                         double d_sum_sq_mag_inv_sqrt) { // Pass 1.0/sqrt(sum_sq_mag)
    size_t N = 1ULL << numQubits;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        state[tid].x *= d_sum_sq_mag_inv_sqrt;
        state[tid].y *= d_sum_sq_mag_inv_sqrt;
    }
}

// Placeholder for sum of square magnitudes kernel (also needs proper reduction)
__global__ void sum_sq_magnitudes_kernel(const rocComplex* state, unsigned numQubits, double* d_sum_sq_mag) {
    size_t N = 1ULL << numQubits;
    // HACK: Single thread calculation for now
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double current_sum_sq_mag = 0.0;
        for (size_t i = 0; i < N; ++i) {
            current_sum_sq_mag += (double)state[i].x * state[i].x + (double)state[i].y * state[i].y;
        }
        *d_sum_sq_mag = current_sum_sq_mag;
    }
}
