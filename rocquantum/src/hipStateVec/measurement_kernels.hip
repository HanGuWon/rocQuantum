#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h" // For rocComplex
#include <hiprand/hiprand.h> // For basic random number generation

// Kernel to calculate the sum of squared magnitudes of amplitudes where targetQubit is 0
// This is prob0. prob1 = 1.0 - prob0.
// This kernel performs a reduction. It's a simplified version.
// For high performance, a multi-stage reduction (e.g., using shared memory) is better.
// THIS IS A PLACEHOLDER KERNEL LOGIC FOR PROBABILITY CALCULATION (SINGLE GPU CONTEXT)
// DO NOT USE AS IS FOR A PRODUCTION SYSTEM DUE TO RACE CONDITIONS AND INCORRECT PARALLEL SUMMATION
// if not using a proper reduction framework like rocThrust.
__global__ void calculate_prob0_kernel(const rocComplex* state,
                                       unsigned numQubits, // Total qubits in this state vector (slice)
                                       unsigned targetQubit, // Local index of target qubit in this slice
                                       real_t* d_prob0_sum) { // Device pointer for sum (use real_t)
    size_t N = 1ULL << numQubits;
    // size_t k = 1ULL << targetQubit; // Stride related to targetQubit

    // HACK: Only one thread does the sum for now to avoid race for this placeholder
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        real_t p0 = 0.0;
        if (N > 0) { // Ensure state is not empty (e.g. 0 local qubits for this slice)
            for (size_t i = 0; i < N; ++i) {
                // Check if the targetQubit bit is 0 for state i
                if (((i >> targetQubit) & 1) == 0) {
                    p0 += (real_t)state[i].x * state[i].x + (real_t)state[i].y * state[i].y;
                }
            }
        }
        *d_prob0_sum = p0;
    }
}

// Kernel to collapse the state vector based on measurement outcome
// If outcome is 0, zero out states where targetQubit is 1.
// If outcome is 1, zero out states where targetQubit is 0.
__global__ void collapse_state_kernel(rocComplex* state,
                                      unsigned numQubits, // Total qubits in this state vector (slice)
                                      unsigned targetQubit, // Local index of target qubit
                                      int measuredOutcome) {
    size_t N = 1ULL << numQubits;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) { // Check bounds for the current slice
        bool targetQubitIsOne = ((tid >> targetQubit) & 1);
        if (measuredOutcome == 0 && targetQubitIsOne) { // Measured 0, so zero out states where qubit is 1
#ifdef ROCQ_PRECISION_DOUBLE
            state[tid] = {0.0, 0.0};
#else
            state[tid] = {0.0f, 0.0f};
#endif
        } else if (measuredOutcome == 1 && !targetQubitIsOne) { // Measured 1, so zero out states where qubit is 0
#ifdef ROCQ_PRECISION_DOUBLE
            state[tid] = {0.0, 0.0};
#else
            state[tid] = {0.0f, 0.0f};
#endif
        }
        // Else, the amplitude remains (it's part of the collapsed state)
    }
}

// Kernel to re-normalize the state vector
// sum_sq_mag = sum(|amp_i|^2) over remaining non-zero amplitudes
// norm_factor = 1.0 / sqrt(sum_sq_mag)
// amp_i_new = amp_i_old * norm_factor
__global__ void renormalize_state_kernel(rocComplex* state,
                                         unsigned numQubits, // Total qubits in this state vector (slice)
                                         real_t d_sum_sq_mag_inv_sqrt) { // Pass 1.0/sqrt(sum_sq_mag) (use real_t)
    size_t N = 1ULL << numQubits;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) { // Check bounds for the current slice
        state[tid].x *= d_sum_sq_mag_inv_sqrt;
        state[tid].y *= d_sum_sq_mag_inv_sqrt;
    }
}

// Placeholder for sum of square magnitudes kernel (also needs proper reduction)
// THIS IS A PLACEHOLDER KERNEL LOGIC (SINGLE GPU CONTEXT)
__global__ void sum_sq_magnitudes_kernel(const rocComplex* state,
                                         unsigned numQubits, // Total qubits in this state vector (slice)
                                         real_t* d_sum_sq_mag) { // Use real_t
    size_t N = 1ULL << numQubits;
    // HACK: Single thread calculation for now
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        real_t current_sum_sq_mag = 0.0;
        if (N > 0) {
            for (size_t i = 0; i < N; ++i) {
                current_sum_sq_mag += (real_t)state[i].x * state[i].x + (real_t)state[i].y * state[i].y;
            }
        }
        *d_sum_sq_mag = current_sum_sq_mag;
    }
}


// --- New Kernels for Multi-GPU Measurement ---

// Performs a block-level reduction to calculate sum of probabilities for |0> and |1> states of a target qubit.
// Output d_block_partial_probs must be pre-allocated to hold 2 * gridDim.x real_ts.
// Each block writes its two sums (prob0_block, prob1_block) to its designated slot.
__global__ void calculate_local_slice_probabilities_kernel(
    const rocComplex* local_slice_data,
    size_t local_slice_num_elements,    // Number of elements in this GPU's slice
    unsigned num_local_qubits,          // Number of qubits represented by local_slice_data
    unsigned local_target_qubit,        // Index of the qubit to measure (local to this slice)
    real_t* d_block_partial_probs) {    // Output: array for [block0_p0, block0_p1, block1_p0, block1_p1, ...] (use real_t)

    extern __shared__ real_t sdata[]; // Shared memory for reduction, size: blockDim.x * 2 real_ts

    unsigned int tid_in_block = threadIdx.x;
    size_t global_idx_start = blockIdx.x * blockDim.x;

    real_t my_prob0 = 0.0;
    real_t my_prob1 = 0.0;

    // Each thread processes multiple elements if necessary (grid-stride loop)
    for (size_t i = tid_in_block; i < blockDim.x; i += blockDim.x) { // This loop is over elements within a block's responsibility
        size_t current_local_idx = global_idx_start + i;
        if (current_local_idx < local_slice_num_elements) {
            rocComplex amp = local_slice_data[current_local_idx];
            real_t mag_sq = (real_t)amp.x * amp.x + (real_t)amp.y * amp.y;

            if (((current_local_idx >> local_target_qubit) & 1) == 0) { // Qubit is |0>
                my_prob0 += mag_sq;
            } else { // Qubit is |1>
                my_prob1 += mag_sq;
            }
        }
    }

    sdata[tid_in_block] = my_prob0;
    sdata[tid_in_block + blockDim.x] = my_prob1; // Store prob1 sums after prob0 sums
    __syncthreads();

    // Parallel reduction within the block for prob0
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads();
    }
    // Parallel reduction within the block for prob1
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata[tid_in_block + blockDim.x] += sdata[tid_in_block + blockDim.x + s];
        }
        __syncthreads();
    }

    // Block leader writes the reduced sum for this block
    if (tid_in_block == 0) {
        d_block_partial_probs[blockIdx.x * 2 + 0] = sdata[0];             // prob0 for this block
        d_block_partial_probs[blockIdx.x * 2 + 1] = sdata[blockDim.x];    // prob1 for this block
    }
}


// Performs a block-level reduction for sum of squared magnitudes.
// Output d_block_sum_sq_mag must be pre-allocated to hold gridDim.x real_ts.
// Each block writes its sum to its designated slot.
__global__ void calculate_local_slice_sum_sq_mag_kernel(
    const rocComplex* local_slice_data,
    size_t local_slice_num_elements,    // Number of elements in this GPU's slice
    real_t* d_block_sum_sq_mag) {       // Output: array for [block0_sum, block1_sum, ...] (use real_t)

    extern __shared__ real_t sdata_sum[]; // Shared memory for reduction, size: blockDim.x real_ts

    unsigned int tid_in_block = threadIdx.x;
    size_t global_idx_start = blockIdx.x * blockDim.x;

    real_t my_sum_sq = 0.0;

    for (size_t i = tid_in_block; i < blockDim.x; i += blockDim.x) {
        size_t current_local_idx = global_idx_start + i;
        if (current_local_idx < local_slice_num_elements) {
            rocComplex amp = local_slice_data[current_local_idx];
            my_sum_sq += (real_t)amp.x * amp.x + (real_t)amp.y * amp.y;
        }
    }

    sdata_sum[tid_in_block] = my_sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata_sum[tid_in_block] += sdata_sum[tid_in_block + s];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        d_block_sum_sq_mag[blockIdx.x] = sdata_sum[0];
    }
}

// Kernel to reduce an array of block-level probability sums (prob0, prob1 pairs) to a final slice total.
// d_block_partial_probs: input array of size num_blocks * 2.
// d_slice_total_probs_out: output array of size 2 (total_prob0, total_prob1 for the slice).
// Assumes this kernel is launched with enough threads/blocks to efficiently sum d_block_partial_probs.
// For simplicity, this can be a single block if num_blocks is not excessively large (e.g., < 1024*2).
__global__ void reduce_block_sums_to_slice_total_probs_kernel(
    const real_t* d_block_partial_probs, // Flattened array: [b0p0, b0p1, b1p0, b1p1, ...] (use real_t)
    unsigned num_blocks_from_previous_kernel,
    real_t* d_slice_total_probs_out) { // Output: [total_p0, total_p1] (use real_t)

    extern __shared__ real_t s_reduce_probs[]; // Shared memory, size: blockDim.x * 2 for p0 and p1 sums

    unsigned int tid = threadIdx.x;

    real_t my_p0_sum = 0.0;
    real_t my_p1_sum = 0.0;

    // Each thread sums a portion of the d_block_partial_probs array
    // This loop structure assumes this kernel is launched with a single block,
    // and blockDim.x is large enough, or it's a grid-stride loop.
    // For a simple single-block reduction:
    for (unsigned i = tid; i < num_blocks_from_previous_kernel; i += blockDim.x) {
        my_p0_sum += d_block_partial_probs[i * 2 + 0];
        my_p1_sum += d_block_partial_probs[i * 2 + 1];
    }

    s_reduce_probs[tid] = my_p0_sum;
    s_reduce_probs[tid + blockDim.x] = my_p1_sum; // Store p1 sums after p0 sums
    __syncthreads();

    // Reduce p0 sums in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce_probs[tid] += s_reduce_probs[tid + s];
        }
        __syncthreads();
    }
    // Reduce p1 sums in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { // tid is relative to the start of p1 sums in shared memory
            s_reduce_probs[tid + blockDim.x] += s_reduce_probs[tid + blockDim.x + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_slice_total_probs_out[0] = s_reduce_probs[0];
        d_slice_total_probs_out[1] = s_reduce_probs[blockDim.x];
    }
}


// Kernel to reduce an array of block-level sum-squared-magnitudes to a final slice total.
// d_block_sum_sq_mag: input array of size num_blocks.
// d_slice_total_sum_sq_mag_out: output array of size 1.
__global__ void reduce_block_sums_to_slice_total_sum_sq_mag_kernel(
    const real_t* d_block_sum_sq_mag_in, // Array of sums from each block of previous kernel (use real_t)
    unsigned num_blocks_from_previous_kernel,
    real_t* d_slice_total_sum_sq_mag_out) { // Output: single real_t for the slice's total sum_sq_mag

    extern __shared__ real_t s_reduce_sum_sq[]; // Shared memory, size: blockDim.x
    unsigned int tid = threadIdx.x;
    real_t my_sum = 0.0;

    for (unsigned i = tid; i < num_blocks_from_previous_kernel; i += blockDim.x) {
        my_sum += d_block_sum_sq_mag_in[i];
    }
    s_reduce_sum_sq[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce_sum_sq[tid] += s_reduce_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_slice_total_sum_sq_mag_out[0] = s_reduce_sum_sq[0];
    }
}

// Kernel to calculate probabilities for 2^k outcomes of k target Z measurements.
// Output: d_outcome_probs_blocks - an array where each block writes its 2^k partial probability sums.
// The size of d_outcome_probs_blocks should be gridDim.x * (1 << num_target_paulis).
__global__ void calculate_multi_z_probabilities_kernel(
    const rocComplex* local_slice_data,
    size_t local_slice_num_elements,
    unsigned num_local_qubits,         // Total qubits in this slice/state_vector
    const unsigned* d_target_qubits,   // Device array of target qubit indices (local to slice)
    unsigned num_target_paulis,        // Number of Z paulistrings in the product (k)
    real_t* d_outcome_probs_blocks) {  // Output for block-level sums

    unsigned int KERNEL_MAX_TARGET_PAULIS = 8; // Max k for this kernel (2^8=256 outcomes)
                                              // Adjust based on shared memory limits / complexity.
                                              // For >8, might need different strategy or multiple passes.
    if (num_target_paulis == 0 || num_target_paulis > KERNEL_MAX_TARGET_PAULIS) {
        // Or handle error, assert, etc. For now, just return if invalid.
        // If num_target_paulis is 0, result is just total sum of squares (norm), which is 1.0.
        // The calling C++ function should handle num_target_paulis = 0 separately.
        return;
    }

    unsigned num_outcomes = 1 << num_target_paulis; // 2^k

    // Shared memory for per-block reduction of 2^k probabilities
    // Max size: 256 * sizeof(real_t) for KERNEL_MAX_TARGET_PAULIS = 8
    extern __shared__ real_t s_prob_bins[]; // Size should be blockDim.x * num_outcomes if each thread had its own bins
                                         // Or just num_outcomes if we reduce carefully.
                                         // Let's use num_outcomes and atomicAdd or careful per-thread accumulation.
                                         // For a warp-level or block-level reduction, size is num_outcomes.

    unsigned int tid_in_block = threadIdx.x;

    // Initialize shared memory for this block
    if (tid_in_block < num_outcomes) {
        s_prob_bins[tid_in_block] = 0.0;
    }
    __syncthreads();

    size_t global_idx_start = blockIdx.x * blockDim.x;

    // Each thread iterates over its assigned elements from the slice
    for (size_t i = tid_in_block; i < blockDim.x; i += blockDim.x) { // This loop over elements within a block's responsibility
        size_t current_local_idx = global_idx_start + i;
        if (current_local_idx < local_slice_num_elements) {
            rocComplex amp = local_slice_data[current_local_idx];
            real_t mag_sq = (real_t)amp.x * amp.x + (real_t)amp.y * amp.y;

            // Determine the outcome bin for this amplitude
            unsigned outcome_bin_idx = 0;
            for (unsigned pauli_k = 0; pauli_k < num_target_paulis; ++pauli_k) {
                unsigned target_q_idx = d_target_qubits[pauli_k]; // Qubit index for the k-th Pauli Z
                if (((current_local_idx >> target_q_idx) & 1)) { // If this qubit is 1 for current amplitude
                    outcome_bin_idx |= (1 << pauli_k); // Set the k-th bit in the outcome bin index
                }
            }
            // Atomically add to the correct bin in shared memory
            // This is okay for up to ~32-64 bins if contention is managed.
            // For larger num_outcomes, a different reduction is needed.
            // If num_outcomes is small (e.g., <= 64), atomicAdd on shared mem is often fine.
            // Otherwise, each thread maintains its own num_outcomes bins and then reduce them.
            // Let's assume num_target_paulis is small enough that num_outcomes is manageable for atomicAdd.
            if (outcome_bin_idx < num_outcomes) { // Should always be true
                 atomicAdd(&s_prob_bins[outcome_bin_idx], mag_sq);
            }
        }
    }
    __syncthreads();

    // Block leader (thread 0 in block) writes the block's sums to global memory
    if (tid_in_block == 0) {
        for (unsigned bin_k = 0; bin_k < num_outcomes; ++bin_k) {
            d_outcome_probs_blocks[blockIdx.x * num_outcomes + bin_k] = s_prob_bins[bin_k];
        }
    }
}

// Kernel to reduce block-level sums of multi-Z outcome probabilities to slice totals.
// d_block_outcome_probs: input from previous kernel, size: num_prev_blocks * num_outcomes
// d_slice_total_outcome_probs: output, size: num_outcomes
__global__ void reduce_multi_z_block_probs_to_slice_total_kernel(
    const real_t* d_block_outcome_probs,
    unsigned num_prev_blocks,
    unsigned num_outcomes, // 2^k (number of Zs in product)
    real_t* d_slice_total_outcome_probs) {

    // Shared memory for reduction. Size: blockDim.x (if reducing one outcome bin per thread)
    // or num_outcomes (if each thread sums for one outcome bin across blocks).
    // Let's choose the latter: each thread in this kernel is responsible for one outcome bin.
    // So, this kernel should be launched with num_outcomes threads (if num_outcomes <= max_block_size).
    extern __shared__ real_t s_final_probs[]; // Size: num_outcomes (passed as shared mem size at launch)

    unsigned int outcome_idx_this_thread = threadIdx.x + blockIdx.x * blockDim.x; // This thread handles this outcome_idx

    if (outcome_idx_this_thread < num_outcomes) {
        real_t my_sum_for_this_outcome = 0.0;
        // Sum probabilities for outcome_idx_this_thread across all blocks from previous kernel
        for (unsigned block_i = 0; block_i < num_prev_blocks; ++block_i) {
            my_sum_for_this_outcome += d_block_outcome_probs[block_i * num_outcomes + outcome_idx_this_thread];
        }
        s_final_probs[outcome_idx_this_thread] = my_sum_for_this_outcome; // Storing in shared, though not strictly needed if 1 block
                                                                    // and writing directly to global.
                                                                    // If launched with 1 block, num_outcomes threads:
        d_slice_total_outcome_probs[outcome_idx_this_thread] = my_sum_for_this_outcome;
    }
    // If launched with multiple blocks, a further reduction of s_final_probs would be needed.
    // For now, assume this kernel is launched with 1 block and num_outcomes threads.
    // The C++ host code will need to ensure this launch configuration.
}
