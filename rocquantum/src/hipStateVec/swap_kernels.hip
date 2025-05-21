#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h" // For rocComplex

// Note: These helpers are specific to kernel execution.
// The convention assumed for distribution: global_idx = (source_gpu_rank << num_local_qubits_per_gpu) | current_local_idx;
// This means the 'source_gpu_rank' forms the most significant bits of the part of the global index
// that determines the distribution across GPUs. 'num_local_qubits_per_gpu' defines the size
// of the state vector part local to each GPU.

__device__ static inline size_t rocquant_kernel_reconstruct_global_idx(
    int rank, 
    size_t local_idx, 
    unsigned num_local_qubits_on_gpu) {
    return (static_cast<size_t>(rank) << num_local_qubits_on_gpu) | local_idx;
}

__device__ static inline int rocquant_kernel_get_target_rank_from_global_idx(
    size_t global_idx, 
    unsigned num_local_qubits_on_gpu) {
    // This extracts the bits that were originally the rank.
    return static_cast<int>(global_idx >> num_local_qubits_on_gpu);
}

// __device__ static inline size_t rocquant_kernel_get_target_local_idx_from_global_idx(
//     size_t global_idx,
//     unsigned num_local_qubits_on_gpu
// ){
//     size_t target_local_idx_mask = (1ULL << num_local_qubits_on_gpu) - 1;
//     return global_idx & target_local_idx_mask;
// }

// Swaps specified bits in a given number.
__device__ static inline size_t rocquant_kernel_swap_bits(size_t value, unsigned bit_pos1, unsigned bit_pos2) {
    unsigned bit1_val = (value >> bit_pos1) & 1;
    unsigned bit2_val = (value >> bit_pos2) & 1;
    if (bit1_val != bit2_val) {
        value ^= (1ULL << bit_pos1); 
        value ^= (1ULL << bit_pos2); 
    }
    return value;
}

/**
 * @brief Kernel to calculate send counts for each target GPU.
 */
__global__ void calculate_swap_counts_kernel(
    size_t local_slice_num_elements,
    unsigned qubit_idx1_global,           
    unsigned qubit_idx2_global,           
    unsigned num_local_qubits_per_gpu,  
    int source_gpu_rank,
    int* d_send_counts_for_source_gpu 
) {
    size_t current_local_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    
    for (; current_local_idx < local_slice_num_elements; current_local_idx += gridDim.x * blockDim.x) {
        size_t current_global_idx = rocquant_kernel_reconstruct_global_idx(source_gpu_rank, current_local_idx, num_local_qubits_per_gpu);
        size_t new_global_idx = rocquant_kernel_swap_bits(current_global_idx, qubit_idx1_global, qubit_idx2_global);
        int target_gpu_rank = rocquant_kernel_get_target_rank_from_global_idx(new_global_idx, num_local_qubits_per_gpu);
        hipAtomicAdd(&d_send_counts_for_source_gpu[target_gpu_rank], 1);
    }
}


/**
 * @brief Kernel to shuffle data into a packed send buffer for Alltoallv.
 */
__global__ void shuffle_data_for_swap_kernel(
    const rocComplex* local_slice_data_in,
    size_t local_slice_num_elements,
    unsigned qubit_idx1_global,
    unsigned qubit_idx2_global,
    unsigned num_local_qubits_per_gpu,
    int source_gpu_rank,
    const int* d_send_displacements_for_source_gpu, 
    rocComplex* d_packed_send_buffer_out,           
    int* d_output_buffer_atomic_counters            
) {
    size_t current_local_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (; current_local_idx < local_slice_num_elements; current_local_idx += gridDim.x * blockDim.x) {
        size_t current_global_idx = rocquant_kernel_reconstruct_global_idx(source_gpu_rank, current_local_idx, num_local_qubits_per_gpu);
        size_t new_global_idx = rocquant_kernel_swap_bits(current_global_idx, qubit_idx1_global, qubit_idx2_global);
        int target_gpu_rank = rocquant_kernel_get_target_rank_from_global_idx(new_global_idx, num_local_qubits_per_gpu);
        int write_offset_in_segment = hipAtomicAdd(&d_output_buffer_atomic_counters[target_gpu_rank], 1);
        size_t final_buffer_idx = static_cast<size_t>(d_send_displacements_for_source_gpu[target_gpu_rank]) + write_offset_in_segment;
        d_packed_send_buffer_out[final_buffer_idx] = local_slice_data_in[current_local_idx];
    }
}

/**
 * @brief Kernel to permute elements within a local slice.
 * This is used when qubit_idx1 and qubit_idx2 are both within the local domain.
 */
__global__ void local_bit_swap_permutation_kernel(
    rocComplex* d_local_slice,          // Data to permute in-place
    rocComplex* d_temp_buffer_for_slice,// Temporary buffer of the same size as d_local_slice
    size_t local_slice_num_elements,
    unsigned local_qubit_idx1,          // Qubit index relative to the start of local bits
    unsigned local_qubit_idx2           // Qubit index relative to the start of local bits
) {
    size_t current_local_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Copy to temp buffer first
    if (current_local_idx < local_slice_num_elements) {
        d_temp_buffer_for_slice[current_local_idx] = d_local_slice[current_local_idx];
    }
    __syncthreads(); // Ensure all data is copied before proceeding

    if (current_local_idx < local_slice_num_elements) {
        size_t new_local_idx = rocquant_kernel_swap_bits(current_local_idx, local_qubit_idx1, local_qubit_idx2);
        d_local_slice[new_local_idx] = d_temp_buffer_for_slice[current_local_idx];
    }
}
