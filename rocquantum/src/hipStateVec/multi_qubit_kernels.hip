#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h" // For rocComplex

// Helper function to apply a generic matrix (matrix_dim x matrix_dim)
// to a local array of amplitudes.
// matrix is column-major.
__device__ inline void apply_small_matrix_local(rocComplex* local_amps,
                                                const rocComplex* matrixDevice,
                                                unsigned matrix_dim) {
    rocComplex temp_amps[16]; // Max for 4-qubit (16x16). Should be <= matrix_dim.
                              // For safety, ensure matrix_dim <= 16 for this buffer.

    for (unsigned i = 0; i < matrix_dim; ++i) {
        temp_amps[i] = local_amps[i]; // Copy to temp array
#ifdef ROCQ_PRECISION_DOUBLE
        local_amps[i] = {0.0, 0.0}; // Zero out for accumulation
#else
        local_amps[i] = {0.0f, 0.0f}; // Zero out for accumulation
#endif
    }

    for (unsigned i = 0; i < matrix_dim; ++i) { // Output amplitude index
        for (unsigned j = 0; j < matrix_dim; ++j) { // Input amplitude index
            // Matrix is column-major: M_ij is matrixDevice[i + j * matrix_dim]
            rocComplex M_ij = matrixDevice[i + j * matrix_dim];
            rocComplex val_j = temp_amps[j];
            // local_amps[i] += M_ij * val_j;
            local_amps[i].x += M_ij.x * val_j.x - M_ij.y * val_j.y;
            local_amps[i].y += M_ij.x * val_j.y + M_ij.y * val_j.x;
        }
    }
}

// Kernel for applying an m-qubit generic matrix.
// targetQubitIndices should be sorted for consistent index generation.
// This kernel is designed to be somewhat general for small m (3 or 4).
__global__ void apply_multi_qubit_generic_matrix_kernel(
    rocComplex* state,
    unsigned numQubits,         // Total number of qubits
    const unsigned* targetQubitIndices, // Array of m target qubit indices (GPU accessible)
    unsigned m,                 // Number of target qubits (3 or 4)
    const rocComplex* matrixDevice) { // 2^m x 2^m matrix on device (column-major)

    size_t N = 1ULL << numQubits;
    unsigned matrix_dim = 1U << m; // 8 for m=3, 16 for m=4

    // Each thread will handle one block of 2^m states.
    // The number of such blocks is N / (2^m).
    size_t num_blocks_of_2_m_states = N >> m; // N / matrix_dim
    size_t thread_idx_overall = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx_overall >= num_blocks_of_2_m_states) {
        return;
    }

    // --- Complex Indexing Logic ---
    // The goal is to map thread_idx_overall to the starting configuration of non-target qubits,
    // and then iterate through the 2^m states of the target qubits.

    // 1. Create bitmasks for target qubits and non-target qubits
    unsigned long long target_qubits_mask = 0;
    // unsigned long long non_target_qubits_mask = 0; // Not explicitly used later, but good for understanding

    // Pointers to targetQubitIndices must be accessible on device if kernel needs to read it per thread.
    // For m=3,4 this is small enough to pass by value or reconstruct if needed, but array is more general.
    // Assuming targetQubitIndices is small and can be loaded into registers or shared memory by the compiler.
    
    unsigned temp_target_indices[4]; // Max m=4
    for(unsigned i=0; i<m; ++i) {
        temp_target_indices[i] = targetQubitIndices[i]; // Copy to avoid repeated global memory access if targetQubitIndices is global
        target_qubits_mask |= (1ULL << temp_target_indices[i]);
    }
    // non_target_qubits_mask = (~target_qubits_mask) & ((1ULL << numQubits) - 1); // Definition

    // 2. Determine the state of non-target qubits based on thread_idx_overall
    // thread_idx_overall effectively iterates through all combinations of non_target_qubits.
    size_t base_idx_for_non_targets = 0;
    unsigned current_non_target_bit_pos = 0; // which bit of thread_idx_overall we are using
    for (unsigned i = 0; i < numQubits; ++i) { // iterate over all qubit positions in the full state vector
        if (!((target_qubits_mask >> i) & 1)) { // if i-th qubit is a non-target qubit
            if (((thread_idx_overall >> current_non_target_bit_pos) & 1)) { // if this non-target qubit should be 1
                base_idx_for_non_targets |= (1ULL << i);
            }
            current_non_target_bit_pos++;
        }
    }

    // 3. Load the 2^m amplitudes for the current block into local memory
    rocComplex local_amps[16]; // Max for m=4. Use matrix_dim.
    
    for (unsigned i = 0; i < matrix_dim; ++i) { // i iterates 0 to 2^m - 1, representing target qubit configurations
        size_t current_target_config_idx = 0;
        // Construct the target qubit part of the index based on 'i'
        for (unsigned k=0; k<m; ++k) { // iterate through the m target qubits (using temp_target_indices)
            if(((i >> k) & 1)) { // if k-th bit of 'i' is 1
                 current_target_config_idx |= (1ULL << temp_target_indices[k]);
            }
        }
        local_amps[i] = state[base_idx_for_non_targets | current_target_config_idx];
    }

    // 4. Apply the matrix
    apply_small_matrix_local(local_amps, matrixDevice, matrix_dim);

    // 5. Write back the transformed amplitudes
    for (unsigned i = 0; i < matrix_dim; ++i) {
        size_t current_target_config_idx = 0;
        for (unsigned k=0; k<m; ++k) {
            if(((i >> k) & 1)) {
                 current_target_config_idx |= (1ULL << temp_target_indices[k]);
            }
        }
        state[base_idx_for_non_targets | current_target_config_idx] = local_amps[i];
    }
}

// Explicit versions for m=3 and m=4 that call the generic one.
// The compiler might inline this.
// Alternatively, rocsvApplyMatrix can directly call apply_multi_qubit_generic_matrix_kernel with correct m.

__global__ void apply_three_qubit_generic_matrix_kernel(
    rocComplex* state,
    unsigned numQubits,
    const unsigned* targetQubitIndices_gpu, // GPU accessible array of 3 indices
    const rocComplex* matrixDevice) {       // 8x8 matrix on device

    apply_multi_qubit_generic_matrix_kernel(state, numQubits, targetQubitIndices_gpu, 3, matrixDevice);
}

__global__ void apply_four_qubit_generic_matrix_kernel(
    rocComplex* state,
    unsigned numQubits,
    const unsigned* targetQubitIndices_gpu, // GPU accessible array of 4 indices
    const rocComplex* matrixDevice) {       // 16x16 matrix on device
    
    apply_multi_qubit_generic_matrix_kernel(state, numQubits, targetQubitIndices_gpu, 4, matrixDevice);
}

// Kernel to gather 2^m elements from their scattered positions in d_in_strided (full state vector)
// into a contiguous d_out_contiguous buffer.
// non_target_config_thread_idx: Identifies which configuration of non-target qubits this gather operation is for.
//                               It ranges from 0 to (2^(N-m) - 1).
// targetQubitIndices_gpu: Array of m target qubit indices, already on GPU.
/*
__global__ void gather_elements_kernel(
    rocComplex* d_out_contiguous,         // Output: Contiguous buffer of size 2^m
    const rocComplex* d_in_strided,       // Input: Full state vector (size 2^N)
    unsigned numQubits,                   // N
    const unsigned* targetQubitIndices_gpu, // m target qubit indices (on GPU)
    unsigned m) {                         // Number of target qubits

    // This kernel is launched with (2^m) threads, one for each element to gather.
    // It performs a single gather operation for *one* specific configuration of non-target qubits.
    // The information about which non-target configuration is processed must be implicitly
    // managed by how this kernel is called or by an additional parameter if this kernel
    // itself were to loop (which it currently does not).
    // For the current plan (host-side loop over non_target_config_idx), this kernel
    // would be called 2^(N-m) times. The d_out_contiguous would point to the start
    // of the *same* temp buffer each time. The d_in_strided is always the full state vector.
    // The key is that the *effective* addresses read from d_in_strided must change
    // for each of those 2^(N-m) calls. This is achieved by the calling code preparing
    // the correct base_idx_for_non_targets and adding it to target_config_component.

    // Let's refine the gather kernel to take base_idx_for_non_targets directly.
    // This means the host loop calculates base_idx_for_non_targets and passes it.
    // The kernel is launched with 2^m threads.

    // This refined kernel is called by the host 2^(N-m) times.
    // Each launch has 2^m threads.
    // targetQubitIndices_gpu must be sorted for this indexing to map i correctly.
    // (Caller of rocsvApplyMatrix should ensure qubitIndices are sorted before passing to kernel or sort internally)
    
    // This kernel is launched with `matrix_dim` (i.e., 2^m) threads.
    // Each thread `i` (from 0 to 2^m - 1) calculates one source index and copies one element.
    // `base_idx_for_non_targets_host_provided` is the component of the index from non-target qubits,
    // pre-calculated by the host for the current block.
    
    // For gather_elements_kernel_v2:
    // Grid: 1 block, Block: matrix_dim (2^m) threads.
    // Or, if matrix_dim is large, (matrix_dim / threads_per_block_max_256) blocks.
}
*/

// Refined Gather Kernel:
// Gathers 2^m elements for a *single* specified configuration of non-target qubits.
// Launched with matrix_dim threads (e.g., 32 threads for m=5).
// base_idx_non_targets: The component of the index determined by the fixed state of non-target qubits for this specific gather op.
__global__ void gather_elements_kernel_v2(
    rocComplex* d_out_contiguous,           // Output: Contiguous buffer of size 2^m (matrix_dim)
    const rocComplex* d_in_strided,         // Input: Full state vector (size 2^N)
    const unsigned* targetQubitIndices_gpu, // m target qubit indices (on GPU, assumed sorted by convention for this kernel)
    unsigned m,                             // Number of target qubits
    size_t base_idx_non_targets) {          // Index component from non-target qubits

    unsigned matrix_dim = 1U << m;
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x; // i from 0 to matrix_dim - 1

    if (i < matrix_dim) {
        size_t target_config_component = 0;
        // Construct the target qubit component of the index based on bits of 'i'
        for (unsigned k = 0; k < m; ++k) {
            if (((i >> k) & 1)) { // If k-th bit of 'i' is 1
                target_config_component |= (1ULL << targetQubitIndices_gpu[k]);
            }
        }
        d_out_contiguous[i] = d_in_strided[base_idx_non_targets | target_config_component];
    }
}

// Refined Scatter Kernel:
// Scatters 2^m elements for a *single* specified configuration of non-target qubits.
// Launched with matrix_dim threads.
// base_idx_non_targets: The component of the index determined by the fixed state of non-target qubits for this specific scatter op.
__global__ void scatter_elements_kernel_v2(
    rocComplex* d_out_strided,              // Output: Full state vector (size 2^N)
    const rocComplex* d_in_contiguous,      // Input: Contiguous buffer of size 2^m (matrix_dim)
    const unsigned* targetQubitIndices_gpu, // m target qubit indices (on GPU, assumed sorted)
    unsigned m,                             // Number of target qubits
    size_t base_idx_non_targets) {          // Index component from non-target qubits

    unsigned matrix_dim = 1U << m;
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x; // i from 0 to matrix_dim - 1

    if (i < matrix_dim) {
        size_t target_config_component = 0;
        // Construct the target qubit component of the index based on bits of 'i'
        for (unsigned k = 0; k < m; ++k) {
            if (((i >> k) & 1)) { // If k-th bit of 'i' is 1
                target_config_component |= (1ULL << targetQubitIndices_gpu[k]);
            }
        }
        d_out_strided[base_idx_non_targets | target_config_component] = d_in_contiguous[i];
    }
}
