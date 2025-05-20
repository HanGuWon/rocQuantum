#include "rocquantum/hipStateVec.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <cstring> // For memset
#include <string>  // Not strictly used but good include
#include <cmath>   // For sqrt, fabs, etc.
#include <hiprand/hiprand.h> // For hiprand calls in measurement (though simplified)
#include <cstdlib> // For rand, srand (used in simplified measurement)
#include <ctime>   // For time (used in simplified measurement)
#include <algorithm> // For std::sort
#include <iostream>  // For potential debug (can be removed)

// Define the internal handle structure
struct rocsvInternalHandle {
    hipStream_t stream;
    int deviceId;
    rocblas_handle blasHandle;
    // hiprandGenerator_t rand_generator; // For future rocRAND integration
};

// Helper to check HIP errors and convert to rocqStatus_t
rocqStatus_t checkHipError(hipError_t err, const char* operation = "") {
    if (err != hipSuccess) {
        // fprintf(stderr, "HIP Error during %s: %s\n", operation, hipGetErrorString(err));
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

// Helper to check rocBLAS errors and convert to rocqStatus_t
rocqStatus_t checkRocblasError(rocblas_status err, const char* operation = "") {
    if (err != rocblas_status_success) {
        // fprintf(stderr, "rocBLAS Error during %s: %s\n", operation, rocblas_status_to_string(err));
        return ROCQ_STATUS_FAILURE; // Or a new ROCQ_STATUS_ROCBLAS_ERROR
    }
    return ROCQ_STATUS_SUCCESS;
}

// Kernel Forward Declarations
__global__ void initializeToZeroStateKernel(rocComplex* state, size_t num_elements);
__global__ void apply_single_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, const rocComplex* matrixDevice);
__global__ void apply_X_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_Y_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_Z_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_H_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_S_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_T_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_Rx_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, float theta);
__global__ void apply_Ry_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, float theta);
__global__ void apply_Rz_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, float theta);

__global__ void apply_two_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, unsigned qubit0_idx, unsigned qubit1_idx, const rocComplex* matrixDevice);
__global__ void apply_CNOT_kernel(rocComplex* state, unsigned numQubits, unsigned controlQubit_idx, unsigned targetQubit_idx);
__global__ void apply_CZ_kernel(rocComplex* state, unsigned numQubits, unsigned qubit0_idx, unsigned qubit1_idx);
__global__ void apply_SWAP_kernel(rocComplex* state, unsigned numQubits, unsigned qubit0_idx, unsigned qubit1_idx);

__global__ void calculate_prob0_kernel(const rocComplex* state, unsigned numQubits, unsigned targetQubit, double* d_prob0_sum);
__global__ void collapse_state_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, int measuredOutcome);
__global__ void sum_sq_magnitudes_kernel(const rocComplex* state, unsigned numQubits, double* d_sum_sq_mag);
__global__ void renormalize_state_kernel(rocComplex* state, unsigned numQubits, double d_sum_sq_mag_inv_sqrt);

__global__ void apply_three_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, const unsigned* targetQubitIndices_gpu, const rocComplex* matrixDevice);
__global__ void apply_four_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, const unsigned* targetQubitIndices_gpu, const rocComplex* matrixDevice);

__global__ void gather_elements_kernel_v2(rocComplex* d_out_contiguous, const rocComplex* d_in_strided, const unsigned* targetQubitIndices_gpu, unsigned m, size_t base_idx_non_targets);
__global__ void scatter_elements_kernel_v2(rocComplex* d_out_strided, const rocComplex* d_in_contiguous, const unsigned* targetQubitIndices_gpu, unsigned m, size_t base_idx_non_targets);


extern "C" {

rocqStatus_t rocsvCreate(rocsvHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = new(std::nothrow) rocsvInternalHandle;
    if (!internal_handle) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    internal_handle->stream = nullptr;
    internal_handle->blasHandle = nullptr;

    hipError_t hip_err;
    rocblas_status blas_err;

    hip_err = hipGetDevice(&internal_handle->deviceId);
    if (hip_err != hipSuccess) {
        delete internal_handle;
        return checkHipError(hip_err, "rocsvCreate hipGetDevice");
    }
    hip_err = hipStreamCreate(&internal_handle->stream);
    if (hip_err != hipSuccess) {
        delete internal_handle;
        return checkHipError(hip_err, "rocsvCreate hipStreamCreate");
    }
    blas_err = rocblas_create_handle(&internal_handle->blasHandle);
    if (blas_err != rocblas_status_success) {
        if (internal_handle->stream) hipStreamDestroy(internal_handle->stream);
        delete internal_handle;
        return checkRocblasError(blas_err, "rocsvCreate rocblas_create_handle");
    }
    blas_err = rocblas_set_stream(internal_handle->blasHandle, internal_handle->stream);
    if (blas_err != rocblas_status_success) {
        if (internal_handle->blasHandle) rocblas_destroy_handle(internal_handle->blasHandle);
        if (internal_handle->stream) hipStreamDestroy(internal_handle->stream);
        delete internal_handle;
        return checkRocblasError(blas_err, "rocsvCreate rocblas_set_stream");
    }
    *handle = internal_handle;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvDestroy(rocsvHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    rocqStatus_t first_error_status = ROCQ_STATUS_SUCCESS;
    if (internal_handle->blasHandle) {
        rocblas_status blas_err = rocblas_destroy_handle(internal_handle->blasHandle);
        if (blas_err != rocblas_status_success) {
            first_error_status = checkRocblasError(blas_err, "rocsvDestroy rocblas_destroy_handle");
        }
    }
    if (internal_handle->stream) {
        hipError_t hip_err = hipStreamDestroy(internal_handle->stream);
        if (hip_err != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
            first_error_status = checkHipError(hip_err, "rocsvDestroy hipStreamDestroy");
        }
    }
    delete internal_handle;
    return first_error_status;
}

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle, unsigned numQubits, rocComplex** d_state) {
    if (!handle || !d_state || numQubits == 0 || numQubits > 60) { // Max qubits practical limit for full state vector
        return ROCQ_STATUS_INVALID_VALUE;
    }
    size_t num_elements = 1ULL << numQubits;
    size_t size_bytes = num_elements * sizeof(rocComplex);
    hipError_t err = hipMalloc(d_state, size_bytes);
    if (err != hipSuccess) {
        *d_state = nullptr;
        return checkHipError(err, "rocsvAllocateState hipMalloc");
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvInitializeState(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits) {
    if (!handle || !d_state || numQubits > 60) { // Allow numQubits = 0 for 1-element state vector
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    size_t num_elements = 1ULL << numQubits;
    hipError_t err = hipMemsetAsync(d_state, 0, num_elements * sizeof(rocComplex), internal_handle->stream);
    if (err != hipSuccess) {
        return checkHipError(err, "rocsvInitializeState hipMemsetAsync");
    }
    if (num_elements > 0) { // Only set first element if vector is not empty
        rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
        err = hipMemcpyAsync(d_state, &zero_state_amplitude, sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->stream);
        if (err != hipSuccess) {
            return checkHipError(err, "rocsvInitializeState hipMemcpyAsync for first element");
        }
    }
    err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(err, "rocsvInitializeState hipStreamSynchronize");
}


rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle,
                              rocComplex* d_state,
                              unsigned numQubits,
                              const unsigned* qubitIndices, // HOST pointer
                              unsigned numTargetQubits,    // m
                              const rocComplex* matrixDevice, // Gate matrix on DEVICE memory
                              unsigned matrixDim) {        // Should be 2^m
    if (!handle || !d_state || !qubitIndices || !matrixDevice) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numTargetQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    // Validate matrixDim against numTargetQubits
    // (1U << numTargetQubits) can overflow if numTargetQubits is large (e.g. >=31 for unsigned)
    // However, we limit numTargetQubits below, so this is safe.
    if (matrixDim != (1U << numTargetQubits)) {
         return ROCQ_STATUS_INVALID_VALUE;
    }

    for (unsigned i = 0; i < numTargetQubits; ++i) {
        if (qubitIndices[i] >= numQubits) return ROCQ_STATUS_INVALID_VALUE;
        for (unsigned j = i + 1; j < numTargetQubits; ++j) {
            if (qubitIndices[i] == qubitIndices[j]) return ROCQ_STATUS_INVALID_VALUE; // Duplicate indices
        }
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    unsigned threads_per_block = 256;
    size_t total_states = 1ULL << numQubits;

    if (numTargetQubits == 1) {
        unsigned targetQubit = qubitIndices[0];
        size_t num_thread_groups = total_states / 2;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) num_blocks = 0;
        
        if (num_blocks > 0 || (numQubits == 0 && targetQubit == 0)) { // numQubits=0 case: 1 state, 0 pairs, 0 blocks.
                                                                     // if numQubits=0, targetQubit must be 0.
                                                                     // apply_single_qubit_generic_matrix_kernel should handle N=1.
            if (numQubits == 0 && targetQubit == 0) num_blocks = 0; // No kernel launch for 0 qubits

            if (num_blocks > 0) {
                 hipLaunchKernelGGL(apply_single_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, targetQubit, matrixDevice);
                 hip_err = hipGetLastError();
                 if (hip_err != hipSuccess) status = checkHipError(hip_err, "rocsvApplyMatrix (1Q)");
            }
        }
    } else if (numTargetQubits == 2) {
        unsigned q0 = (qubitIndices[0] < qubitIndices[1]) ? qubitIndices[0] : qubitIndices[1];
        unsigned q1 = (qubitIndices[0] < qubitIndices[1]) ? qubitIndices[1] : qubitIndices[0];
        size_t num_thread_groups = total_states / 4;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) num_blocks = 0;
        if (num_blocks > 0) {
            hipLaunchKernelGGL(apply_two_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, q0, q1, matrixDevice);
            hip_err = hipGetLastError();
            if (hip_err != hipSuccess) status = checkHipError(hip_err, "rocsvApplyMatrix (2Q)");
        }
    } else if (numTargetQubits == 3 || numTargetQubits == 4) {
        unsigned* d_targetIndices = nullptr;
        hip_err = hipMalloc(&d_targetIndices, numTargetQubits * sizeof(unsigned));
        if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplyMatrix d_targetIndices malloc (3Q/4Q)");
        
        // Using synchronous copy for d_targetIndices as it's small and needed for kernel launch config
        hip_err = hipMemcpy(d_targetIndices, qubitIndices, numTargetQubits * sizeof(unsigned), hipMemcpyHostToDevice);
        if (hip_err != hipSuccess) {
            hipFree(d_targetIndices);
            return checkHipError(hip_err, "rocsvApplyMatrix d_targetIndices memcpy (3Q/4Q)");
        }
        size_t m_val = numTargetQubits;
        size_t num_kernel_threads = (numQubits < m_val) ? 0 : (total_states >> m_val);
        unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
        if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1;
        if (num_kernel_threads == 0) num_blocks = 0;

        if (num_blocks > 0) {
            if (numTargetQubits == 3) {
                hipLaunchKernelGGL(apply_three_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, d_targetIndices, matrixDevice);
            } else { // numTargetQubits == 4
                hipLaunchKernelGGL(apply_four_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, d_targetIndices, matrixDevice);
            }
            hip_err = hipGetLastError();
            if (hip_err != hipSuccess) status = checkHipError(hip_err, "rocsvApplyMatrix (3Q/4Q)");
        }
        // hipFree is blocking. Ensure kernel is done or use stream-ordered free if available and appropriate.
        // For simplicity here, hipStreamSynchronize before free is safer if kernels were async and used d_targetIndices for long.
        // However, d_targetIndices is used by value in kernel or copied to registers/shared mem quickly.
        if (status == ROCQ_STATUS_SUCCESS && num_blocks > 0) { // Sync only if kernel launched
             hipError_t sync_err = hipStreamSynchronize(internal_handle->stream);
             if(sync_err != hipSuccess) status = checkHipError(sync_err, "rocsvApplyMatrix sync for d_targetIndices (3Q/4Q)");
        }
        hipError_t free_err = hipFree(d_targetIndices);
        if (free_err != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = checkHipError(free_err, "rocsvApplyMatrix free d_targetIndices (3Q/4Q)");

    } else if (numTargetQubits >= 5) {
        const unsigned MAX_M_FOR_GATHER_SCATTER = 10; // Practical limit for this approach
        if (numTargetQubits > MAX_M_FOR_GATHER_SCATTER) return ROCQ_STATUS_NOT_IMPLEMENTED;

        unsigned m = numTargetQubits;
        // matrixDim is already 2^m

        std::vector<unsigned> h_sorted_target_indices(qubitIndices, qubitIndices + m);
        std::sort(h_sorted_target_indices.begin(), h_sorted_target_indices.end());

        unsigned* d_targetIndices = nullptr;
        rocComplex* d_temp_vec_in = nullptr;
        rocComplex* d_temp_vec_out = nullptr;

        hip_err = hipMalloc(&d_targetIndices, m * sizeof(unsigned));
        if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_targetIndices malloc (m>=5)"); goto cleanup_m5; }
        
        // Using synchronous copy for d_targetIndices as it's small and needed for kernel launch config for gather/scatter
        hip_err = hipMemcpy(d_targetIndices, h_sorted_target_indices.data(), m * sizeof(unsigned), hipMemcpyHostToDevice);
        if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_targetIndices memcpy (m>=5)"); goto cleanup_m5; }

        hip_err = hipMalloc(&d_temp_vec_in, matrixDim * sizeof(rocComplex));
        if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_temp_vec_in malloc (m>=5)"); goto cleanup_m5; }
        hip_err = hipMalloc(&d_temp_vec_out, matrixDim * sizeof(rocComplex));
        if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_temp_vec_out malloc (m>=5)"); goto cleanup_m5; }

        rocblas_float_complex alpha_gemv = {1.0f, 0.0f};
        rocblas_float_complex beta_gemv = {0.0f, 0.0f};
        unsigned num_non_target_qubits = numQubits - m;
        size_t num_non_target_configs = 1ULL << num_non_target_qubits;
        unsigned long long target_qubits_mask_val = 0;
        for(unsigned i=0; i<m; ++i) target_qubits_mask_val |= (1ULL << h_sorted_target_indices[i]);

        unsigned gs_threads_per_block = 256;
        if (matrixDim < gs_threads_per_block && matrixDim > 0) gs_threads_per_block = matrixDim;
        else if (matrixDim == 0) gs_threads_per_block = 1; 
        unsigned gs_num_blocks = (matrixDim + gs_threads_per_block - 1) / gs_threads_per_block;
        if (matrixDim == 0) gs_num_blocks = 0;
        else if (gs_num_blocks == 0 && matrixDim > 0) gs_num_blocks = 1;
        
        if (gs_num_blocks == 0 && matrixDim > 0) { // Should not happen with matrixDim > 0
            status = ROCQ_STATUS_INVALID_VALUE; 
            goto cleanup_m5; 
        }

        for (size_t j = 0; j < num_non_target_configs; ++j) {
            if (status != ROCQ_STATUS_SUCCESS) break;
            size_t base_idx_non_targets = 0;
            unsigned current_non_target_bit_pos = 0;
            for (unsigned bit_idx = 0; bit_idx < numQubits; ++bit_idx) {
                if (!((target_qubits_mask_val >> bit_idx) & 1)) {
                    if (((j >> current_non_target_bit_pos) & 1)) {
                        base_idx_non_targets |= (1ULL << bit_idx);
                    }
                    current_non_target_bit_pos++;
                }
            }
            if (gs_num_blocks > 0) {
                 hipLaunchKernelGGL(gather_elements_kernel_v2, dim3(gs_num_blocks), dim3(gs_threads_per_block), 0, internal_handle->stream,
                                   d_temp_vec_in, d_state, d_targetIndices, m, base_idx_non_targets);
                hip_err = hipGetLastError();
                if (hip_err != hipSuccess) { status = checkHipError(hip_err, "gather_elements_kernel_v2"); break; }
            }
            // Ensure gather is complete before rocBLAS uses d_temp_vec_in
            // hipStreamSynchronize(internal_handle->stream); // Or use rocBLAS with same stream

            rocblas_status blas_status = rocblas_cgemv(internal_handle->blasHandle, rocblas_operation_none,
                               matrixDim, matrixDim, 
                               &alpha_gemv, // Pass by pointer
                               (const rocblas_float_complex*)matrixDevice, matrixDim, // Cast matrix
                               (const rocblas_float_complex*)d_temp_vec_in, 1,        // Cast vector
                               &beta_gemv,  // Pass by pointer
                               (rocblas_float_complex*)d_temp_vec_out, 1);       // Cast vector
            if (blas_status != rocblas_status_success) {
                status = checkRocblasError(blas_status, "rocblas_cgemv (m>=5)");
                break;
            }
             // Ensure rocBLAS is complete before scatter uses d_temp_vec_out
            // hipStreamSynchronize(internal_handle->stream); // Or use rocBLAS with same stream

            if (gs_num_blocks > 0) {
                hipLaunchKernelGGL(scatter_elements_kernel_v2, dim3(gs_num_blocks), dim3(gs_threads_per_block), 0, internal_handle->stream,
                                   d_state, d_temp_vec_out, d_targetIndices, m, base_idx_non_targets);
                hip_err = hipGetLastError();
                if (hip_err != hipSuccess) { status = checkHipError(hip_err, "scatter_elements_kernel_v2"); break; }
            }
        }
cleanup_m5:
        if (d_temp_vec_out) hipFree(d_temp_vec_out);
        if (d_temp_vec_in) hipFree(d_temp_vec_in);
        if (d_targetIndices) hipFree(d_targetIndices);
    } else {
        // This case implies numTargetQubits is 0 or some other unhandled case by prior checks.
        // Or if numTargetQubits > MAX_M_FOR_GATHER_SCATTER and it wasn't returned early.
        // Should be caught by initial validation (numTargetQubits == 0 or matrixDim mismatch)
        status = ROCQ_STATUS_INVALID_VALUE; 
    }

    if (status == ROCQ_STATUS_SUCCESS) {
        hip_err = hipStreamSynchronize(internal_handle->stream);
        if (hip_err != hipSuccess) {
            status = checkHipError(hip_err, "rocsvApplyMatrix hipStreamSynchronize at end");
        }
    }
    return status;
}

rocqStatus_t rocsvMeasure(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned qubitToMeasure,
                          int* h_outcome,
                          double* h_probability) {
    if (!handle || !d_state || !h_outcome || !h_probability || qubitToMeasure >= numQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numQubits == 0 && qubitToMeasure == 0) { // Special case: 1 state, measure Q0
        // Probability of |0> is |state[0]|^2. Outcome is 0.
        // This needs d_state to be copied to host to determine.
        // For simplicity, assume this is an edge case not fully supported by placeholder kernels.
        // Or, if state is |0>, outcome 0, prob 1.
        // For now, let it pass to kernels which might handle N=1 state.
    } else if (numQubits == 0 && qubitToMeasure !=0) {
        return ROCQ_STATUS_INVALID_VALUE; // Cannot measure non-existent qubit
    }


    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    double* d_prob0_sum = nullptr;
    double h_prob0_sum = 0.0;

    hip_err = hipMalloc(&d_prob0_sum, sizeof(double));
    if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvMeasure hipMalloc d_prob0_sum");

    // Placeholder kernel needs 1 block, 1 thread
    hipLaunchKernelGGL(calculate_prob0_kernel, dim3(1), dim3(1), 0, internal_handle->stream,
                       d_state, numQubits, qubitToMeasure, d_prob0_sum);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(calculate_prob0_kernel)");
        hipFree(d_prob0_sum); return status;
    }
    hip_err = hipMemcpyAsync(&h_prob0_sum, d_prob0_sum, sizeof(double), hipMemcpyDeviceToHost, internal_handle->stream);
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipMemcpyAsync d_prob0_sum");
        hipFree(d_prob0_sum); return status;
    }
    hip_err = hipStreamSynchronize(internal_handle->stream);
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize after prob calc");
        hipFree(d_prob0_sum); return status;
    }
    hipFree(d_prob0_sum);

    double prob0 = h_prob0_sum;
    double prob1 = 1.0 - prob0;
    if (prob0 < 0.0) prob0 = 0.0; if (prob0 > 1.0) prob0 = 1.0;
    if (prob1 < 0.0) prob1 = 0.0; if (prob1 > 1.0) prob1 = 1.0;

    static bool seeded = false; if (!seeded) { srand((unsigned int)time(NULL)); seeded = true; }
    double rand_val = (double)rand() / RAND_MAX;

    if (rand_val < prob0) {
        *h_outcome = 0;
        *h_probability = prob0;
    } else {
        *h_outcome = 1;
        *h_probability = prob1;
    }

    unsigned threads_per_block_measure = 256;
    size_t total_states_measure = (1ULL << numQubits);
    unsigned num_blocks_measure = (total_states_measure + threads_per_block_measure - 1) / threads_per_block_measure;
    if (total_states_measure == 0) num_blocks_measure = 0; // Should not happen if numQubits>=0
    else if (num_blocks_measure == 0 && total_states_measure > 0) num_blocks_measure = 1;


    if (num_blocks_measure > 0) {
        hipLaunchKernelGGL(collapse_state_kernel, dim3(num_blocks_measure), dim3(threads_per_block_measure), 0, internal_handle->stream,
                           d_state, numQubits, qubitToMeasure, *h_outcome);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(collapse_state_kernel)");
    }

    double* d_sum_sq_mag = nullptr;
    double h_sum_sq_mag = 0.0;
    hip_err = hipMalloc(&d_sum_sq_mag, sizeof(double));
    if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvMeasure hipMalloc d_sum_sq_mag");

    // Placeholder kernel needs 1 block, 1 thread
    hipLaunchKernelGGL(sum_sq_magnitudes_kernel, dim3(1), dim3(1), 0, internal_handle->stream,
                       d_state, numQubits, d_sum_sq_mag);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(sum_sq_magnitudes_kernel)");
        hipFree(d_sum_sq_mag); return status;
    }
    hip_err = hipMemcpyAsync(&h_sum_sq_mag, d_sum_sq_mag, sizeof(double), hipMemcpyDeviceToHost, internal_handle->stream);
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipMemcpyAsync d_sum_sq_mag");
        hipFree(d_sum_sq_mag); return status;
    }
    hip_err = hipStreamSynchronize(internal_handle->stream);
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize after sum_sq_mag calc");
        hipFree(d_sum_sq_mag); return status;
    }
    hipFree(d_sum_sq_mag);

    if (h_sum_sq_mag > 1e-12) { // Avoid division by zero or normalizing an almost zero state
        double norm_factor = 1.0 / sqrt(h_sum_sq_mag);
        if (num_blocks_measure > 0) {
            hipLaunchKernelGGL(renormalize_state_kernel, dim3(num_blocks_measure), dim3(threads_per_block_measure), 0, internal_handle->stream,
                               d_state, numQubits, norm_factor);
            hip_err = hipGetLastError();
            if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(renormalize_state_kernel)");
        }
    } else if (*h_probability > 1e-9) { // If outcome was probable but state norm is near zero
        // This is an inconsistent state, potentially due to errors in placeholder reduction kernels
        // or an unnormalized input state prior to measurement.
        // For now, we don't change status, but in a robust library, this might be an error.
    }


    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize at end");
}

// Static helper functions for specific gate API calls
static rocqStatus_t launch_single_qubit_kernel(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, const char* opName, void (*kernel_func)(rocComplex*, unsigned, unsigned)) {
    if (!handle || !d_state || targetQubit >= numQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    // Handle numQubits = 0 case (1 state)
    if (numQubits == 0 && targetQubit == 0) { // Op on Q0 of a 1-state vector
         // Most single qubit gates are identity on a 0-qubit system's single state |->
         // or it implies an error depending on gate type. For generic kernel, it's up to kernel.
         // apply_single_qubit_generic_matrix_kernel expects N>=2 (num_thread_groups > 0).
         // For simplicity, treat as no-op or rely on kernel to handle N=1 if it's designed for it.
         // Current kernels are designed for N/2 pairs, so N must be at least 2.
         // If numQubits = 0, N=1, N/2 = 0. num_blocks will be 0.
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    size_t total_states = 1ULL << numQubits;
    unsigned threads_per_block = 256;
    size_t num_thread_groups = total_states / 2;
    unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
    if (num_thread_groups == 0) num_blocks = 0;
    else if (num_blocks == 0 && num_thread_groups > 0) num_blocks = 1;

    if (num_blocks > 0) {
        hipLaunchKernelGGL(kernel_func, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, targetQubit);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) return checkHipError(hip_err, opName);
    } else if (numQubits == 0 && targetQubit == 0) { // No kernel launch for 0-qubit system
        return ROCQ_STATUS_SUCCESS; // No operation performed, successful by definition
    } else if (numQubits > 0 && num_blocks == 0 && num_thread_groups == 0) { // e.g. numQubits=1, N=2, N/2=1 thread group.
         // This case might mean 1 thread group, if threads_per_block is e.g. 256, num_blocks is 1.
         // The logic above `if (num_blocks == 0 && num_thread_groups > 0) num_blocks = 1;` handles this.
         // This else-if is likely redundant or for very specific unhandled edge cases.
    }


    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, opName);
}

static rocqStatus_t launch_single_qubit_rotation_kernel(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta, const char* opName, void (*kernel_func)(rocComplex*, unsigned, unsigned, float)) {
    if (!handle || !d_state || targetQubit >= numQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
     if (numQubits == 0 && targetQubit == 0) {
        // As above, typically a no-op for 0-qubit system.
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    size_t total_states = 1ULL << numQubits;
    unsigned threads_per_block = 256;
    size_t num_thread_groups = total_states / 2;
    unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
    if (num_thread_groups == 0) num_blocks = 0;
    else if (num_blocks == 0 && num_thread_groups > 0) num_blocks = 1;

    if (num_blocks > 0) {
        hipLaunchKernelGGL(kernel_func, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, targetQubit, static_cast<float>(theta));
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) return checkHipError(hip_err, opName);
    } else if (numQubits == 0 && targetQubit == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, opName);
}

// API Functions for Specific Gates
rocqStatus_t rocsvApplyX(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    return launch_single_qubit_kernel(handle, d_state, numQubits, targetQubit, "rocsvApplyX", apply_X_kernel);
}
rocqStatus_t rocsvApplyY(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    return launch_single_qubit_kernel(handle, d_state, numQubits, targetQubit, "rocsvApplyY", apply_Y_kernel);
}
rocqStatus_t rocsvApplyZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    return launch_single_qubit_kernel(handle, d_state, numQubits, targetQubit, "rocsvApplyZ", apply_Z_kernel);
}
rocqStatus_t rocsvApplyH(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    return launch_single_qubit_kernel(handle, d_state, numQubits, targetQubit, "rocsvApplyH", apply_H_kernel);
}
rocqStatus_t rocsvApplyS(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    return launch_single_qubit_kernel(handle, d_state, numQubits, targetQubit, "rocsvApplyS", apply_S_kernel);
}
rocqStatus_t rocsvApplyT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    return launch_single_qubit_kernel(handle, d_state, numQubits, targetQubit, "rocsvApplyT", apply_T_kernel);
}
rocqStatus_t rocsvApplyRx(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    return launch_single_qubit_rotation_kernel(handle, d_state, numQubits, targetQubit, theta, "rocsvApplyRx", apply_Rx_kernel);
}
rocqStatus_t rocsvApplyRy(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    return launch_single_qubit_rotation_kernel(handle, d_state, numQubits, targetQubit, theta, "rocsvApplyRy", apply_Ry_kernel);
}
rocqStatus_t rocsvApplyRz(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    return launch_single_qubit_rotation_kernel(handle, d_state, numQubits, targetQubit, theta, "rocsvApplyRz", apply_Rz_kernel);
}

rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit) {
    if (!handle || !d_state || controlQubit >= numQubits || targetQubit >= numQubits || controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numQubits < 2 && (controlQubit > 0 || targetQubit > 0) ) return ROCQ_STATUS_INVALID_VALUE; // Not enough qubits for non-trivial CNOT
    if (numQubits < 2 && controlQubit ==0 && targetQubit == 0) return ROCQ_STATUS_INVALID_VALUE; // CNOT on same qubit
    
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    unsigned threads_per_block = 256;
    size_t num_kernel_threads = (numQubits < 2) ? 0 : (1ULL << (numQubits - 2));
    unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
    if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1;
    if (num_kernel_threads == 0) num_blocks = 0;

    if (num_blocks > 0) {
        hipLaunchKernelGGL(apply_CNOT_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, controlQubit, targetQubit);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplyCNOT");
    } else if (numQubits < 2) { // No operation for CNOT on <2 qubits
        return ROCQ_STATUS_SUCCESS;
    }
    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, "rocsvApplyCNOT");
}

rocqStatus_t rocsvApplyCZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2) {
    if (!handle || !d_state || qubit1 >= numQubits || qubit2 >= numQubits || qubit1 == qubit2) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numQubits < 2 && (qubit1 > 0 || qubit2 > 0) ) return ROCQ_STATUS_INVALID_VALUE;
    if (numQubits < 2 && qubit1 ==0 && qubit2 == 0) return ROCQ_STATUS_INVALID_VALUE;


    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    unsigned threads_per_block = 256;
    size_t num_kernel_threads = (numQubits < 2) ? 0 : (1ULL << (numQubits - 2));
    unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
    if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1;
    if (num_kernel_threads == 0) num_blocks = 0;
    
    if (num_blocks > 0) {
        hipLaunchKernelGGL(apply_CZ_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, qubit1, qubit2);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplyCZ");
    } else if (numQubits < 2) {
        return ROCQ_STATUS_SUCCESS;
    }
    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, "rocsvApplyCZ");
}

rocqStatus_t rocsvApplySWAP(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2) {
    if (!handle || !d_state || qubit1 >= numQubits || qubit2 >= numQubits || qubit1 == qubit2) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numQubits < 2 && (qubit1 > 0 || qubit2 > 0) ) return ROCQ_STATUS_INVALID_VALUE;
    if (numQubits < 2 && qubit1 ==0 && qubit2 == 0) return ROCQ_STATUS_INVALID_VALUE;

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    unsigned threads_per_block = 256;
    size_t num_kernel_threads = (numQubits < 2) ? 0 : (1ULL << (numQubits - 2));
    unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
    if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1;
    if (num_kernel_threads == 0) num_blocks = 0;

    if (num_blocks > 0) {
        hipLaunchKernelGGL(apply_SWAP_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, qubit1, qubit2);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplySWAP");
    } else if (numQubits < 2) {
        return ROCQ_STATUS_SUCCESS;
    }
    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, "rocsvApplySWAP");
}

} // extern "C"
