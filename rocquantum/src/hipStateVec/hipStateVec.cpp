#include "rocquantum/hipStateVec.h" // Assuming this path will be set up in CMake include_directories
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h> // For rocBLAS
#include <vector> // For std::vector, used in InitializeState for host-side buffer
#include <cstring> // For memset, strcmp
#include <string> // For std::string in comparisons (though not strictly needed for C API matrix type)
#include <cmath> // For fabsf, sqrtf, acosf, asinf, sqrt
#include <hiprand/hiprand.h>    // For hiprand_uniform_double
#include <cstdlib> // For rand, srand
#include <ctime> // For time

// Define the internal handle structure
struct rocsvInternalHandle {
    hipStream_t stream;
    // hiprandGenerator_t rand_generator; // For future use with rocRAND state in handle
    int deviceId;
    rocblas_handle blasHandle; // Added for rocBLAS
    // rocrand_generator rand_generator; // For measurement later
};

// Helper to check HIP errors and convert to rocqStatus_t
rocqStatus_t checkHipError(hipError_t err, const char* operation = "") {
    if (err != hipSuccess) {
        // In a real library, you might log hipGetErrorString(err)
        // fprintf(stderr, "HIP Error during %s: %s
", operation, hipGetErrorString(err));
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

// Helper to check rocBLAS errors and convert to rocqStatus_t
rocqStatus_t checkRocblasError(rocblas_status err, const char* operation = "") {
    if (err != rocblas_status_success) {
        // In a real library, you might log rocblas_status_to_string(err)
        // fprintf(stderr, "rocBLAS Error during %s: %s
", operation, rocblas_status_to_string(err));
        // Assuming a generic failure status for now. Could map specific rocBLAS errors to rocqStatus_t.
        return ROCQ_STATUS_FAILURE; // Or a new ROCQ_STATUS_ROCBLAS_ERROR
    }
    return ROCQ_STATUS_SUCCESS;
}


extern "C" {

rocqStatus_t rocsvCreate(rocsvHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocsvInternalHandle* internal_handle = new(std::nothrow) rocsvInternalHandle;
    if (!internal_handle) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    internal_handle->stream = nullptr; // Initialize to prevent dangling pointer in cleanup
    internal_handle->blasHandle = nullptr; // Initialize

    hipError_t hip_err;
    rocblas_status blas_err;
    // rocqStatus_t status; // Not needed here as we return directly

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
    
    // TODO: Initialize rocRAND generator here if needed for measurement later
    
    *handle = internal_handle;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvDestroy(rocsvHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE; // Or SUCCESS if null handle is acceptable
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    rocqStatus_t first_error_status = ROCQ_STATUS_SUCCESS; 

    if (internal_handle->blasHandle) {
        rocblas_status blas_err = rocblas_destroy_handle(internal_handle->blasHandle);
        if (blas_err != rocblas_status_success) {
            first_error_status = checkRocblasError(blas_err, "rocsvDestroy rocblas_destroy_handle");
            // Log error but continue cleanup
        }
    }

    if (internal_handle->stream) {
        hipError_t hip_err = hipStreamDestroy(internal_handle->stream);
        // Only update status if no prior error from rocblas_destroy_handle
        if (hip_err != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) { 
            first_error_status = checkHipError(hip_err, "rocsvDestroy hipStreamDestroy");
            // Log error but continue cleanup
        }
    }

    // TODO: Destroy rocRAND generator
    
    delete internal_handle;
    return first_error_status; // Return status of first error encountered, or SUCCESS
}

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle, unsigned numQubits, rocComplex** d_state) {
    if (!handle || !d_state || numQubits == 0 || numQubits > 60) { // Max qubits check for safety
        return ROCQ_STATUS_INVALID_VALUE;
    }
    // rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle); // Not used yet

    size_t num_elements = 1ULL << numQubits;
    size_t size_bytes = num_elements * sizeof(rocComplex);

    hipError_t err = hipMalloc(d_state, size_bytes);
    if (err != hipSuccess) {
        *d_state = nullptr;
        return checkHipError(err, "rocsvAllocateState hipMalloc");
    }
    
    return ROCQ_STATUS_SUCCESS;
}

// Simple kernel to initialize state to |0...0>
__global__ void initializeToZeroStateKernel(rocComplex* state, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        state[0] = make_hipFloatComplex(1.0f, 0.0f);
    } else if (idx < num_elements) {
        state[idx] = make_hipFloatComplex(0.0f, 0.0f);
    }
}


rocqStatus_t rocsvInitializeState(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits) {
    if (!handle || !d_state || numQubits == 0 || numQubits > 60) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);

    size_t num_elements = 1ULL << numQubits;

    // Option 1: Using hipMemset for all elements to zero, then setting the first element.
    // This is often clearer and can be efficient.
    hipError_t err = hipMemsetAsync(d_state, 0, num_elements * sizeof(rocComplex), internal_handle->stream);
    if (err != hipSuccess) {
        return checkHipError(err, "rocsvInitializeState hipMemsetAsync");
    }

    rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
    err = hipMemcpyAsync(d_state, &zero_state_amplitude, sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->stream);
    if (err != hipSuccess) {
        return checkHipError(err, "rocsvInitializeState hipMemcpyAsync for first element");
    }
    
    // Wait for operations to complete on the stream
    err = hipStreamSynchronize(internal_handle->stream);
    if (err != hipSuccess) {
        return checkHipError(err, "rocsvInitializeState hipStreamSynchronize");
    }


    // Option 2: Using a custom kernel (defined above)
    // This might be slightly less efficient for this specific task than hipMemset + hipMemcpy
    // but demonstrates kernel usage. For this initial implementation, hipMemset is preferred.
    /*
    unsigned threads_per_block = 256;
    unsigned num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    hipLaunchKernelGGL(initializeToZeroStateKernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, num_elements);
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return checkHipError(err, "rocsvInitializeState hipLaunchKernelGGL(initializeToZeroStateKernel)");
    }
    err = hipStreamSynchronize(internal_handle->stream);
    if (err != hipSuccess) {
        return checkHipError(err, "rocsvInitializeState hipStreamSynchronize after kernel");
    }
    */

    return ROCQ_STATUS_SUCCESS;
}

// Forward declarations for kernels from single_qubit_kernels.hip
// These would typically be in an internal header, but for simplicity here:
__global__ void apply_single_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, const rocComplex* matrix);
__global__ void apply_X_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_Y_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_Z_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_H_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_S_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_T_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit);
__global__ void apply_Rx_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, float theta);
__global__ void apply_Ry_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, float theta);
__global__ void apply_Rz_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, float theta);

// Forward declarations for kernels from two_qubit_kernels.hip
__global__ void apply_two_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, unsigned qubit0_idx, unsigned qubit1_idx, const rocComplex* matrix_gpu);
__global__ void apply_CNOT_kernel(rocComplex* state, unsigned numQubits, unsigned controlQubit_idx, unsigned targetQubit_idx);
__global__ void apply_CZ_kernel(rocComplex* state, unsigned numQubits, unsigned qubit0_idx, unsigned qubit1_idx);
__global__ void apply_SWAP_kernel(rocComplex* state, unsigned numQubits, unsigned qubit0_idx, unsigned qubit1_idx);

// Forward declarations for measurement kernels
__global__ void calculate_prob0_kernel(const rocComplex* state, unsigned numQubits, unsigned targetQubit, double* d_prob0_sum);
__global__ void collapse_state_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, int measuredOutcome);
__global__ void sum_sq_magnitudes_kernel(const rocComplex* state, unsigned numQubits, double* d_sum_sq_mag);
__global__ void renormalize_state_kernel(rocComplex* state, unsigned numQubits, double d_sum_sq_mag_inv_sqrt);

// Forward declarations for kernels from multi_qubit_kernels.hip
__global__ void apply_three_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, const unsigned* targetQubitIndices_gpu, const rocComplex* matrixDevice);
__global__ void apply_four_qubit_generic_matrix_kernel(rocComplex* state, unsigned numQubits, const unsigned* targetQubitIndices_gpu, const rocComplex* matrixDevice);

// rocsvApplyMatrix now takes matrixDevice directly.
// The get_gate_type function and GateType enum are removed as per simplification.
// Dispatch will rely on numTargetQubits, and always use generic kernels.
// Specialized kernels (X, H, CNOT etc.) would typically be called via
// dedicated API functions (e.g., rocsvApplyX, rocsvApplyCNOT) which are not part of this scope.

rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle,
                              rocComplex* d_state,
                              unsigned numQubits,
                              const unsigned* qubitIndices, // HOST pointer to target qubit indices
                              unsigned numTargetQubits,    // Number of qubits the gate acts on
                              const rocComplex* matrixDevice, // Gate matrix on DEVICE memory
                              unsigned matrixDim) {
    if (!handle || !d_state || !qubitIndices || !matrixDevice) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numTargetQubits == 0) { // Should have at least one target qubit
        return ROCQ_STATUS_INVALID_VALUE;
    }
    // Validate matrixDim against numTargetQubits
    if (!((numTargetQubits == 1 && matrixDim == 2) || 
          (numTargetQubits == 2 && matrixDim == 4) ||
          (numTargetQubits == 3 && matrixDim == 8) ||
          (numTargetQubits == 4 && matrixDim == 16))) {
        // If it's not one of the valid combinations and not covered by the >4 case below
        if (numTargetQubits <= 4) return ROCQ_STATUS_INVALID_VALUE; 
    }
    
    if (numTargetQubits > 4) { // Currently not implemented for more than 4 target qubits
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;

    unsigned threads_per_block = 256; // A common choice, can be tuned
    size_t total_states = 1ULL << numQubits;

    if (numTargetQubits == 1) {
        unsigned targetQubit = qubitIndices[0];
        if (targetQubit >= numQubits) return ROCQ_STATUS_INVALID_VALUE;

        size_t num_thread_groups = total_states / 2;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1; // Ensure at least 1 block if work
        else if (num_thread_groups == 0) num_blocks = 0; // No work if numQubits=0 (total_states=1, num_thread_groups=0)
            
        hipLaunchKernelGGL(apply_single_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, targetQubit, matrixDevice);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) status = checkHipError(hip_err, "rocsvApplyMatrix hipLaunchKernelGGL (1Q)");

    } else if (numTargetQubits == 2) {
        if (qubitIndices[0] >= numQubits || qubitIndices[1] >= numQubits || qubitIndices[0] == qubitIndices[1]) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        unsigned q0 = (qubitIndices[0] < qubitIndices[1]) ? qubitIndices[0] : qubitIndices[1];
        unsigned q1 = (qubitIndices[0] < qubitIndices[1]) ? qubitIndices[1] : qubitIndices[0];

        size_t num_thread_groups = total_states / 4;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) num_blocks = 0; // No work if numQubits<2

        hipLaunchKernelGGL(apply_two_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, q0, q1, matrixDevice);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) status = checkHipError(hip_err, "rocsvApplyMatrix hipLaunchKernelGGL (2Q)");

    } else if (numTargetQubits == 3 || numTargetQubits == 4) {
        for (unsigned i = 0; i < numTargetQubits; ++i) {
            if (qubitIndices[i] >= numQubits) return ROCQ_STATUS_INVALID_VALUE;
            for (unsigned j = i + 1; j < numTargetQubits; ++j) {
                if (qubitIndices[i] == qubitIndices[j]) return ROCQ_STATUS_INVALID_VALUE; 
            }
        }

        unsigned* d_targetIndices = nullptr;
        hip_err = hipMalloc(&d_targetIndices, numTargetQubits * sizeof(unsigned));
        if (hip_err != hipSuccess) {
            return checkHipError(hip_err, "rocsvApplyMatrix hipMalloc d_targetIndices");
        }
        // Ensure stream synchronization for hipMemcpyAsync if d_targetIndices is used immediately by a kernel on a different stream,
        // but here it's the same stream. Kernel launch will be queued after memcpy.
        hip_err = hipMemcpyAsync(d_targetIndices, qubitIndices, numTargetQubits * sizeof(unsigned), hipMemcpyHostToDevice, internal_handle->stream);
        if (hip_err != hipSuccess) {
            hipFree(d_targetIndices); // Clean up allocated memory
            return checkHipError(hip_err, "rocsvApplyMatrix hipMemcpyAsync d_targetIndices");
        }

        size_t m_val = numTargetQubits;
        size_t num_kernel_threads = (numQubits < m_val) ? 0 : (total_states >> m_val); // N / (2^m)
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
            if (hip_err != hipSuccess) { // Kernel launch error
                 status = checkHipError(hip_err, "rocsvApplyMatrix hipLaunchKernelGGL (MQ)");
            }
            // No specific stream sync here for d_targetIndices before free, as hipFree syncs on the device.
            // However, if kernel failed to launch, d_targetIndices still needs freeing.
            // If status is already error, we might skip further operations or syncs.
        }
        
        // Free device memory for target indices
        // This hipFree will synchronize the device, ensuring kernel completes before memory is freed,
        // if the kernel was using d_targetIndices. This is a blocking call.
        // If async kernel and async free were desired, a different sync mechanism would be needed.
        hipError_t free_err = hipFree(d_targetIndices);
        if (free_err != hipSuccess && status == ROCQ_STATUS_SUCCESS) { 
            status = checkHipError(free_err, "rocsvApplyMatrix hipFree d_targetIndices");
        }
    }
    // else: numTargetQubits > 4 is already handled at the start.
    
    if (status == ROCQ_STATUS_SUCCESS) {
        hip_err = hipStreamSynchronize(internal_handle->stream);
        if (hip_err != hipSuccess) {
            status = checkHipError(hip_err, "rocsvApplyMatrix hipStreamSynchronize at end");
        }
    }
    return status;
}

// Actual implementation for rocsvMeasure
rocqStatus_t rocsvMeasure(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned qubitToMeasure,
                          int* h_outcome,         // Outcome back to host
                          double* h_probability) { // Probability of that outcome to host
    if (!handle || !d_state || !h_outcome || !h_probability || qubitToMeasure >= numQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numQubits == 0) return ROCQ_STATUS_INVALID_VALUE;

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;

    double* d_prob0_sum = nullptr;
    double h_prob0_sum = 0.0;

    // 1. Calculate probability of measuring 0 for the target qubit
    hip_err = hipMalloc(&d_prob0_sum, sizeof(double));
    if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvMeasure hipMalloc d_prob0_sum");

    // IMPORTANT: The calculate_prob0_kernel is a HACK (single-threaded sum).
    // A real version needs a proper parallel reduction.
    // For now, launch with 1 thread, 1 block for this HACKED kernel.
    hipLaunchKernelGGL(calculate_prob0_kernel, dim3(1), dim3(1), 0, internal_handle->stream,
                       d_state, numQubits, qubitToMeasure, d_prob0_sum);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(calculate_prob0_kernel)");
        hipFree(d_prob0_sum);
        return status;
    }
    hip_err = hipMemcpyAsync(&h_prob0_sum, d_prob0_sum, sizeof(double), hipMemcpyDeviceToHost, internal_handle->stream);
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipMemcpyAsync d_prob0_sum");
        hipFree(d_prob0_sum);
        return status;
    }
    hip_err = hipStreamSynchronize(internal_handle->stream); // Wait for calc and copy
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize after prob calc");
        hipFree(d_prob0_sum);
        return status;
    }
    hipFree(d_prob0_sum); // Done with d_prob0_sum

    double prob0 = h_prob0_sum;
    double prob1 = 1.0 - prob0;

    // Address potential floating point inaccuracies for probabilities
    if (prob0 < 0.0) prob0 = 0.0;
    if (prob0 > 1.0) prob0 = 1.0;
    if (prob1 < 0.0) prob1 = 0.0;
    if (prob1 > 1.0) prob1 = 1.0;


    // 2. Determine outcome using random number generator
    // Using hiprand for simplicity. rocRAND would be more robust.
    // TODO: Initialize hiprand generator in rocsvCreate (e.g. internal_handle->rand_generator)
    // and use it here. For now, using CPU rand() for simplicity of this step.
    // hiprandStatus_t rand_status = hiprandGenerateUniformDouble(internal_handle->rand_generator, d_rand_val, 1);
    
    static bool seeded = false; if (!seeded) { srand((unsigned int)time(NULL)); seeded = true; }
    double rand_val = (double)rand() / RAND_MAX;


    if (rand_val < prob0) {
        *h_outcome = 0;
        *h_probability = prob0;
    } else {
        *h_outcome = 1;
        *h_probability = prob1;
    }

    // 3. Collapse state vector
    unsigned threads_per_block_measure = 256;
    unsigned num_blocks_measure = ( (1ULL << numQubits) + threads_per_block_measure - 1) / threads_per_block_measure;
    if (numQubits == 0) num_blocks_measure = 0; 

    hipLaunchKernelGGL(collapse_state_kernel, dim3(num_blocks_measure), dim3(threads_per_block_measure), 0, internal_handle->stream,
                       d_state, numQubits, qubitToMeasure, *h_outcome);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        return checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(collapse_state_kernel)");
    }

    // 4. Re-normalize state vector
    double* d_sum_sq_mag = nullptr;
    double h_sum_sq_mag = 0.0;
    hip_err = hipMalloc(&d_sum_sq_mag, sizeof(double));
    if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvMeasure hipMalloc d_sum_sq_mag");

    // IMPORTANT: sum_sq_magnitudes_kernel is also a HACK (single-threaded sum).
    hipLaunchKernelGGL(sum_sq_magnitudes_kernel, dim3(1), dim3(1), 0, internal_handle->stream,
                       d_state, numQubits, d_sum_sq_mag);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(sum_sq_magnitudes_kernel)");
        hipFree(d_sum_sq_mag);
        return status;
    }
    hip_err = hipMemcpyAsync(&h_sum_sq_mag, d_sum_sq_mag, sizeof(double), hipMemcpyDeviceToHost, internal_handle->stream);
     if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipMemcpyAsync d_sum_sq_mag");
        hipFree(d_sum_sq_mag);
        return status;
    }
    hip_err = hipStreamSynchronize(internal_handle->stream); 
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize after sum_sq_mag calc");
        hipFree(d_sum_sq_mag);
        return status;
    }
    hipFree(d_sum_sq_mag);

    if (h_sum_sq_mag < 1e-12) { 
        if (*h_probability > 1e-9) { 
             // This is an unexpected scenario if prob of outcome was high.
        }
        return ROCQ_STATUS_SUCCESS; 
    }
    double norm_factor = 1.0 / sqrt(h_sum_sq_mag);

    hipLaunchKernelGGL(renormalize_state_kernel, dim3(num_blocks_measure), dim3(threads_per_block_measure), 0, internal_handle->stream,
                       d_state, numQubits, norm_factor);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        return checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(renormalize_state_kernel)");
    }

    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize at end");
}

} // extern "C"
