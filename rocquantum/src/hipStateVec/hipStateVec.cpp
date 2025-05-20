#include "rocquantum/hipStateVec.h" // Assuming this path will be set up in CMake include_directories
#include <hip/hip_runtime.h>
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
    int deviceId;
    // Add other resources like rocBLAS handle if needed later
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

extern "C" {

rocqStatus_t rocsvCreate(rocsvHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocsvInternalHandle* internal_handle = new(std::nothrow) rocsvInternalHandle;
    if (!internal_handle) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    hipError_t err = hipGetDevice(&internal_handle->deviceId);
    if (err != hipSuccess) {
        delete internal_handle;
        return checkHipError(err, "rocsvCreate hipGetDevice");
    }

    err = hipStreamCreate(&internal_handle->stream);
    if (err != hipSuccess) {
        delete internal_handle;
        return checkHipError(err, "rocsvCreate hipStreamCreate");
    }
    
    // TODO: Initialize rocRAND generator here if needed for measurement later
    // TODO: Initialize rocBLAS handle here if it's to be part of the handle

    *handle = internal_handle;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvDestroy(rocsvHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE; // Or SUCCESS if null handle is acceptable
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);

    if (internal_handle->stream) {
        hipError_t err = hipStreamDestroy(internal_handle->stream);
        // Log error but continue cleanup
        checkHipError(err, "rocsvDestroy hipStreamDestroy");
    }

    // TODO: Destroy rocRAND generator
    // TODO: Destroy rocBLAS handle

    delete internal_handle;
    return ROCQ_STATUS_SUCCESS;
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


// Helper to get matrix type (for specialized gates)
// This is a simplification; a real implementation might pass gate type enum
// or always use the generic kernel if matrix is arbitrary.
enum GateType {
    GATE_X, GATE_Y, GATE_Z, GATE_H, GATE_S, GATE_T,
    GATE_RX, GATE_RY, GATE_RZ,
    GATE_CNOT, GATE_CZ, GATE_SWAP,
    GATE_GENERIC_1Q, GATE_GENERIC_2Q, GATE_UNKNOWN
};

GateType get_gate_type(unsigned numTargetQubits, const rocComplex* matrix, unsigned matrixDim, float& angle) {
    // This is a placeholder. In a real system, you wouldn't typically deduce gate type from matrix.
    // Instead, the API would take a gate enum or the matrix itself would be the sole source of truth for generic kernels.
    // For this exercise, we'll make some simple checks for common known matrices to call specialized kernels.
    // Angle is an out-parameter for rotation gates.
    
    // Note: Comparing floating point numbers for equality is generally bad.
    // This is purely for enabling the dispatch to specialized kernels as per plan.
    // A robust solution would involve specific API entry points for named gates,
    // or passing an enum for the gate type if a common applyMatrix is used.

    if (numTargetQubits == 1 && matrixDim == 2) {
        // X: [[0,1],[1,0]] -> M[0]={0,0}, M[1]={1,0}, M[2]={1,0}, M[3]={0,0} (col-major)
        if (matrix[0].x == 0.f && matrix[0].y == 0.f &&
            matrix[1].x == 1.f && matrix[1].y == 0.f &&
            matrix[2].x == 1.f && matrix[2].y == 0.f &&
            matrix[3].x == 0.f && matrix[3].y == 0.f) return GATE_X;
        // H: 1/sqrt(2) * [[1,1],[1,-1]]
        float h_val = 1.f/sqrtf(2.f);
        if (fabsf(matrix[0].x - h_val) < 1e-6 && fabsf(matrix[0].y) < 1e-6 &&
            fabsf(matrix[1].x - h_val) < 1e-6 && fabsf(matrix[1].y) < 1e-6 &&
            fabsf(matrix[2].x - h_val) < 1e-6 && fabsf(matrix[2].y) < 1e-6 &&
            fabsf(matrix[3].x - (-h_val)) < 1e-6 && fabsf(matrix[3].y) < 1e-6) return GATE_H;
        // Could add more checks for Y, Z, S, T, Rx, Ry, Rz if specific matrices are passed...
        // For Rz(theta) = [[exp(-it/2),0],[0,exp(it/2)]]
        // M00 = cos(t/2)-i*sin(t/2), M11 = cos(t/2)+i*sin(t/2)
        // If M01 and M10 are zero
        if (matrix[2].x == 0.f && matrix[2].y == 0.f && matrix[1].x == 0.f && matrix[1].y == 0.f) {
            // Check if it's Rz
            // angle = 2 * acosf(matrix[0].x) or 2 * asinf(-matrix[0].y) if cos(t/2) > 0
            // This is tricky. Let's assume if no other known gate matches, it's generic.
            // For simplicity, we'll mostly rely on generic for rotations unless a dedicated API is made.
        }
        return GATE_GENERIC_1Q;
    }
    if (numTargetQubits == 2 && matrixDim == 4) {
        // CNOT (control=q1, target=q0): [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        // M[0]=1, M[1]=0, M[2]=0, M[3]=0
        // M[4]=0, M[5]=1, M[6]=0, M[7]=0
        // M[8]=0, M[9]=0, M[10]=0,M[11]=1
        // M[12]=0,M[13]=0, M[14]=1,M[15]=0
        bool is_cnot = matrix[0].x == 1.f && matrix[0].y == 0.f &&
                       matrix[1].x == 0.f && matrix[1].y == 0.f &&
                       matrix[2].x == 0.f && matrix[2].y == 0.f &&
                       matrix[3].x == 0.f && matrix[3].y == 0.f &&
                       matrix[4].x == 0.f && matrix[4].y == 0.f &&
                       matrix[5].x == 1.f && matrix[5].y == 0.f &&
                       matrix[6].x == 0.f && matrix[6].y == 0.f &&
                       matrix[7].x == 0.f && matrix[7].y == 0.f &&
                       matrix[8].x == 0.f && matrix[8].y == 0.f &&
                       matrix[9].x == 0.f && matrix[9].y == 0.f &&
                       matrix[10].x == 0.f && matrix[10].y == 0.f && // This was M[10]=0
                       matrix[11].x == 1.f && matrix[11].y == 0.f && // This was M[11]=1 (oops, this is M[2][3])
                                                                  // Correct CNOT matrix (Col Major):
                                                                  // 1 0 0 0
                                                                  // 0 1 0 0
                                                                  // 0 0 0 1
                                                                  // 0 0 1 0
                                                                  // M[0]=1, M[5]=1, M[11]=1 (this is M[3][2]), M[14]=1 (this is M[2][3])
                                                                  // The indices for CNOT matrix:
                                                                  // M[0]=1 (0,0)
                                                                  // M[5]=1 (1,1)
                                                                  // M[11]=1 (3,2) maps input 2 to output 3
                                                                  // M[14]=1 (2,3) maps input 3 to output 2
                       // Checking the provided CNOT matrix values directly:
                       matrix[0].x == 1.f && matrix[0].y == 0.f && /* M[0][0] */
                       matrix[1].x == 0.f && matrix[1].y == 0.f && /* M[1][0] */
                       matrix[2].x == 0.f && matrix[2].y == 0.f && /* M[2][0] */
                       matrix[3].x == 0.f && matrix[3].y == 0.f && /* M[3][0] */

                       matrix[4].x == 0.f && matrix[4].y == 0.f && /* M[0][1] */
                       matrix[5].x == 1.f && matrix[5].y == 0.f && /* M[1][1] */
                       matrix[6].x == 0.f && matrix[6].y == 0.f && /* M[2][1] */
                       matrix[7].x == 0.f && matrix[7].y == 0.f && /* M[3][1] */

                       matrix[8].x == 0.f && matrix[8].y == 0.f &&  /* M[0][2] */
                       matrix[9].x == 0.f && matrix[9].y == 0.f &&  /* M[1][2] */
                       matrix[10].x == 0.f && matrix[10].y == 0.f &&/* M[2][2] */ // This should be 0 for CNOT
                       matrix[11].x == 1.f && matrix[11].y == 0.f &&/* M[3][2] */ // This should be 1 for CNOT

                       matrix[12].x == 0.f && matrix[12].y == 0.f &&/* M[0][3] */
                       matrix[13].x == 0.f && matrix[13].y == 0.f &&/* M[1][3] */
                       matrix[14].x == 1.f && matrix[14].y == 0.f &&/* M[2][3] */ // This should be 1 for CNOT
                       matrix[15].x == 0.f && matrix[15].y == 0.f;  /* M[3][3] */ // This should be 0 for CNOT

        if (is_cnot) return GATE_CNOT;
        return GATE_GENERIC_2Q;
    }
    return GATE_UNKNOWN;
}


// Replacement for the rocsvApplyMatrix stub
rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle,
                              rocComplex* d_state,
                              unsigned numQubits,
                              const unsigned* qubitIndices, // Array of target qubit indices
                              unsigned numTargetQubits,    // Number of qubits the gate acts on (1 or 2)
                              const rocComplex* h_matrix,   // Gate matrix on HOST memory, column-major
                              unsigned matrixDim) {        // Dimension of the matrix (2 for 1Q, 4 for 2Q)
    if (!handle || !d_state || !qubitIndices || !h_matrix || numTargetQubits == 0 || numTargetQubits > 2 || matrixDim == 0) {
        // Current implementation only supports 1 or 2 target qubits.
        // numTargetQubits > 2 will be handled by rocBLAS or specialized kernels later.
        if (numTargetQubits > 2) return ROCQ_STATUS_NOT_IMPLEMENTED; // Placeholder for rocBLAS
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if ((numTargetQubits == 1 && matrixDim != 2) || (numTargetQubits == 2 && matrixDim != 4)) {
        return ROCQ_STATUS_INVALID_VALUE; // Matrix dimension mismatch
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    hipError_t hip_err;

    unsigned threads_per_block = 256; // Common choice
    size_t total_states = 1ULL << numQubits;

    if (numTargetQubits == 1) {
        unsigned targetQubit = qubitIndices[0];
        if (targetQubit >= numQubits) return ROCQ_STATUS_INVALID_VALUE;

        size_t num_thread_groups = total_states / 2; // Each thread handles a pair
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (numQubits == 0) num_blocks = 0; // Edge case for 0 qubits, num_thread_groups would be 0.5 -> 0


        float angle = 0.f; // Placeholder for rotation angles
        GateType type = get_gate_type(numTargetQubits, h_matrix, matrixDim, angle);

        rocComplex* d_matrix = nullptr;
        if (type == GATE_GENERIC_1Q) { 
            hip_err = hipMalloc(&d_matrix, matrixDim * matrixDim * sizeof(rocComplex));
            if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplyMatrix hipMalloc for d_matrix");
            hip_err = hipMemcpyAsync(d_matrix, h_matrix, matrixDim * matrixDim * sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->stream);
            if (hip_err != hipSuccess) {
                hipFree(d_matrix);
                return checkHipError(hip_err, "rocsvApplyMatrix hipMemcpyAsync for d_matrix");
            }
        }
        
        if (type == GATE_X) {
            hipLaunchKernelGGL(apply_X_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, targetQubit);
        } else if (type == GATE_H) {
            hipLaunchKernelGGL(apply_H_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, targetQubit);
        }
        else { 
             if (!d_matrix) { 
                hip_err = hipMalloc(&d_matrix, matrixDim * matrixDim * sizeof(rocComplex));
                if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplyMatrix hipMalloc for d_matrix (fallback)");
                hip_err = hipMemcpyAsync(d_matrix, h_matrix, matrixDim * matrixDim * sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->stream);
                if (hip_err != hipSuccess) {
                    hipFree(d_matrix);
                    return checkHipError(hip_err, "rocsvApplyMatrix hipMemcpyAsync for d_matrix (fallback)");
                }
            }
            hipLaunchKernelGGL(apply_single_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, targetQubit, d_matrix);
        }

        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) {
            if (d_matrix) hipFree(d_matrix);
            return checkHipError(hip_err, "rocsvApplyMatrix hipLaunchKernelGGL (1Q)");
        }
        if (d_matrix) { 
            hipStreamSynchronize(internal_handle->stream); 
            hipFree(d_matrix);
        }

    } else if (numTargetQubits == 2) {
        unsigned q0 = qubitIndices[0]; 
        unsigned q1 = qubitIndices[1];
        if (q0 >= numQubits || q1 >= numQubits || q0 == q1) return ROCQ_STATUS_INVALID_VALUE;
        // Sorting for generic kernel is done below. For specific kernels like CNOT, the order might matter based on API contract.

        size_t num_thread_groups = total_states / 4; 
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (numQubits < 2) num_blocks = 0; // Edge case for <2 qubits
        
        float angle = 0.f; 
        GateType type = get_gate_type(numTargetQubits, h_matrix, matrixDim, angle);

        rocComplex* d_matrix = nullptr;
        if (type == GATE_GENERIC_2Q) { 
            hip_err = hipMalloc(&d_matrix, matrixDim * matrixDim * sizeof(rocComplex));
            if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplyMatrix hipMalloc for d_matrix (2Q)");
            hip_err = hipMemcpyAsync(d_matrix, h_matrix, matrixDim * matrixDim * sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->stream);
            if (hip_err != hipSuccess) {
                hipFree(d_matrix);
                return checkHipError(hip_err, "rocsvApplyMatrix hipMemcpyAsync for d_matrix (2Q)");
            }
        }

        if (type == GATE_CNOT) {
            // The CNOT kernel expects controlQubit_idx and targetQubit_idx.
            // The get_gate_type CNOT check is for a matrix where q1 (higher index in pair) is control, q0 is target.
            // API contract: qubitIndices[0] is control, qubitIndices[1] is target by default.
            // If user provides [target, control], they must ensure matrix matches that, or API must be more explicit.
            // For now, let's assume API implies qubitIndices[0]=control, qubitIndices[1]=target
            // If the matrix provided (h_matrix) corresponds to CNOT(q1,q0) (q1=control)
            // then controlIdx should be qubitIndices[1] and targetIdx qubitIndices[0] IF qubitIndices was [q0,q1].
            // This part is tricky and depends on precise API definition vs matrix structure.
            // Let's assume the API defines qubitIndices[0] as control and qubitIndices[1] as target.
            unsigned controlIdx = qubitIndices[0];
            unsigned targetIdx = qubitIndices[1];

            // The CNOT kernel in two_qubit_kernels.hip is designed for (N/4) groups of threads,
            // where each thread processes a pair of amplitudes where the control bit is 1.
            // The number of such pairs is (total_states / 4) if control bit fixed, then /2 for pairs.
            // No, the CNOT kernel iterates (1ULL<<(numQubits-2)) times.
            unsigned cnot_work_items = (numQubits < 2) ? 0 : (1ULL << (numQubits - 2));
            unsigned cnot_num_blocks = (cnot_work_items + threads_per_block - 1) / threads_per_block;


            hipLaunchKernelGGL(apply_CNOT_kernel, dim3(cnot_num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, controlIdx, targetIdx);
        }
        else { 
            if (!d_matrix) { 
                hip_err = hipMalloc(&d_matrix, matrixDim * matrixDim * sizeof(rocComplex));
                if (hip_err != hipSuccess) return checkHipError(hip_err, "rocsvApplyMatrix hipMalloc for d_matrix (2Q fallback)");
                hip_err = hipMemcpyAsync(d_matrix, h_matrix, matrixDim * matrixDim * sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->stream);
                if (hip_err != hipSuccess) {
                    hipFree(d_matrix);
                    return checkHipError(hip_err, "rocsvApplyMatrix hipMemcpyAsync for d_matrix (2Q fallback)");
                }
            }
            unsigned sorted_q0 = (qubitIndices[0] < qubitIndices[1]) ? qubitIndices[0] : qubitIndices[1];
            unsigned sorted_q1 = (qubitIndices[0] < qubitIndices[1]) ? qubitIndices[1] : qubitIndices[0];
            hipLaunchKernelGGL(apply_two_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream, d_state, numQubits, sorted_q0, sorted_q1, d_matrix);
        }

        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) {
            if (d_matrix) hipFree(d_matrix);
            return checkHipError(hip_err, "rocsvApplyMatrix hipLaunchKernelGGL (2Q)");
        }
        if (d_matrix) {
            hipStreamSynchronize(internal_handle->stream);
            hipFree(d_matrix);
        }

    } else {
        return ROCQ_STATUS_NOT_IMPLEMENTED; 
    }
    
    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, "rocsvApplyMatrix hipStreamSynchronize");
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
    // TODO: Initialize hiprand generator in rocsvCreate and store in handle if we want sequences.
    // For a single call, this might be okay.
    
    // Quick and dirty for now, seed with time. NOT THREAD SAFE if multiple streams/calls.
    // A proper rocRAND setup per handle is needed.
    // hiprandState_t rand_state; 
    // hiprandCreateGenerator(&rand_state, HIPRAND_RNG_PSEUDO_DEFAULT);
    // hiprandSetPseudoRandomGeneratorSeed(rand_state, (unsigned long long)time(NULL)); // Bad seed for HPC
    // double* d_rand_val; 
    // hip_err = hipMalloc(&d_rand_val, sizeof(double));
    // // hiprandGenerateUniformDouble (rand_state, d_rand_val, 1); // This is a device call, result on device
    // // hiprandDestroyGenerator(rand_state); // Clean up temp generator
    // // hipMemcpy(&rand_val, d_rand_val, sizeof(double), hipMemcpyDeviceToHost, internal_handle->stream);
    // // hipFree(d_rand_val);
    // // hipStreamSynchronize(internal_handle->stream);
    
    // For this example, let's use a CPU-based random number.
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
    unsigned threads_per_block = 256;
    unsigned num_blocks = ( (1ULL << numQubits) + threads_per_block - 1) / threads_per_block;
    if (numQubits == 0) num_blocks = 0; // handle numQubits = 0 case for (1ULL << 0) = 1 state

    hipLaunchKernelGGL(collapse_state_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream,
                       d_state, numQubits, qubitToMeasure, *h_outcome);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        return checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(collapse_state_kernel)");
    }

    // 4. Re-normalize state vector
    // First, calculate new sum of squared magnitudes of the collapsed state
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
    hip_err = hipStreamSynchronize(internal_handle->stream); // Wait for calc and copy
    if (hip_err != hipSuccess) {
        status = checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize after sum_sq_mag calc");
        hipFree(d_sum_sq_mag);
        return status;
    }
    hipFree(d_sum_sq_mag);

    if (h_sum_sq_mag < 1e-12) { // Effectively zero, avoid division by zero. State might be all zeros.
        // This can happen if the measured state had zero probability.
        // Or if original state was not normalized. For now, just return.
        // A robust system might re-initialize or error.
        // If the probability of the measured outcome was itself ~0, then sum_sq_mag will be ~0.
        // In this case, the state is effectively zero and doesn't need renormalization.
        // Or, if *h_probability is very small, this is expected.
        if (*h_probability > 1e-9) { // Only if the probability of this outcome was non-trivial
             // This is an unexpected scenario if prob of outcome was high.
        }
        return ROCQ_STATUS_SUCCESS; 
    }
    double norm_factor = 1.0 / sqrt(h_sum_sq_mag);

    hipLaunchKernelGGL(renormalize_state_kernel, dim3(num_blocks), dim3(threads_per_block), 0, internal_handle->stream,
                       d_state, numQubits, norm_factor);
    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        return checkHipError(hip_err, "rocsvMeasure hipLaunchKernelGGL(renormalize_state_kernel)");
    }

    hip_err = hipStreamSynchronize(internal_handle->stream);
    return checkHipError(hip_err, "rocsvMeasure hipStreamSynchronize at end");
}

} // extern "C"
