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
#include <numeric> // For std::accumulate
#include "rccl.h"  // For RCCL operations

// Define the internal handle structure
struct rocsvInternalHandle {
    // Multi-GPU configuration
    int numGpus;                    // Number of GPUs participating in the simulation
    unsigned globalNumQubits;       // Total number of qubits in the global simulation state
    unsigned numLocalQubitsPerGpu;  // Number of qubits whose amplitudes are entirely on one GPU slice
                                    // (globalNumQubits - numGlobalSliceQubits)
    unsigned numGlobalSliceQubits;  // Number of qubits used to determine the GPU slice (log2(numGpus))

    // Per-GPU resources
    std::vector<int> deviceIds;                 // List of device IDs used
    std::vector<hipStream_t> streams;           // One stream per GPU
    std::vector<rocblas_handle> blasHandles;    // One rocBLAS handle per GPU
    std::vector<rcclComm_t> comms;              // One RCCL communicator per GPU

    // Distributed state vector data
    std::vector<rocComplex*> d_local_state_slices; // Pointer to device memory for each GPU's slice
    std::vector<size_t> localStateSizes;         // Number of amplitudes stored by each GPU's slice

    // Potentially keep a primary device/rank for certain operations or defaults, if needed.
    // For now, assuming operations will iterate or explicitly target a device/rank.
    // int primaryDeviceId; // Example if a concept of a 'main' GPU is retained.

    // Temporary buffers for data redistribution (e.g., Alltoallv)
    std::vector<rocComplex*> d_swap_buffers;    // One swap buffer per GPU, same size as its local_state_slice

    // hiprandGenerator_t rand_generator; // For future rocRAND integration (would also need to be per-GPU or managed)
    // Adding legacy single-GPU stream and blas handle for functions not yet fully multi-GPU aware
    // These will typically point to the resources of device 0 if numGpus > 0
    hipStream_t stream = nullptr; // Legacy single stream
    rocblas_handle blasHandle = nullptr; // Legacy single rocBLAS handle
    int localRank = -1; // Legacy: for single GPU, this might be 0 or -1. For multi-GPU, this is rank.
    size_t localStateSize = 0; // Legacy: for single GPU context
    // rocComplex* d_state_gpu = nullptr; // Legacy single d_state pointer, now part of d_local_state_slices[0] for single GPU.
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

// Helper to check RCCL errors and convert to rocqStatus_t
rocqStatus_t checkRcclError(rcclResult_t err, const char* operation = "") {
    if (err != rcclSuccess) {
        // fprintf(stderr, "RCCL Error during %s: %s\n", operation, rcclGetErrorString(err));
        return ROCQ_STATUS_RCCL_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

// Helper function to determine if all target qubits are "local" to each slice
// based on the current distribution scheme.
// "Local" means the qubit index falls within the `numLocalQubitsPerGpu` range.
static bool are_qubits_local(rocsvInternalHandle* h, const unsigned* qubitIndices, unsigned numTargetQubits) {
    if (!h || h->numGpus == 0) {
        return false; 
    }
    if (h->numGpus == 1) { // If only one GPU, all qubits are effectively local from its perspective.
        return true;
    }
    // For multi-GPU, a qubit is local if its global index is less than numLocalQubitsPerGpu.
    // This assumes qubitIndices are global indices.
    // The definition of "local" here means that the gate, when applied to any slice,
    // will only involve qubits whose indices are < numLocalQubitsPerGpu (i.e., not slice-determining bits).
    for (unsigned i = 0; i < numTargetQubits; ++i) {
        if (qubitIndices[i] >= h->numLocalQubitsPerGpu) {
            return false;
        }
    }
    return true;
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

// Original placeholder measurement kernels (still used by single-GPU path in rocsvMeasure for now)
__global__ void calculate_prob0_kernel(const rocComplex* state, unsigned numQubits, unsigned targetQubit, double* d_prob0_sum);
__global__ void sum_sq_magnitudes_kernel(const rocComplex* state, unsigned numQubits, double* d_sum_sq_mag);

// Common measurement utility kernels (used by both single and multi-GPU paths after refactor)
__global__ void collapse_state_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, int measuredOutcome);
__global__ void renormalize_state_kernel(rocComplex* state, unsigned numQubits, double d_sum_sq_mag_inv_sqrt);

// New kernels for multi-GPU measurement reduction
__global__ void calculate_local_slice_probabilities_kernel(
    const rocComplex* local_slice_data,
    size_t local_slice_num_elements,
    unsigned num_local_qubits,
    unsigned local_target_qubit,
    double* d_block_partial_probs);

__global__ void calculate_local_slice_sum_sq_mag_kernel(
    const rocComplex* local_slice_data,
    size_t local_slice_num_elements,
    double* d_block_sum_sq_mag);


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
    // Initialize legacy single-GPU members
    internal_handle->stream = nullptr;
    internal_handle->blasHandle = nullptr;
    internal_handle->localRank = -1;
    internal_handle->localStateSize = 0;

    // Initialize multi-GPU members
    internal_handle->numGpus = 0;
    internal_handle->globalNumQubits = 0;
    internal_handle->numLocalQubitsPerGpu = 0;
    internal_handle->numGlobalSliceQubits = 0;

    hipError_t hip_err;
    rocblas_status blas_err;
    rcclResult_t rccl_err;

    int device_count = 0;
    hip_err = hipGetDeviceCount(&device_count);
    if (hip_err != hipSuccess) {
        delete internal_handle;
        return checkHipError(hip_err, "rocsvCreate hipGetDeviceCount");
    }
    if (device_count <= 0) { // No GPUs or error
        delete internal_handle;
        return ROCQ_STATUS_FAILURE;
    }
    internal_handle->numGpus = device_count;

    try {
        internal_handle->deviceIds.resize(internal_handle->numGpus);
        internal_handle->streams.resize(internal_handle->numGpus);
        internal_handle->blasHandles.resize(internal_handle->numGpus);
        internal_handle->comms.resize(internal_handle->numGpus);
        internal_handle->d_local_state_slices.resize(internal_handle->numGpus, nullptr);
        internal_handle->localStateSizes.resize(internal_handle->numGpus, 0);
        internal_handle->d_swap_buffers.resize(internal_handle->numGpus, nullptr);
    } catch (const std::bad_alloc& e) {
        delete internal_handle;
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    rcclUniqueId uniqueId;
    if (internal_handle->numGpus > 0) {
        if (rcclGetUniqueId(&uniqueId) != rcclSuccess) {
             delete internal_handle;
             return ROCQ_STATUS_RCCL_ERROR;
        }
    }

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        internal_handle->deviceIds[i] = i;
        
        hip_err = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err != hipSuccess) {
            for (int j = 0; j < i; ++j) {
                if (internal_handle->comms[j]) rcclCommDestroy(internal_handle->comms[j]);
                if (internal_handle->blasHandles[j]) rocblas_destroy_handle(internal_handle->blasHandles[j]);
                if (internal_handle->streams[j]) hipStreamDestroy(internal_handle->streams[j]);
            }
            delete internal_handle;
            return checkHipError(hip_err, "rocsvCreate hipSetDevice");
        }

        hip_err = hipStreamCreate(&internal_handle->streams[i]);
        if (hip_err != hipSuccess) {
             for (int j = 0; j < i; ++j) { /* ... */ }
             if(internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]);
             delete internal_handle; return checkHipError(hip_err, "rocsvCreate hipStreamCreate");
        }

        blas_err = rocblas_create_handle(&internal_handle->blasHandles[i]);
        if (blas_err != rocblas_status_success) {
            for (int j = 0; j < i; ++j) { /* ... */ }
            if(internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]);
            if(internal_handle->blasHandles[i]) rocblas_destroy_handle(internal_handle->blasHandles[i]);
            delete internal_handle; return checkRocblasError(blas_err, "rocsvCreate rocblas_create_handle");
        }

        blas_err = rocblas_set_stream(internal_handle->blasHandles[i], internal_handle->streams[i]);
        if (blas_err != rocblas_status_success) {
            for (int j = 0; j < i; ++j) { /* ... */ }
            if(internal_handle->blasHandles[i]) rocblas_destroy_handle(internal_handle->blasHandles[i]);
            if(internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]);
            delete internal_handle; return checkRocblasError(blas_err, "rocsvCreate rocblas_set_stream");
        }
        
        rccl_err = rcclCommInitRank(&internal_handle->comms[i], internal_handle->numGpus, uniqueId, i);
        if (rccl_err != rcclSuccess) {
            for (int j = 0; j < i; ++j) { /* ... */ }
            if(internal_handle->comms[i]) rcclCommDestroy(internal_handle->comms[i]);
            if(internal_handle->blasHandles[i]) rocblas_destroy_handle(internal_handle->blasHandles[i]);
            if(internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]);
            delete internal_handle; return checkRcclError(rccl_err, "rocsvCreate rcclCommInitRank");
        }

        if (i == 0) {
            internal_handle->stream = internal_handle->streams[0];
            internal_handle->blasHandle = internal_handle->blasHandles[0];
            internal_handle->localRank = 0;
        }
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

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_destroy = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_destroy != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
            first_error_status = checkHipError(hip_err_destroy, "rocsvDestroy hipSetDevice");
        }

        if (internal_handle->comms[i]) {
            rcclResult_t rccl_err_destroy = rcclCommDestroy(internal_handle->comms[i]);
            if (rccl_err_destroy != rcclSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
                first_error_status = checkRcclError(rccl_err_destroy, "rocsvDestroy rcclCommDestroy");
            }
        }
        if (internal_handle->blasHandles[i]) {
            rocblas_status blas_err_destroy = rocblas_destroy_handle(internal_handle->blasHandles[i]);
            if (blas_err_destroy != rocblas_status_success && first_error_status == ROCQ_STATUS_SUCCESS) {
                first_error_status = checkRocblasError(blas_err_destroy, "rocsvDestroy rocblas_destroy_handle");
            }
        }
        if (internal_handle->streams[i]) {
            hipError_t stream_err_destroy = hipStreamDestroy(internal_handle->streams[i]);
            if (stream_err_destroy != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
                first_error_status = checkHipError(stream_err_destroy, "rocsvDestroy hipStreamDestroy");
            }
        }
        if (internal_handle->d_local_state_slices[i]) {
            hipError_t free_err = hipFree(internal_handle->d_local_state_slices[i]);
            if (free_err != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
                first_error_status = checkHipError(free_err, "rocsvDestroy hipFree d_local_state_slices");
            }
            internal_handle->d_local_state_slices[i] = nullptr; 
        }
        if (internal_handle->d_swap_buffers[i]) {
            hipError_t free_err_swap = hipFree(internal_handle->d_swap_buffers[i]);
            if (free_err_swap != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
                first_error_status = checkHipError(free_err_swap, "rocsvDestroy hipFree d_swap_buffers");
            }
            internal_handle->d_swap_buffers[i] = nullptr;
        }
    }

    internal_handle->deviceIds.clear();
    internal_handle->streams.clear();
    internal_handle->blasHandles.clear();
    internal_handle->comms.clear();
    internal_handle->d_local_state_slices.clear();
    internal_handle->localStateSizes.clear();
    internal_handle->d_swap_buffers.clear();
    
    delete internal_handle;
    return first_error_status;
}

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle, unsigned numQubits, rocComplex** d_state) {
    if (!handle || !d_state || numQubits > 60) {
        if (numQubits == 0 && d_state == nullptr) return ROCQ_STATUS_INVALID_VALUE;
        if (numQubits > 60) return ROCQ_STATUS_INVALID_VALUE;
    }

    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    if (internal_handle->numGpus == 0 || internal_handle->deviceIds.empty() || internal_handle->streams.empty()) {
        return ROCQ_STATUS_FAILURE;
    }

    hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[0]);
    if (hip_err_set != hipSuccess) return checkHipError(hip_err_set, "rocsvAllocateState hipSetDevice");

    size_t num_elements = 1ULL << numQubits;
    size_t size_bytes = num_elements * sizeof(rocComplex);
    
    if (internal_handle->d_local_state_slices[0] != nullptr) {
        hipFree(internal_handle->d_local_state_slices[0]);
        internal_handle->d_local_state_slices[0] = nullptr;
    }
    if (internal_handle->d_swap_buffers[0] != nullptr) {
        hipFree(internal_handle->d_swap_buffers[0]);
        internal_handle->d_swap_buffers[0] = nullptr;
    }

    hipError_t err = hipMalloc(&internal_handle->d_local_state_slices[0], size_bytes);
    if (err != hipSuccess) {
        *d_state = nullptr;
        internal_handle->d_local_state_slices[0] = nullptr;
        return checkHipError(err, "rocsvAllocateState hipMalloc");
    }
    hipError_t err_swap = hipMalloc(&internal_handle->d_swap_buffers[0], size_bytes);
    if (err_swap != hipSuccess) {
        hipFree(internal_handle->d_local_state_slices[0]);
        internal_handle->d_local_state_slices[0] = nullptr;
        *d_state = nullptr;
        return checkHipError(err_swap, "rocsvAllocateState hipMalloc swap_buffer");
    }

    internal_handle->localStateSizes[0] = num_elements;
    *d_state = internal_handle->d_local_state_slices[0];

    internal_handle->globalNumQubits = numQubits;
    internal_handle->numGlobalSliceQubits = 0;
    internal_handle->numLocalQubitsPerGpu = numQubits; 
    internal_handle->localStateSize = num_elements;

    return ROCQ_STATUS_SUCCESS;
}


rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle, unsigned totalNumQubits) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);

    if (internal_handle->numGpus == 0) return ROCQ_STATUS_FAILURE;

    if ((internal_handle->numGpus > 1) && ((internal_handle->numGpus & (internal_handle->numGpus - 1)) != 0)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (totalNumQubits == 0 && internal_handle->numGpus > 1) return ROCQ_STATUS_INVALID_VALUE;
    if (totalNumQubits > 60) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    
    unsigned num_global_slice_qubits = 0;
    if (internal_handle->numGpus > 1) {
        num_global_slice_qubits = static_cast<unsigned>(std::log2(internal_handle->numGpus));
    }

    if (totalNumQubits < num_global_slice_qubits && internal_handle->numGpus > 1) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    internal_handle->globalNumQubits = totalNumQubits;
    internal_handle->numGlobalSliceQubits = num_global_slice_qubits;
    internal_handle->numLocalQubitsPerGpu = (totalNumQubits >= num_global_slice_qubits) ? (totalNumQubits - num_global_slice_qubits) : 0;

    if (internal_handle->numGpus == 1) {
        internal_handle->numLocalQubitsPerGpu = totalNumQubits;
        internal_handle->numGlobalSliceQubits = 0;
    }
    
    size_t sliceNumElements = 1ULL << internal_handle->numLocalQubitsPerGpu;
    size_t sliceSizeBytes = sliceNumElements * sizeof(rocComplex);

    rocqStatus_t status = ROCQ_STATUS_SUCCESS;

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_set != hipSuccess) {
            status = checkHipError(hip_err_set, "rocsvAllocateDistributedState hipSetDevice");
            for (int j = 0; j < i; ++j) {
                if (internal_handle->d_local_state_slices[j]) { hipSetDevice(internal_handle->deviceIds[j]); hipFree(internal_handle->d_local_state_slices[j]); internal_handle->d_local_state_slices[j] = nullptr; }
                if (internal_handle->d_swap_buffers[j]) { hipSetDevice(internal_handle->deviceIds[j]); hipFree(internal_handle->d_swap_buffers[j]); internal_handle->d_swap_buffers[j] = nullptr; }
            }
            return status; 
        }

        if (internal_handle->d_local_state_slices[i] != nullptr) { hipFree(internal_handle->d_local_state_slices[i]); internal_handle->d_local_state_slices[i] = nullptr; }
        if (internal_handle->d_swap_buffers[i] != nullptr) { hipFree(internal_handle->d_swap_buffers[i]); internal_handle->d_swap_buffers[i] = nullptr; }

        if (sliceSizeBytes > 0) {
            hipError_t hip_err_alloc = hipMalloc(&internal_handle->d_local_state_slices[i], sliceSizeBytes);
            if (hip_err_alloc != hipSuccess) {
                internal_handle->d_local_state_slices[i] = nullptr;
                status = checkHipError(hip_err_alloc, "rocsvAllocateDistributedState hipMalloc slice");
                for (int k = 0; k < i; ++k) { if(internal_handle->d_local_state_slices[k]) hipFree(internal_handle->d_local_state_slices[k]); if(internal_handle->d_swap_buffers[k]) hipFree(internal_handle->d_swap_buffers[k]); } return status;
            }
            internal_handle->localStateSizes[i] = sliceNumElements;

            hipError_t hip_err_alloc_swap = hipMalloc(&internal_handle->d_swap_buffers[i], sliceSizeBytes);
            if (hip_err_alloc_swap != hipSuccess) {
                internal_handle->d_swap_buffers[i] = nullptr;
                hipFree(internal_handle->d_local_state_slices[i]);
                internal_handle->d_local_state_slices[i] = nullptr;
                status = checkHipError(hip_err_alloc_swap, "rocsvAllocateDistributedState hipMalloc swap_buffer");
                 for (int k = 0; k < i; ++k) { if(internal_handle->d_local_state_slices[k]) hipFree(internal_handle->d_local_state_slices[k]); if(internal_handle->d_swap_buffers[k]) hipFree(internal_handle->d_swap_buffers[k]); } return status;
            }
        } else {
            internal_handle->d_local_state_slices[i] = nullptr;
            internal_handle->d_swap_buffers[i] = nullptr;
            internal_handle->localStateSizes[i] = 0;
        }
    }
    if (internal_handle->numGpus == 1) {
        internal_handle->localStateSize = sliceNumElements;
    }
    return status;
}

rocqStatus_t rocsvInitializeState(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits) {
    if (!handle || !d_state || numQubits > 60 ) {
        if (numQubits == 0 && d_state == nullptr) return ROCQ_STATUS_INVALID_VALUE;
        if (numQubits > 60) return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    if (internal_handle->numGpus == 0 || internal_handle->streams.empty() || 
        internal_handle->d_local_state_slices.empty() || internal_handle->d_local_state_slices[0] != d_state) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    
    hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[0]);
    if (hip_err_set != hipSuccess) return checkHipError(hip_err_set, "rocsvInitializeState hipSetDevice");

    size_t num_elements = 1ULL << numQubits;
    if (internal_handle->localStateSizes[0] != num_elements) {
         if (!(numQubits == 0 && internal_handle->localStateSizes[0] == 1 && num_elements == 1)){
            return ROCQ_STATUS_INVALID_VALUE;
         }
    }
    if (num_elements == 0 && numQubits > 0) return ROCQ_STATUS_INVALID_VALUE;


    hipError_t err = hipMemsetAsync(internal_handle->d_local_state_slices[0], 0, num_elements * sizeof(rocComplex), internal_handle->streams[0]);
    if (err != hipSuccess) {
        return checkHipError(err, "rocsvInitializeState hipMemsetAsync");
    }
    if (num_elements > 0) { 
        rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
        err = hipMemcpyAsync(internal_handle->d_local_state_slices[0], &zero_state_amplitude, sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->streams[0]);
        if (err != hipSuccess) {
            return checkHipError(err, "rocsvInitializeState hipMemcpyAsync for first element");
        }
    }
    err = hipStreamSynchronize(internal_handle->streams[0]);
    return checkHipError(err, "rocsvInitializeState hipStreamSynchronize");
}


rocqStatus_t rocsvInitializeDistributedState(rocsvHandle_t handle) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);

    bool is_zero_qubit_single_gpu_case = (internal_handle->globalNumQubits == 0 && internal_handle->numGpus == 1 && internal_handle->numLocalQubitsPerGpu == 0);
    if (internal_handle->numGpus == 0 || (internal_handle->globalNumQubits == 0 && !is_zero_qubit_single_gpu_case)) {
            return ROCQ_STATUS_FAILURE;
    }
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_set != hipSuccess) {
            return checkHipError(hip_err_set, "rocsvInitializeDistributedState hipSetDevice");
        }
        if (internal_handle->localStateSizes[i] > 0 && internal_handle->d_local_state_slices[i] == nullptr ) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (internal_handle->localStateSizes[i] > 0) {
            hipError_t hip_err_memset = hipMemsetAsync(internal_handle->d_local_state_slices[i], 0,
                                                   internal_handle->localStateSizes[i] * sizeof(rocComplex),
                                                   internal_handle->streams[i]);
            if (hip_err_memset != hipSuccess) {
                return checkHipError(hip_err_memset, "rocsvInitializeDistributedState hipMemsetAsync");
            }
        }
    }

    if (internal_handle->numGpus > 0) {
        hipError_t hip_err_set_dev0 = hipSetDevice(internal_handle->deviceIds[0]);
        if (hip_err_set_dev0 != hipSuccess) {
             return checkHipError(hip_err_set_dev0, "rocsvInitializeDistributedState hipSetDevice for dev0");
        }

        if (internal_handle->d_local_state_slices[0] != nullptr && internal_handle->localStateSizes[0] > 0) {
            rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
            hipError_t hip_err_memcpy = hipMemcpyAsync(internal_handle->d_local_state_slices[0], &zero_state_amplitude, 
                                                   sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->streams[0]);
            if (hip_err_memcpy != hipSuccess) {
                return checkHipError(hip_err_memcpy, "rocsvInitializeDistributedState hipMemcpyAsync for zero state");
            }
        } else if (is_zero_qubit_single_gpu_case && internal_handle->d_local_state_slices[0] != nullptr && internal_handle->localStateSizes[0] == 1) {
             rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
             hipError_t hip_err_memcpy = hipMemcpyAsync(internal_handle->d_local_state_slices[0], &zero_state_amplitude,
                                                   sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->streams[0]);
            if (hip_err_memcpy != hipSuccess) {
                return checkHipError(hip_err_memcpy, "rocsvInitializeDistributedState hipMemcpyAsync for 0-qubit state");
            }
        }
    }
    
    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
         if (hip_err_set != hipSuccess) {
            status = checkHipError(hip_err_set, "rocsvInitializeDistributedState hipSetDevice for sync");
        }
        hipError_t hip_err_sync = hipStreamSynchronize(internal_handle->streams[i]);
        if (hip_err_sync != hipSuccess && status == ROCQ_STATUS_SUCCESS) {
            status = checkHipError(hip_err_sync, "rocsvInitializeDistributedState hipStreamSynchronize");
        }
    }
    return status;
}


rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle,
                              rocComplex* d_state_legacy,
                              unsigned globalNumQubits_param, 
                              const unsigned* qubitIndices,
                              unsigned numTargetQubits,    
                              const rocComplex* matrixDevice,
                              unsigned matrixDim) {        
    if (!handle || !qubitIndices || !matrixDevice) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (h->numGpus == 0 || h->d_local_state_slices.empty()) {
        return ROCQ_STATUS_FAILURE; 
    }
    unsigned currentGlobalNumQubits = (h->globalNumQubits > 0) ? h->globalNumQubits : globalNumQubits_param;
    if (currentGlobalNumQubits == 0 && numTargetQubits > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (currentGlobalNumQubits == 0 && numTargetQubits == 0 && matrixDim == 1) { return ROCQ_STATUS_SUCCESS;}


    if (numTargetQubits == 0 && matrixDim == 1) return ROCQ_STATUS_SUCCESS;
    if (numTargetQubits == 0 && matrixDim != 1) return ROCQ_STATUS_INVALID_VALUE;
    if (matrixDim != (1U << numTargetQubits)) return ROCQ_STATUS_INVALID_VALUE;

    for (unsigned i = 0; i < numTargetQubits; ++i) {
        if (qubitIndices[i] >= currentGlobalNumQubits) return ROCQ_STATUS_INVALID_VALUE;
        for (unsigned j = i + 1; j < numTargetQubits; ++j) {
            if (qubitIndices[i] == qubitIndices[j]) return ROCQ_STATUS_INVALID_VALUE; 
        }
    }

    hipError_t hip_err;
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    unsigned threads_per_block = 256;

    if (are_qubits_local(h, qubitIndices, numTargetQubits)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);

        std::vector<unsigned> local_qubit_indices_vec(qubitIndices, qubitIndices + numTargetQubits);

        for (int rank = 0; rank < h->numGpus; ++rank) {
            hip_err = hipSetDevice(h->deviceIds[rank]);
            if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix hipSetDevice (Local)"); break; }

            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->numGpus == 1 && h->localStateSizes[rank] != (1ULL << currentGlobalNumQubits) ) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (local_slice_num_elements > 0 && h->d_local_state_slices[rank] == nullptr) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (local_slice_num_elements == 0 && numTargetQubits > 0) { continue;}
            if (local_slice_num_elements == 0 && numTargetQubits == 0) { continue;}


            rocComplex* current_local_slice_ptr = h->d_local_state_slices[rank];
            unsigned* d_target_indices_gpu = nullptr;

            if (numTargetQubits == 1) {
                unsigned targetQubitLocal = local_qubit_indices_vec[0];
                size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
                unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
                if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
                else if (num_thread_groups == 0 && local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) {
                     if (local_num_qubits_for_kernel == 0) num_blocks = 0;
                } else if (num_thread_groups == 0) num_blocks = 0;
                
                if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                    hipLaunchKernelGGL(apply_single_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                       current_local_slice_ptr, local_num_qubits_for_kernel, targetQubitLocal, matrixDevice);
                    hip_err = hipGetLastError();
                    if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix (1Q Local Kernel)"); break;}
                }
            } else if (numTargetQubits == 2) {
                unsigned q0_local = local_qubit_indices_vec[0];
                unsigned q1_local = local_qubit_indices_vec[1];

                size_t num_thread_groups = (local_num_qubits_for_kernel >=2) ? local_slice_num_elements / 4 : 0;
                unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
                if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1; else if (num_thread_groups == 0) num_blocks = 0;

                if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                    hipLaunchKernelGGL(apply_two_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                       current_local_slice_ptr, local_num_qubits_for_kernel, q0_local, q1_local, matrixDevice);
                    hip_err = hipGetLastError();
                    if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix (2Q Local Kernel)"); break;}
                }
            } else if (numTargetQubits == 3 || numTargetQubits == 4) {
                hip_err = hipMalloc(&d_target_indices_gpu, numTargetQubits * sizeof(unsigned));
                if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix hipMalloc d_target_indices_gpu (Local)"); break;}
                hip_err = hipMemcpyAsync(d_target_indices_gpu, local_qubit_indices_vec.data(), numTargetQubits * sizeof(unsigned), hipMemcpyHostToDevice, h->streams[rank]);
                if (hip_err != hipSuccess) { hipFree(d_target_indices_gpu); status = checkHipError(hip_err, "rocsvApplyMatrix hipMemcpyAsync d_target_indices_gpu (Local)"); break;}
                
                size_t m_val = numTargetQubits;
                size_t num_kernel_threads = (local_num_qubits_for_kernel < m_val) ? 0 : (local_slice_num_elements >> m_val);
                unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
                if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1; if (num_kernel_threads == 0) num_blocks = 0;

                if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                    if (numTargetQubits == 3) {
                        hipLaunchKernelGGL(apply_three_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                           current_local_slice_ptr, local_num_qubits_for_kernel, d_target_indices_gpu, matrixDevice);
                    } else { 
                        hipLaunchKernelGGL(apply_four_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                           current_local_slice_ptr, local_num_qubits_for_kernel, d_target_indices_gpu, matrixDevice);
                    }
                    hip_err = hipGetLastError();
                    if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix (3Q/4Q Local Kernel)");}
                }
                hipError_t sync_err_kernel = hipStreamSynchronize(h->streams[rank]);
                if (sync_err_kernel != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = checkHipError(sync_err_kernel, "rocsvApplyMatrix sync for d_target_indices_gpu (Local)");
                hipFree(d_target_indices_gpu);
                if(status != ROCQ_STATUS_SUCCESS) break;

            } else if (numTargetQubits >= 5) { 
                const unsigned MAX_M_FOR_GATHER_SCATTER = 10; 
                if (numTargetQubits > MAX_M_FOR_GATHER_SCATTER) { status = ROCQ_STATUS_NOT_IMPLEMENTED; break; }
                 if (h->localStateSizes[rank] == 0) continue;

                unsigned m = numTargetQubits;
                std::vector<unsigned> h_sorted_local_indices(local_qubit_indices_vec.begin(), local_qubit_indices_vec.end());
                std::sort(h_sorted_local_indices.begin(), h_sorted_local_indices.end());

                unsigned* d_targetIndices_gs = nullptr;
                rocComplex* d_temp_vec_in = nullptr;
                rocComplex* d_temp_vec_out = nullptr;
                bool gs_error_rank = false;

                hip_err = hipMalloc(&d_targetIndices_gs, m * sizeof(unsigned));
                if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_targetIndices_gs malloc (m>=5 Local)"); gs_error_rank = true; }
                else {
                    hip_err = hipMemcpyAsync(d_targetIndices_gs, h_sorted_local_indices.data(), m * sizeof(unsigned), hipMemcpyHostToDevice, h->streams[rank]);
                    if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_targetIndices_gs memcpy (m>=5 Local)"); gs_error_rank = true; }
                }
                if (!gs_error_rank) {
                    hip_err = hipMalloc(&d_temp_vec_in, matrixDim * sizeof(rocComplex));
                    if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_temp_vec_in malloc (m>=5 Local)"); gs_error_rank = true; }
                }
                if (!gs_error_rank) {
                    hip_err = hipMalloc(&d_temp_vec_out, matrixDim * sizeof(rocComplex));
                    if (hip_err != hipSuccess) { status = checkHipError(hip_err, "d_temp_vec_out malloc (m>=5 Local)"); gs_error_rank = true; }
                }

                if (gs_error_rank) { 
                    if(d_targetIndices_gs) hipFree(d_targetIndices_gs);
                    if(d_temp_vec_in) hipFree(d_temp_vec_in);
                    if(d_temp_vec_out) hipFree(d_temp_vec_out);
                    if (status == ROCQ_STATUS_SUCCESS) status = ROCQ_STATUS_ALLOCATION_FAILED;
                    break; 
                }
                
                rocblas_float_complex alpha_gemv = {1.0f, 0.0f};
                rocblas_float_complex beta_gemv = {0.0f, 0.0f};
                unsigned num_non_target_qubits_local = (local_num_qubits_for_kernel >= m) ? (local_num_qubits_for_kernel - m) : 0;
                size_t num_non_target_configs_local = 1ULL << num_non_target_qubits_local;
                if (local_num_qubits_for_kernel < m) num_non_target_configs_local = 0;
                
                unsigned gs_threads = 256; 
                if (matrixDim < gs_threads && matrixDim > 0) gs_threads = matrixDim; else if (matrixDim == 0 && numTargetQubits == 0) gs_threads = 1;  else if (matrixDim == 0) gs_threads=1;
                unsigned gs_blocks = (matrixDim + gs_threads - 1) / gs_threads;
                if (matrixDim == 0 && numTargetQubits == 0) gs_blocks = 0; else if (gs_blocks == 0 && matrixDim > 0) gs_blocks = 1;


                for (size_t j = 0; j < num_non_target_configs_local; ++j) {
                    if (status != ROCQ_STATUS_SUCCESS) break;
                    size_t base_idx_non_targets = 0; 
                    unsigned current_non_target_bit_pos = 0;
                    for (unsigned bit_idx = 0; bit_idx < local_num_qubits_for_kernel; ++bit_idx) {
                        bool is_target = false;
                        for(unsigned k=0; k<m; ++k) if(h_sorted_local_indices[k] == bit_idx) {is_target = true; break;}
                        if (!is_target) {
                            if (((j >> current_non_target_bit_pos) & 1)) base_idx_non_targets |= (1ULL << bit_idx);
                            current_non_target_bit_pos++;
                        }
                    }

                    if (gs_blocks > 0) {
                        hipLaunchKernelGGL(gather_elements_kernel_v2, dim3(gs_blocks), dim3(gs_threads), 0, h->streams[rank],
                                           d_temp_vec_in, current_local_slice_ptr, d_targetIndices_gs, m, base_idx_non_targets);
                        if (hipGetLastError() != hipSuccess) { status = checkHipError(hipGetLastError(), "gather_elements_kernel_v2 (Local)"); break; }
                    
                        rocblas_status blas_status = rocblas_cgemv(h->blasHandles[rank], rocblas_operation_none,
                                           matrixDim, matrixDim, &alpha_gemv,
                                           (const rocblas_float_complex*)matrixDevice, matrixDim,
                                           (const rocblas_float_complex*)d_temp_vec_in, 1, &beta_gemv,
                                           (rocblas_float_complex*)d_temp_vec_out, 1);
                        if (blas_status != rocblas_status_success) { status = checkRocblasError(blas_status, "rocblas_cgemv (m>=5 Local)"); break; }
                    
                        hipLaunchKernelGGL(scatter_elements_kernel_v2, dim3(gs_blocks), dim3(gs_threads), 0, h->streams[rank],
                                           current_local_slice_ptr, d_temp_vec_out, d_targetIndices_gs, m, base_idx_non_targets);
                        if (hipGetLastError() != hipSuccess) { status = checkHipError(hipGetLastError(), "scatter_elements_kernel_v2 (Local)"); break; }
                    }
                } 
                
                hipError_t sync_err_gs = hipStreamSynchronize(h->streams[rank]);
                if (sync_err_gs != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = checkHipError(sync_err_gs, "rocsvApplyMatrix sync for GS (Local)");

                hipFree(d_targetIndices_gs);
                hipFree(d_temp_vec_in);
                hipFree(d_temp_vec_out);
                if (status != ROCQ_STATUS_SUCCESS) break; 
            } else { 
                 if (numTargetQubits == 0 && matrixDim == 1) {
                    status = ROCQ_STATUS_NOT_IMPLEMENTED;
                 } else {
                    status = ROCQ_STATUS_INVALID_VALUE;
                 }
                 break;
            }
            if (numTargetQubits < 3 && numTargetQubits > 0) {
                 hip_err = hipStreamSynchronize(h->streams[rank]);
                 if (hip_err != hipSuccess && status == ROCQ_STATUS_SUCCESS) { status = checkHipError(hip_err, "rocsvApplyMatrix hipStreamSynchronize (Local)"); break; }
            }
        } 
    } else {
        status = ROCQ_STATUS_NOT_IMPLEMENTED; 
    }
    
    return status;
}

rocqStatus_t rocsvMeasure(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned qubitToMeasure,
                          int* h_outcome,
                          double* h_probability) {
    if (!handle || !h_outcome || !h_probability) return ROCQ_STATUS_INVALID_VALUE;

    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    hipError_t hip_err;

    // Validate global qubit number (numQubits parameter here is the total number of qubits in the system)
    if (numQubits != h->globalNumQubits && h->globalNumQubits > 0) {
        // If d_state is provided (legacy single GPU call), numQubits should match that allocation.
        // If d_state is null (new multi-GPU call style), numQubits should match handle's globalNumQubits.
        if (d_state != nullptr && h->numGpus == 1 && (1ULL << numQubits) != h->localStateSizes[0]) {
             return ROCQ_STATUS_INVALID_VALUE; // Mismatch for single GPU with explicit d_state
        }
        if (d_state == nullptr && numQubits != h->globalNumQubits) { // Multi-GPU style call
             return ROCQ_STATUS_INVALID_VALUE;
        }
    }
    unsigned current_global_qubits = h->globalNumQubits;
    if (qubitToMeasure >= current_global_qubits && !(current_global_qubits == 0 && qubitToMeasure == 0)) {
         return ROCQ_STATUS_INVALID_VALUE;
    }

    // --- Single GPU Path ---
    if (h->numGpus == 1) {
        hipSetDevice(h->deviceIds[0]);
        hipStream_t current_stream = h->streams[0];
        rocComplex* current_d_state = h->d_local_state_slices[0];
        unsigned num_qubits_for_kernel = h->numLocalQubitsPerGpu;

        if (d_state != nullptr && d_state != current_d_state) return ROCQ_STATUS_INVALID_VALUE;
        if (numQubits != current_global_qubits) return ROCQ_STATUS_INVALID_VALUE; // numQubits param should be total for the system

        double h_prob0_sum_single = 0.0;
        double h_prob1_sum_single = 0.0; // For more accurate total prob calculation

        unsigned KERNEL_BLOCK_SIZE_S = 256;
        unsigned num_kernel_blocks_s = 0;
        if (h->localStateSizes[0] > 0) {
            num_kernel_blocks_s = (h->localStateSizes[0] + KERNEL_BLOCK_SIZE_S -1) / KERNEL_BLOCK_SIZE_S;
            if (num_kernel_blocks_s == 0) num_kernel_blocks_s = 1;
        }
        
        double* d_block_probs_s = nullptr;
        if (num_kernel_blocks_s > 0) {
            hipMalloc(&d_block_probs_s, num_kernel_blocks_s * 2 * sizeof(double));
            hipMemsetAsync(d_block_probs_s, 0, num_kernel_blocks_s * 2 * sizeof(double), current_stream);
            size_t shared_mem_size = KERNEL_BLOCK_SIZE_S * 2 * sizeof(double);
            hipLaunchKernelGGL(calculate_local_slice_probabilities_kernel,
                               dim3(num_kernel_blocks_s), dim3(KERNEL_BLOCK_SIZE_S), shared_mem_size, current_stream,
                               current_d_state, h->localStateSizes[0],
                               num_qubits_for_kernel, qubitToMeasure,
                               d_block_probs_s);
            hip_err = hipGetLastError();
            if (hip_err != hipSuccess) { if(d_block_probs_s) hipFree(d_block_probs_s); return checkHipError(hip_err, "rocsvMeasure single-GPU calc_probs_kernel"); }

            std::vector<double> h_block_probs_s_vec(num_kernel_blocks_s * 2);
            hipMemcpy(h_block_probs_s_vec.data(), d_block_probs_s, num_kernel_blocks_s * 2 * sizeof(double), hipMemcpyDeviceToHost);
            hipFree(d_block_probs_s);
            for(unsigned j=0; j < num_kernel_blocks_s; ++j) {
                h_prob0_sum_single += h_block_probs_s_vec[j * 2 + 0];
                h_prob1_sum_single += h_block_probs_s_vec[j * 2 + 1];
            }
        } else if (num_qubits_for_kernel == 0 && qubitToMeasure == 0 && h->localStateSizes[0] == 1) {
            rocComplex h_amp;
            hipMemcpy(&h_amp, current_d_state, sizeof(rocComplex), hipMemcpyDeviceToHost);
            h_prob0_sum_single = (double)h_amp.x * h_amp.x + (double)h_amp.y * h_amp.y;
            h_prob1_sum_single = 0.0; // Only |0> state for 0-qubit system
        }

        double prob0 = h_prob0_sum_single;
        double prob1 = h_prob1_sum_single;
        double total_prob_check_s = prob0 + prob1;

        if (fabs(total_prob_check_s) < 1e-12) { prob0 = 0.5; prob1 = 0.5;}
        else if (fabs(total_prob_check_s - 1.0) > 1e-9) { prob0 /= total_prob_check_s; prob1 = 1.0 - prob0;}
        if (prob0 < 0.0) prob0 = 0.0; if (prob0 > 1.0) prob0 = 1.0;
        prob1 = 1.0 - prob0;


        static bool seeded_single = false; if (!seeded_single) { srand((unsigned int)time(NULL)+1); seeded_single = true; }
        double rand_val = (double)rand() / RAND_MAX;

        if (rand_val < prob0) { *h_outcome = 0; *h_probability = prob0; }
        else { *h_outcome = 1; *h_probability = prob1; }

        unsigned threads_per_block_measure = 256;
        size_t total_states_measure = h->localStateSizes[0];
        unsigned num_blocks_measure = (total_states_measure + threads_per_block_measure - 1) / threads_per_block_measure;
        if (num_blocks_measure == 0 && total_states_measure > 0) num_blocks_measure = 1;
        
        if (num_blocks_measure > 0 && total_states_measure > 0) {
            hipLaunchKernelGGL(collapse_state_kernel, dim3(num_blocks_measure), dim3(threads_per_block_measure), 0, current_stream,
                               current_d_state, num_qubits_for_kernel, qubitToMeasure, *h_outcome);
            if (hipGetLastError() != hipSuccess) return ROCQ_STATUS_HIP_ERROR;
        }

        double h_sum_sq_mag_single = 0.0;
        unsigned num_kernel_blocks_sum_sq = (h->localStateSizes[0] + KERNEL_BLOCK_SIZE_S -1) / KERNEL_BLOCK_SIZE_S;
        if (num_kernel_blocks_sum_sq == 0 && h->localStateSizes[0] > 0) num_kernel_blocks_sum_sq = 1;

        double* d_block_sum_sq_s = nullptr;
        if (num_kernel_blocks_sum_sq > 0) {
            hipMalloc(&d_block_sum_sq_s, num_kernel_blocks_sum_sq * sizeof(double));
            hipMemsetAsync(d_block_sum_sq_s, 0, num_kernel_blocks_sum_sq * sizeof(double), current_stream);
            size_t shared_mem_size = KERNEL_BLOCK_SIZE_S * sizeof(double);
            hipLaunchKernelGGL(calculate_local_slice_sum_sq_mag_kernel,
                           dim3(num_kernel_blocks_sum_sq), dim3(KERNEL_BLOCK_SIZE_S), shared_mem_size, current_stream,
                           current_d_state, h->localStateSizes[0], d_block_sum_sq_s);
            if (hipGetLastError() != hipSuccess) { if(d_block_sum_sq_s) hipFree(d_block_sum_sq_s); return ROCQ_STATUS_HIP_ERROR; }
            
            std::vector<double> h_block_sum_sq_vec_s(num_kernel_blocks_sum_sq);
            hipMemcpy(h_block_sum_sq_vec_s.data(), d_block_sum_sq_s, num_kernel_blocks_sum_sq * sizeof(double), hipMemcpyDeviceToHost);
            hipFree(d_block_sum_sq_s);
            for(double val : h_block_sum_sq_vec_s) h_sum_sq_mag_single += val;
        } else if (h->localStateSizes[0] == 1) {
             rocComplex h_amp_collapsed;
             hipMemcpy(&h_amp_collapsed, current_d_state, sizeof(rocComplex), hipMemcpyDeviceToHost);
             h_sum_sq_mag_single = (double)h_amp_collapsed.x * h_amp_collapsed.x + (double)h_amp_collapsed.y * h_amp_collapsed.y;
        }

        if (fabs(h_sum_sq_mag_single) > 1e-12) {
            double norm_factor = 1.0 / sqrt(fabs(h_sum_sq_mag_single));
            if (num_blocks_measure > 0 && total_states_measure > 0) {
                hipLaunchKernelGGL(renormalize_state_kernel, dim3(num_blocks_measure), dim3(threads_per_block_measure), 0, current_stream,
                                   current_d_state, num_qubits_for_kernel, norm_factor);
                if (hipGetLastError() != hipSuccess) return ROCQ_STATUS_HIP_ERROR;
            }
        }
        hipStreamSynchronize(current_stream);
        return ROCQ_STATUS_SUCCESS;
    }

    // --- Multi-GPU Path ---
    if (qubitToMeasure >= h->numLocalQubitsPerGpu && h->numGpus > 1) {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
    unsigned local_target_qubit = qubitToMeasure;

    double global_prob0 = 0.0;
    double global_prob1 = 0.0;

    unsigned KERNEL_BLOCK_SIZE = 256;

    std::vector<double*> d_block_partial_probs_all_gpus(h->numGpus, nullptr);
    std::vector<std::vector<double>> h_block_partial_probs_all_gpus_host(h->numGpus);


    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;

        unsigned num_kernel_blocks = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
        if (num_kernel_blocks == 0 && h->localStateSizes[i] > 0) num_kernel_blocks = 1;
        if (num_kernel_blocks == 0) continue;

        hipError_t err_malloc = hipMalloc(&d_block_partial_probs_all_gpus[i], num_kernel_blocks * 2 * sizeof(double));
        if(err_malloc != hipSuccess) { status = checkHipError(err_malloc, "MGPU Measure Malloc Probs"); goto mgpu_measure_cleanup_probs; }
        hipMemsetAsync(d_block_partial_probs_all_gpus[i], 0, num_kernel_blocks * 2 * sizeof(double), h->streams[i]);

        size_t shared_mem_size = KERNEL_BLOCK_SIZE * 2 * sizeof(double);
        hipLaunchKernelGGL(calculate_local_slice_probabilities_kernel,
                           dim3(num_kernel_blocks), dim3(KERNEL_BLOCK_SIZE), shared_mem_size, h->streams[i],
                           h->d_local_state_slices[i], h->localStateSizes[i],
                           h->numLocalQubitsPerGpu, local_target_qubit,
                           d_block_partial_probs_all_gpus[i]);
        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) { status = checkHipError(hip_err, "launch calculate_local_slice_probabilities_kernel"); goto mgpu_measure_cleanup_probs; }
    }

    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0 || d_block_partial_probs_all_gpus[i] == nullptr) continue;
        hipStreamSynchronize(h->streams[i]);

        unsigned num_kernel_blocks = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
        if (num_kernel_blocks == 0 && h->localStateSizes[i] > 0) num_kernel_blocks = 1;
        if (num_kernel_blocks == 0) continue;

        h_block_partial_probs_all_gpus_host[i].resize(num_kernel_blocks * 2);
        hipMemcpy(h_block_partial_probs_all_gpus_host[i].data(), d_block_partial_probs_all_gpus[i], num_kernel_blocks * 2 * sizeof(double), hipMemcpyDeviceToHost);

        for(unsigned j=0; j < num_kernel_blocks; ++j) {
            global_prob0 += h_block_partial_probs_all_gpus_host[i][j * 2 + 0];
            global_prob1 += h_block_partial_probs_all_gpus_host[i][j * 2 + 1];
        }
    }

mgpu_measure_cleanup_probs:
    for(int i=0; i<h->numGpus; ++i) if(d_block_partial_probs_all_gpus[i]) { hipSetDevice(h->deviceIds[i]); hipFree(d_block_partial_probs_all_gpus[i]); d_block_partial_probs_all_gpus[i]=nullptr;}
    if(status != ROCQ_STATUS_SUCCESS) return status;

    double total_prob_check = global_prob0 + global_prob1;
    if (fabs(total_prob_check) < 1e-12) {
        global_prob0 = 0.5;
        global_prob1 = 0.5;
    } else if (fabs(total_prob_check - 1.0) > 1e-9) {
        global_prob0 /= total_prob_check;
        global_prob1 = 1.0 - global_prob0;
    } else {
         global_prob1 = 1.0 - global_prob0;
    }
    if (global_prob0 < 0.0) global_prob0 = 0.0; if (global_prob0 > 1.0) global_prob0 = 1.0;
    global_prob1 = 1.0 - global_prob0;


    static bool seeded_multi = false; if (!seeded_multi) { srand((unsigned int)time(NULL)+2); seeded_multi = true; }
    double rand_val_multi = (double)rand() / RAND_MAX;

    if (rand_val_multi < global_prob0) {
        *h_outcome = 0;
        *h_probability = global_prob0;
    } else {
        *h_outcome = 1;
        *h_probability = global_prob1;
    }

    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;

        unsigned num_blocks_collapse = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
        if (num_blocks_collapse == 0 && h->localStateSizes[i] > 0) num_blocks_collapse = 1;

        if (num_blocks_collapse > 0) {
            hipLaunchKernelGGL(collapse_state_kernel, dim3(num_blocks_collapse), dim3(KERNEL_BLOCK_SIZE), 0, h->streams[i],
                               h->d_local_state_slices[i], h->numLocalQubitsPerGpu, local_target_qubit, *h_outcome);
            if(hipGetLastError() != hipSuccess) {status = ROCQ_STATUS_HIP_ERROR; goto mgpu_measure_cleanup_renorm;}
        }
    }
    for (int i = 0; i < h->numGpus; ++i) { // Sync collapse
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;
        if(hipStreamSynchronize(h->streams[i]) != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = ROCQ_STATUS_HIP_ERROR;
    }
    if(status != ROCQ_STATUS_SUCCESS) goto mgpu_measure_cleanup_renorm;


    double global_sum_sq_mag_collapsed = 0.0;
    std::vector<double*> d_block_sum_sq_mag_all_gpus(h->numGpus, nullptr);
    std::vector<std::vector<double>> h_block_sum_sq_mag_all_gpus_host(h->numGpus);

    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;

        unsigned num_kernel_blocks = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
        if (num_kernel_blocks == 0 && h->localStateSizes[i] > 0) num_kernel_blocks = 1;
        if (num_kernel_blocks == 0) continue;

        hipMalloc(&d_block_sum_sq_mag_all_gpus[i], num_kernel_blocks * sizeof(double));
        hipMemsetAsync(d_block_sum_sq_mag_all_gpus[i], 0, num_kernel_blocks * sizeof(double), h->streams[i]);

        size_t shared_mem_size = KERNEL_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(calculate_local_slice_sum_sq_mag_kernel,
                           dim3(num_kernel_blocks), dim3(KERNEL_BLOCK_SIZE), shared_mem_size, h->streams[i],
                           h->d_local_state_slices[i], h->localStateSizes[i],
                           d_block_sum_sq_mag_all_gpus[i]);
        if(hipGetLastError() != hipSuccess) {status = ROCQ_STATUS_HIP_ERROR; goto mgpu_measure_cleanup_renorm;}
    }
    
    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0 || d_block_sum_sq_mag_all_gpus[i] == nullptr) continue;
        hipStreamSynchronize(h->streams[i]);

        unsigned num_kernel_blocks = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
        if (num_kernel_blocks == 0 && h->localStateSizes[i] > 0) num_kernel_blocks = 1;
        if (num_kernel_blocks == 0) continue;

        h_block_sum_sq_mag_all_gpus_host[i].resize(num_kernel_blocks);
        hipMemcpy(h_block_sum_sq_mag_all_gpus_host[i].data(), d_block_sum_sq_mag_all_gpus[i], num_kernel_blocks * sizeof(double), hipMemcpyDeviceToHost);
        for(unsigned j=0; j < num_kernel_blocks; ++j) {
            global_sum_sq_mag_collapsed += h_block_sum_sq_mag_all_gpus_host[i][j];
        }
    }

mgpu_measure_cleanup_renorm:
    for(int i=0; i<h->numGpus; ++i) if(d_block_sum_sq_mag_all_gpus[i]) {hipSetDevice(h->deviceIds[i]); hipFree(d_block_sum_sq_mag_all_gpus[i]); d_block_sum_sq_mag_all_gpus[i] = nullptr;}
    if(status != ROCQ_STATUS_SUCCESS) return status;

    if (fabs(global_sum_sq_mag_collapsed) > 1e-12) {
        double norm_factor = 1.0 / sqrt(fabs(global_sum_sq_mag_collapsed));

        for (int i = 0; i < h->numGpus; ++i) {
            hipSetDevice(h->deviceIds[i]);
            if (h->localStateSizes[i] == 0) continue;

            unsigned num_blocks_renorm = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
            if (num_blocks_renorm == 0 && h->localStateSizes[i] > 0) num_blocks_renorm = 1;

            if (num_blocks_renorm > 0) {
                hipLaunchKernelGGL(renormalize_state_kernel, dim3(num_blocks_renorm), dim3(KERNEL_BLOCK_SIZE), 0, h->streams[i],
                                   h->d_local_state_slices[i], h->numLocalQubitsPerGpu, norm_factor);
                 if(hipGetLastError() != hipSuccess) {status = ROCQ_STATUS_HIP_ERROR; goto mgpu_measure_final_sync;}
            }
        }
    } else if (*h_probability > 1e-9 && fabs(global_sum_sq_mag_collapsed) < 1e-12) {
        // This is a potential issue if a probable outcome leads to a zero norm state.
    }

mgpu_measure_final_sync:
    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        hipError_t err_sync = hipStreamSynchronize(h->streams[i]);
        if (err_sync != hipSuccess && status == ROCQ_STATUS_SUCCESS) {
            status = checkHipError(err_sync, "rocsvMeasure multi-GPU final sync");
        }
    }
    return status;
}

// API Functions for Specific Gates
>>>>>>> REPLACE
