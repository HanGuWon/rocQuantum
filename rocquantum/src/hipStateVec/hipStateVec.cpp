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
// Assuming RCCL uses rcclResult_t and rcclGetErrorString, similar to NCCL.
// If RCCL's API differs, these types/functions need to be adjusted.
rocqStatus_t checkRcclError(rcclResult_t err, const char* operation = "") {
    if (err != rcclSuccess) { // Assuming rcclSuccess is the success code
        // fprintf(stderr, "RCCL Error during %s: %s\n", operation, rcclGetErrorString(err)); // Assuming rcclGetErrorString
        return ROCQ_STATUS_RCCL_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

// Helper function to determine if all target qubits are "local" to each slice
// based on the current distribution scheme.
// "Local" means the qubit index falls within the `numLocalQubitsPerGpu` range.
static bool are_qubits_local(rocsvInternalHandle* h, const unsigned* qubitIndices, unsigned numTargetQubits) {
    if (!h || h->numGpus == 0) { // Should not happen if handle is valid
        return false; 
    }
    // If only one GPU, all qubits are effectively local to that GPU's perspective of its slice.
    if (h->numGpus == 1) {
        return true;
    }
    // For multi-GPU, a qubit is local if its global index is less than numLocalQubitsPerGpu.
    // This assumes qubitIndices are global indices.
    for (unsigned i = 0; i < numTargetQubits; ++i) {
        if (qubitIndices[i] >= h->numLocalQubitsPerGpu) {
            return false; // This qubit is a "slice-determining" bit, not local to all slices.
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
    internal_handle->comm = nullptr; 
    internal_handle->numGpus = 0;       // Will be updated based on device count or specific setup
    internal_handle->localRank = -1;    // Will be set to deviceId or rank from RCCL init
    internal_handle->localStateSize = 0;
    internal_handle->globalNumQubits = 0;
    internal_handle->numLocalQubits = 0;
    internal_handle->numGlobalSliceQubits = 0;

    hipError_t hip_err;
    rocblas_status blas_err;
    rcclResult_t rccl_err; 
    hipError_t hip_err;
    rocblas_status blas_err;

    // Determine number of GPUs
    int device_count = 0;
    hip_err = hipGetDeviceCount(&device_count);
    if (hip_err != hipSuccess) {
        delete internal_handle; // Allocation failed earlier
        return checkHipError(hip_err, "rocsvCreate hipGetDeviceCount");
    }
    if (device_count <= 0) {
        delete internal_handle;
        return ROCQ_STATUS_FAILURE; // No GPUs available
    }
    internal_handle->numGpus = device_count;

    // Resize vectors based on numGpus
    try {
        internal_handle->deviceIds.resize(internal_handle->numGpus);
        internal_handle->streams.resize(internal_handle->numGpus);
        internal_handle->blasHandles.resize(internal_handle->numGpus);
        internal_handle->comms.resize(internal_handle->numGpus);
        internal_handle->d_local_state_slices.resize(internal_handle->numGpus, nullptr); // Initialize with nullptr
        internal_handle->localStateSizes.resize(internal_handle->numGpus, 0);
        internal_handle->d_swap_buffers.resize(internal_handle->numGpus, nullptr); // Initialize with nullptr
    } catch (const std::bad_alloc& e) {
        delete internal_handle;
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    // Initialize global config (will be properly set by allocate function)
    internal_handle->globalNumQubits = 0;
    internal_handle->numLocalQubitsPerGpu = 0;
    internal_handle->numGlobalSliceQubits = 0;

    // Get RCCL Unique ID - must be generated once by one process/rank.
    // For single-process multi-GPU, we can generate it here.
    // For multi-process, this ID needs to be generated by rank 0 and broadcast.
    rcclUniqueId uniqueId;
    if (internal_handle->numGpus > 0) { // Check if numGpus could be 0 from device_count
        // Assuming this rocsvCreate is the central point for setup in a single process context
        // For MPI, rank 0 would do this and broadcast 'uniqueId'
        rccl_err = rcclGetUniqueId(&uniqueId);
        if (rccl_err != rcclSuccess) {
            delete internal_handle;
            return checkRcclError(rccl_err, "rocsvCreate rcclGetUniqueId");
        }
    }


    // Initialize per-GPU resources
    for (int i = 0; i < internal_handle->numGpus; ++i) {
        internal_handle->deviceIds[i] = i; // Assuming contiguous device IDs 0, 1, ..., n-1
                                           // A more robust way would be to get a list of available device IDs.
        
        hip_err = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err != hipSuccess) {
            // Cleanup already initialized resources before returning
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
             for (int j = 0; j < i; ++j) { /* ... cleanup ... */ } // Simplified cleanup
             if (internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]); // current one
             delete internal_handle; return checkHipError(hip_err, "rocsvCreate hipStreamCreate");
        }

        blas_err = rocblas_create_handle(&internal_handle->blasHandles[i]);
        if (blas_err != rocblas_status_success) {
            for (int j = 0; j < i; ++j) { /* ... cleanup ... */ }  // Simplified cleanup
            if(internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]);
            if(internal_handle->blasHandles[i]) rocblas_destroy_handle(internal_handle->blasHandles[i]);
            delete internal_handle; return checkRocblasError(blas_err, "rocsvCreate rocblas_create_handle");
        }

        blas_err = rocblas_set_stream(internal_handle->blasHandles[i], internal_handle->streams[i]);
        if (blas_err != rocblas_status_success) {
            for (int j = 0; j < i; ++j) { /* ... cleanup ... */ } // Simplified cleanup
            if(internal_handle->blasHandles[i]) rocblas_destroy_handle(internal_handle->blasHandles[i]);
            if(internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]);
            delete internal_handle; return checkRocblasError(blas_err, "rocsvCreate rocblas_set_stream");
        }
        
        // Initialize RCCL communicator for each rank
        rccl_err = rcclCommInitRank(&internal_handle->comms[i], internal_handle->numGpus, uniqueId, i);
        if (rccl_err != rcclSuccess) {
            for (int j = 0; j < i; ++j) { /* ... cleanup ... */ } // Simplified cleanup
            if(internal_handle->comms[i]) rcclCommDestroy(internal_handle->comms[i]); // current one
            if(internal_handle->blasHandles[i]) rocblas_destroy_handle(internal_handle->blasHandles[i]);
            if(internal_handle->streams[i]) hipStreamDestroy(internal_handle->streams[i]);
            delete internal_handle; return checkRcclError(rccl_err, "rocsvCreate rcclCommInitRank");
        }
    }
    
    // Restore original device context if necessary (good practice)
    // int originalDevice;
    // hipGetDevice(&originalDevice);
    // hipSetDevice(originalDevice); // Though for this handle, operations will target specific devices.

    *handle = internal_handle;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvDestroy(rocsvHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    rocqStatus_t first_error_status = ROCQ_STATUS_SUCCESS;
    // No specific rccl_err for this outer scope, errors are handled per call.

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_destroy = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_destroy != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
            first_error_status = checkHipError(hip_err_destroy, "rocsvDestroy hipSetDevice");
            // Continue to attempt cleanup of other resources
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
        if (internal_handle->d_swap_buffers[i]) { // Free swap buffer
            hipError_t free_err_swap = hipFree(internal_handle->d_swap_buffers[i]);
            if (free_err_swap != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) {
                first_error_status = checkHipError(free_err_swap, "rocsvDestroy hipFree d_swap_buffers");
            }
            internal_handle->d_swap_buffers[i] = nullptr;
        }
    }

    // Clear vectors
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
    // This function is now intended for single-GPU context or as a wrapper.
    // For multi-GPU, rocsvAllocateDistributedState should be used.
    // For now, let's make it allocate on device 0 if called in a multi-GPU handle.
    if (!handle || !d_state || numQubits == 0 || numQubits > 60) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    if (internal_handle->numGpus == 0 || internal_handle->d_local_state_slices.empty()) {
        return ROCQ_STATUS_FAILURE; // Handle not properly initialized for multi-GPU
    }

    // Set context to the first GPU for this legacy function
    hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[0]);
    if (hip_err_set != hipSuccess) return checkHipError(hip_err_set, "rocsvAllocateState hipSetDevice");

    size_t num_elements = 1ULL << numQubits;
    size_t size_bytes = num_elements * sizeof(rocComplex);
    
    // If d_local_state_slices[0] is already allocated, free it first (or error out)
    if (internal_handle->d_local_state_slices[0] != nullptr) {
        // Allowing re-allocation by freeing first.
        // Consider if this should be an error instead.
        hipFree(internal_handle->d_local_state_slices[0]);
        internal_handle->d_local_state_slices[0] = nullptr;
        internal_handle->localStateSizes[0] = 0;
    }

    hipError_t err = hipMalloc(&internal_handle->d_local_state_slices[0], size_bytes);
    if (err != hipSuccess) {
        *d_state = nullptr; // d_state is a bit ambiguous here, but for compatibility...
        internal_handle->d_local_state_slices[0] = nullptr;
        return checkHipError(err, "rocsvAllocateState hipMalloc");
    }
    internal_handle->localStateSizes[0] = num_elements;
    *d_state = internal_handle->d_local_state_slices[0]; // Return pointer to the first slice

    // Update global qubit counts for this single GPU case
    internal_handle->globalNumQubits = numQubits;
    internal_handle->numGlobalSliceQubits = 0; // No slicing
    internal_handle->numLocalQubitsPerGpu = numQubits; 

    return ROCQ_STATUS_SUCCESS;
}


rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle, unsigned totalNumQubits) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);

    if (internal_handle->numGpus == 0) return ROCQ_STATUS_FAILURE; // Not initialized

    // Check if numGpus is a power of 2
    if ((internal_handle->numGpus > 0) && ((internal_handle->numGpus & (internal_handle->numGpus - 1)) != 0)) {
        // fprintf(stderr, "Error: numGpus (%d) must be a power of 2 for distributed state allocation.\n", internal_handle->numGpus);
        return ROCQ_STATUS_INVALID_VALUE; // Or implement non-power-of-2 logic
    }
    if (totalNumQubits == 0 || totalNumQubits > 60) { // Practical limit
        return ROCQ_STATUS_INVALID_VALUE;
    }
    
    unsigned num_global_slice_qubits = 0;
    if (internal_handle->numGpus > 1) { // log2(1) is 0, but can be tricky with float precision
        num_global_slice_qubits = static_cast<unsigned>(std::log2(internal_handle->numGpus));
    } else if (internal_handle->numGpus == 1) {
        num_global_slice_qubits = 0;
    }
    // Sanity check for log2 calculation, e.g. if numGpus was not power of 2 and we didn't error out
    if ((1U << num_global_slice_qubits) != static_cast<unsigned>(internal_handle->numGpus) && internal_handle->numGpus > 1) {
         // This implies numGpus was not a power of 2, should have been caught earlier
        return ROCQ_STATUS_INVALID_VALUE;
    }


    if (totalNumQubits < num_global_slice_qubits) {
        // Not enough qubits to distribute across all GPUs
        return ROCQ_STATUS_INVALID_VALUE;
    }

    internal_handle->globalNumQubits = totalNumQubits;
    internal_handle->numGlobalSliceQubits = num_global_slice_qubits;
    internal_handle->numLocalQubitsPerGpu = totalNumQubits - num_global_slice_qubits;
    
    size_t sliceNumElements = 1ULL << internal_handle->numLocalQubitsPerGpu;
    size_t sliceSizeBytes = sliceNumElements * sizeof(rocComplex);

    rocqStatus_t status = ROCQ_STATUS_SUCCESS;

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_set != hipSuccess) {
            status = checkHipError(hip_err_set, "rocsvAllocateDistributedState hipSetDevice");
            // Attempt to clean up already allocated slices if error occurs mid-loop
            for (int j = 0; j < i; ++j) {
                if (internal_handle->d_local_state_slices[j]) {
                    hipSetDevice(internal_handle->deviceIds[j]); // Ignore error for cleanup
                    hipFree(internal_handle->d_local_state_slices[j]);
                    internal_handle->d_local_state_slices[j] = nullptr;
                }
            }
            return status; 
        }

        // Free if already allocated (e.g. re-allocating)
        if (internal_handle->d_local_state_slices[i] != nullptr) {
            hipFree(internal_handle->d_local_state_slices[i]);
            internal_handle->d_local_state_slices[i] = nullptr;
        }
        if (internal_handle->d_swap_buffers[i] != nullptr) { // Also free corresponding swap buffer if re-allocating state
            hipFree(internal_handle->d_swap_buffers[i]);
            internal_handle->d_swap_buffers[i] = nullptr;
        }

        hipError_t hip_err_alloc = hipMalloc(&internal_handle->d_local_state_slices[i], sliceSizeBytes);
        if (hip_err_alloc != hipSuccess) {
            internal_handle->d_local_state_slices[i] = nullptr; 
            status = checkHipError(hip_err_alloc, "rocsvAllocateDistributedState hipMalloc slice");
            // Cleanup logic for previously allocated slices and swap buffers would be complex here.
            // For simplicity, returning early. A robust implementation would clean up everything allocated in this call.
            // Free already allocated slices and swap buffers for previous ranks
            for (int k = 0; k < i; ++k) {
                if(internal_handle->d_local_state_slices[k]) hipFree(internal_handle->d_local_state_slices[k]);
                if(internal_handle->d_swap_buffers[k]) hipFree(internal_handle->d_swap_buffers[k]);
                internal_handle->d_local_state_slices[k] = nullptr;
                internal_handle->d_swap_buffers[k] = nullptr;
            }
            return status;
        }
        internal_handle->localStateSizes[i] = sliceNumElements;

        // Allocate swap buffer for this GPU
        hipError_t hip_err_alloc_swap = hipMalloc(&internal_handle->d_swap_buffers[i], sliceSizeBytes);
        if (hip_err_alloc_swap != hipSuccess) {
            internal_handle->d_swap_buffers[i] = nullptr;
            hipFree(internal_handle->d_local_state_slices[i]); // Free the just-allocated slice for this GPU
            internal_handle->d_local_state_slices[i] = nullptr;
            status = checkHipError(hip_err_alloc_swap, "rocsvAllocateDistributedState hipMalloc swap_buffer");
            // Free already allocated slices and swap buffers for previous ranks
            for (int k = 0; k < i; ++k) {
                if(internal_handle->d_local_state_slices[k]) hipFree(internal_handle->d_local_state_slices[k]);
                if(internal_handle->d_swap_buffers[k]) hipFree(internal_handle->d_swap_buffers[k]);
                internal_handle->d_local_state_slices[k] = nullptr;
                internal_handle->d_swap_buffers[k] = nullptr;
            }
            return status;
        }
    }

    return status;
}

rocqStatus_t rocsvInitializeState(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits) {
    // This function is now intended for single-GPU context or as a wrapper.
    // For multi-GPU, rocsvInitializeDistributedState should be used.
    if (!handle || !d_state || numQubits > 60) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    if (internal_handle->numGpus == 0 || internal_handle->streams.empty() || 
        internal_handle->d_local_state_slices.empty() || internal_handle->d_local_state_slices[0] != d_state) {
        // Ensure d_state matches the one managed by the handle for single GPU case
        return ROCQ_STATUS_INVALID_VALUE; // Or handle not properly initialized / mismatched d_state
    }
    
    // Set context to the first GPU for this legacy function
    hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[0]);
    if (hip_err_set != hipSuccess) return checkHipError(hip_err_set, "rocsvInitializeState hipSetDevice");

    size_t num_elements = 1ULL << numQubits;
    if (internal_handle->localStateSizes[0] != num_elements) { // Validate size
        return ROCQ_STATUS_INVALID_VALUE;
    }

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

    if (internal_handle->numGpus == 0 || internal_handle->globalNumQubits == 0) {
        // Not allocated or initialized properly
        return ROCQ_STATUS_FAILURE;
    }
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_set != hipSuccess) {
            return checkHipError(hip_err_set, "rocsvInitializeDistributedState hipSetDevice");
        }
        if (internal_handle->d_local_state_slices[i] == nullptr || internal_handle->localStateSizes[i] == 0) {
            return ROCQ_STATUS_INVALID_VALUE; // Slice not allocated
        }
        hipError_t hip_err_memset = hipMemsetAsync(internal_handle->d_local_state_slices[i], 0, 
                                               internal_handle->localStateSizes[i] * sizeof(rocComplex), 
                                               internal_handle->streams[i]);
        if (hip_err_memset != hipSuccess) {
            // No easy rollback here, just return error. Consider more robust error handling for production.
            return checkHipError(hip_err_memset, "rocsvInitializeDistributedState hipMemsetAsync");
        }
    }

    // Set |0...0> state: the 0-th amplitude (overall global index 0) is on rank 0's slice at local index 0.
    if (internal_handle->numGpus > 0) { // Check to prevent access to deviceIds[0] if numGpus is 0
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
        } else {
            // This case (numGpus > 0 but slice[0] is null or size 0) should ideally be caught earlier
            // during allocation checks.
            return ROCQ_STATUS_FAILURE; 
        }
    }
    
    // Synchronize all streams
    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
         if (hip_err_set != hipSuccess) { // Log or handle minorly, primary goal is sync
            status = checkHipError(hip_err_set, "rocsvInitializeDistributedState hipSetDevice for sync");
            // Potentially continue to sync other streams
        }
        hipError_t hip_err_sync = hipStreamSynchronize(internal_handle->streams[i]);
        if (hip_err_sync != hipSuccess && status == ROCQ_STATUS_SUCCESS) { // Store first error
            status = checkHipError(hip_err_sync, "rocsvInitializeDistributedState hipStreamSynchronize");
        }
    }
    return status;
}


// ApplyMatrix for multi-GPU.
// The `d_state` parameter is mostly ignored for multi-GPU as the state is in h->d_local_state_slices.
// `globalNumQubits_param` is the total number of qubits in the simulation.
rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle,
                              rocComplex* d_state_legacy,       // Legacy: for single GPU, points to d_local_state_slices[0]
                              unsigned globalNumQubits_param, 
                              const unsigned* qubitIndices,     // HOST pointer, global indices
                              unsigned numTargetQubits,    
                              const rocComplex* matrixDevice,   // Gate matrix on DEVICE memory (current active device)
                              unsigned matrixDim) {        
    if (!handle || !qubitIndices || !matrixDevice) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (h->numGpus == 0 || h->d_local_state_slices.empty()) { // Ensure handle is initialized
        return ROCQ_STATUS_FAILURE; 
    }
    // Use globalNumQubits from handle if allocated, otherwise from parameter.
    unsigned currentGlobalNumQubits = (h->globalNumQubits > 0) ? h->globalNumQubits : globalNumQubits_param;
    if (currentGlobalNumQubits == 0) return ROCQ_STATUS_INVALID_VALUE;


    if (numTargetQubits == 0) return ROCQ_STATUS_INVALID_VALUE;
    if (matrixDim != (1U << numTargetQubits)) return ROCQ_STATUS_INVALID_VALUE;

    for (unsigned i = 0; i < numTargetQubits; ++i) {
        if (qubitIndices[i] >= currentGlobalNumQubits) return ROCQ_STATUS_INVALID_VALUE;
        for (unsigned j = i + 1; j < numTargetQubits; ++j) {
            if (qubitIndices[i] == qubitIndices[j]) return ROCQ_STATUS_INVALID_VALUE; 
        }
    }

    hipError_t hip_err;
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    unsigned threads_per_block = 256; // Default, can be overridden by specific kernels

    if (are_qubits_local(h, qubitIndices, numTargetQubits)) {
        // --- LOCAL GATE APPLICATION ---
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);

        // Create a local copy of qubitIndices, as they are already local indices.
        // (This is because are_qubits_local ensures qubitIndices[i] < h->numLocalQubitsPerGpu)
        std::vector<unsigned> local_qubit_indices_vec(qubitIndices, qubitIndices + numTargetQubits);

        for (int rank = 0; rank < h->numGpus; ++rank) {
            hip_err = hipSetDevice(h->deviceIds[rank]);
            if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix hipSetDevice (Local)"); break; }

            rocComplex* current_local_slice_ptr = h->d_local_state_slices[rank];
            if (!current_local_slice_ptr || h->localStateSizes[rank] != local_slice_num_elements) {
                status = ROCQ_STATUS_INVALID_VALUE; // Slice not allocated or wrong size
                break; 
            }
            
            unsigned* d_target_indices_gpu = nullptr; // For multi-qubit generic kernels

            if (numTargetQubits == 1) {
                unsigned targetQubitLocal = local_qubit_indices_vec[0];
                size_t num_thread_groups = local_slice_num_elements / 2;
                unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
                if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1; else if (num_thread_groups == 0) num_blocks = 0;
                
                if (num_blocks > 0) { // Ensure kernel is launched only if there's work
                    hipLaunchKernelGGL(apply_single_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                       current_local_slice_ptr, local_num_qubits_for_kernel, targetQubitLocal, matrixDevice);
                    hip_err = hipGetLastError();
                    if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix (1Q Local Kernel)"); break;}
                }
            } else if (numTargetQubits == 2) {
                unsigned q0_local = local_qubit_indices_vec[0];
                unsigned q1_local = local_qubit_indices_vec[1];
                if (q0_local > q1_local) std::swap(q0_local, q1_local); // Kernel might assume order

                size_t num_thread_groups = local_slice_num_elements / 4;
                unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
                if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1; else if (num_thread_groups == 0) num_blocks = 0;

                if (num_blocks > 0) {
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

                if (num_blocks > 0) {
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
                // Sync stream before freeing d_target_indices_gpu (even if kernel not launched, hipMemcpyAsync)
                hipError_t sync_err_kernel = hipStreamSynchronize(h->streams[rank]);
                if (sync_err_kernel != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = checkHipError(sync_err_kernel, "rocsvApplyMatrix sync for d_target_indices_gpu (Local)");
                hipFree(d_target_indices_gpu);
                if(status != ROCQ_STATUS_SUCCESS) break; // Exit rank loop on error

            } else if (numTargetQubits >= 5) { 
                const unsigned MAX_M_FOR_GATHER_SCATTER = 10; 
                if (numTargetQubits > MAX_M_FOR_GATHER_SCATTER) { status = ROCQ_STATUS_NOT_IMPLEMENTED; break; }

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
                    break; 
                }
                
                rocblas_float_complex alpha_gemv = {1.0f, 0.0f};
                rocblas_float_complex beta_gemv = {0.0f, 0.0f};
                unsigned num_non_target_qubits_local = local_num_qubits_for_kernel - m;
                size_t num_non_target_configs_local = 1ULL << num_non_target_qubits_local;
                
                unsigned gs_threads = 256; 
                if (matrixDim < gs_threads && matrixDim > 0) gs_threads = matrixDim; else if (matrixDim == 0) gs_threads = 1; 
                unsigned gs_blocks = (matrixDim + gs_threads - 1) / gs_threads;
                if (matrixDim == 0) gs_blocks = 0; else if (gs_blocks == 0 && matrixDim > 0) gs_blocks = 1;

                for (size_t j = 0; j < num_non_target_configs_local; ++j) {
                    if (status != ROCQ_STATUS_SUCCESS) break;
                    size_t base_idx_non_targets = 0; 
                    unsigned current_non_target_bit_pos = 0;
                    for (unsigned bit_idx = 0; bit_idx < local_num_qubits_for_kernel; ++bit_idx) {
                        bool is_target = false;
                        for(unsigned k=0; k<m; ++k) if(h_sorted_local_indices[k] == bit_idx) is_target = true;
                        if (!is_target) {
                            if (((j >> current_non_target_bit_pos) & 1)) base_idx_non_targets |= (1ULL << bit_idx);
                            current_non_target_bit_pos++;
                        }
                    }

                    if (gs_blocks > 0) {
                        hipLaunchKernelGGL(gather_elements_kernel_v2, dim3(gs_blocks), dim3(gs_threads), 0, h->streams[rank],
                                           d_temp_vec_in, current_local_slice_ptr, d_targetIndices_gs, m, base_idx_non_targets);
                        if (hipGetLastError() != hipSuccess) { status = checkHipError(hipGetLastError(), "gather_elements_kernel_v2 (Local)"); break; }
                    }
                    
                    rocblas_status blas_status = rocblas_cgemv(h->blasHandles[rank], rocblas_operation_none,
                                       matrixDim, matrixDim, &alpha_gemv,
                                       (const rocblas_float_complex*)matrixDevice, matrixDim,
                                       (const rocblas_float_complex*)d_temp_vec_in, 1, &beta_gemv,
                                       (rocblas_float_complex*)d_temp_vec_out, 1);
                    if (blas_status != rocblas_status_success) { status = checkRocblasError(blas_status, "rocblas_cgemv (m>=5 Local)"); break; }
                    
                    if (gs_blocks > 0) {
                        hipLaunchKernelGGL(scatter_elements_kernel_v2, dim3(gs_blocks), dim3(gs_threads), 0, h->streams[rank],
                                           current_local_slice_ptr, d_temp_vec_out, d_targetIndices_gs, m, base_idx_non_targets);
                        if (hipGetLastError() != hipSuccess) { status = checkHipError(hipGetLastError(), "scatter_elements_kernel_v2 (Local)"); break; }
                    }
                } 
                
                hipError_t sync_err_gs = hipStreamSynchronize(h->streams[rank]); // Sync before freeing
                if (sync_err_gs != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = checkHipError(sync_err_gs, "rocsvApplyMatrix sync for GS (Local)");

                hipFree(d_targetIndices_gs);
                hipFree(d_temp_vec_in);
                hipFree(d_temp_vec_out);
                if (status != ROCQ_STATUS_SUCCESS) break; 
            } else { 
                status = ROCQ_STATUS_INVALID_VALUE; break;
            }
            // Sync stream for this rank after its operation, if not already done by GS path
            if (numTargetQubits < 5) { // GS path already synced
                 hip_err = hipStreamSynchronize(h->streams[rank]);
                 if (hip_err != hipSuccess && status == ROCQ_STATUS_SUCCESS) { status = checkHipError(hip_err, "rocsvApplyMatrix hipStreamSynchronize (Local)"); break; }
            }
        } 
    } else {
        // --- GLOBAL GATE APPLICATION ---
        // std::cout << "Global gate: requires rocsvSwapIndexBits." << std::endl;
        // This part requires calling rocsvSwapIndexBits to make qubits local,
        // then applying the gate locally (potentially by recalling this function
        // with modified qubit indices or by duplicating the local logic),
        // and then swapping back. This is highly conceptual given rocsvSwapIndexBits is a stub.
        //
        // Example conceptual flow:
        // std::vector<unsigned> original_indices(qubitIndices, qubitIndices + numTargetQubits);
        // std::vector<std::pair<unsigned, unsigned>> swaps_done;
        // for (unsigned i = 0; i < numTargetQubits; ++i) {
        //    if (qubitIndices[i] >= h->numLocalQubitsPerGpu) { // If this target qubit is a slice bit
        //        unsigned slice_bit_global_idx = qubitIndices[i];
        //        unsigned local_swap_candidate_global_idx = i; // Naive: swap with one of the first few local bits
        //                                                     // A more robust strategy is needed to find an available local bit
        //                                                     // and ensure it's not another target qubit.
        //        // status = rocsvSwapIndexBits(handle, slice_bit_global_idx, local_swap_candidate_global_idx);
        //        // if (status != ROCQ_STATUS_SUCCESS) return status; // Or try to undo previous swaps
        //        // swaps_done.push_back({slice_bit_global_idx, local_swap_candidate_global_idx});
        //        // Update qubitIndices[i] to local_swap_candidate_global_idx for the local application
        //        // The actual data for slice_bit_global_idx is now at local_swap_candidate_global_idx's original data position
        //    }
        // }
        //
        // // After all necessary swaps, all qubitIndices for the gate are now effectively local.
        // // Recursively call rocsvApplyMatrix or duplicate local application logic here.
        // // status = rocsvApplyMatrix(handle, d_state_legacy, currentGlobalNumQubits, /* MODIFIED qubitIndices */, ...);
        //
        // // Swap back
        // // for (auto p = swaps_done.rbegin(); p != swaps_done.rend(); ++p) {
        // //    status = rocsvSwapIndexBits(handle, p->first, p->second); // Swap back
        // //    // if (status != ROCQ_STATUS_SUCCESS) { /* handle error */ }
        // // }
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

// API Functions for Specific Gates
// The d_state parameter is legacy; actual state is in handle->d_local_state_slices.
// numQubits parameter is the global number of qubits.
rocqStatus_t rocsvApplyX(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;

    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit; 

        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = local_slice_num_elements / 2; // Each thread group handles a pair of states
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups == 0 && local_slice_num_elements > 0) { // e.g. 0-qubit system (1 element), no pairs.
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0) num_blocks = 0; // No kernel needed
             else if (num_thread_groups == 0) num_blocks = 0; // Should not happen if local_slice_num_elements > 0 and local_num_qubits > 0
        } else if (num_thread_groups > 0 && num_blocks == 0) {
            num_blocks = 1;
        }
        
        if (local_slice_num_elements == 0 && current_global_qubits > 0) return ROCQ_STATUS_INVALID_VALUE;

        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyX hipSetDevice"); break; }
            
            if (h->d_local_state_slices[rank] == nullptr && local_slice_num_elements > 0) { status = ROCQ_STATUS_INVALID_VALUE; break;}

            if (num_blocks > 0) {
                 hipLaunchKernelGGL(apply_X_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, targetQubitLocal);
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyX apply_X_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyX hipStreamSynchronize"); break; }
        }
        return status;
    } else {
        // Conceptual: rocsvSwapIndexBits to make targetQubit local, apply, swap back.
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
}

rocqStatus_t rocsvApplyY(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;

    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_nq = h->numLocalQubitsPerGpu; size_t local_ne = 1ULL << local_nq;
        rocqStatus_t st = ROCQ_STATUS_SUCCESS; unsigned tpb=256; size_t ntg=local_ne/2; unsigned nb=(ntg+tpb-1)/tpb; if(ntg>0&&nb==0)nb=1;else if(ntg==0)nb=0;
        for(int r=0;r<h->numGpus;++r){hipSetDevice(h->deviceIds[r]);if(h->d_local_state_slices[r]==nullptr&&local_ne>0){st=ROCQ_STATUS_INVALID_VALUE;break;} if(nb>0)hipLaunchKernelGGL(apply_Y_kernel,dim3(nb),dim3(tpb),0,h->streams[r],h->d_local_state_slices[r],local_nq,targetQubit); if(hipGetLastError()!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;} if(hipStreamSynchronize(h->streams[r])!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;}} return st;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}
rocqStatus_t rocsvApplyZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE; rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle); if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned cgq = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits; if (targetQubit >= cgq) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned lnq=h->numLocalQubitsPerGpu; size_t lne=1ULL<<lnq; rocqStatus_t st=ROCQ_STATUS_SUCCESS; unsigned tpb=256; size_t ntg=lne/2; unsigned nb=(ntg+tpb-1)/tpb; if(ntg>0&&nb==0)nb=1;else if(ntg==0)nb=0;
        for(int r=0;r<h->numGpus;++r){hipSetDevice(h->deviceIds[r]);if(h->d_local_state_slices[r]==nullptr&&lne>0){st=ROCQ_STATUS_INVALID_VALUE;break;} if(nb>0)hipLaunchKernelGGL(apply_Z_kernel,dim3(nb),dim3(tpb),0,h->streams[r],h->d_local_state_slices[r],lnq,targetQubit); if(hipGetLastError()!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;} if(hipStreamSynchronize(h->streams[r])!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;}} return st;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}
rocqStatus_t rocsvApplyH(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE; rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle); if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned cgq = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits; if (targetQubit >= cgq) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned lnq=h->numLocalQubitsPerGpu; size_t lne=1ULL<<lnq; rocqStatus_t st=ROCQ_STATUS_SUCCESS; unsigned tpb=256; size_t ntg=lne/2; unsigned nb=(ntg+tpb-1)/tpb; if(ntg>0&&nb==0)nb=1;else if(ntg==0)nb=0;
        for(int r=0;r<h->numGpus;++r){hipSetDevice(h->deviceIds[r]);if(h->d_local_state_slices[r]==nullptr&&lne>0){st=ROCQ_STATUS_INVALID_VALUE;break;} if(nb>0)hipLaunchKernelGGL(apply_H_kernel,dim3(nb),dim3(tpb),0,h->streams[r],h->d_local_state_slices[r],lnq,targetQubit); if(hipGetLastError()!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;} if(hipStreamSynchronize(h->streams[r])!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;}} return st;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}
rocqStatus_t rocsvApplyS(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE; rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle); if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned cgq = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits; if (targetQubit >= cgq) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned lnq=h->numLocalQubitsPerGpu; size_t lne=1ULL<<lnq; rocqStatus_t st=ROCQ_STATUS_SUCCESS; unsigned tpb=256; size_t ntg=lne/2; unsigned nb=(ntg+tpb-1)/tpb; if(ntg>0&&nb==0)nb=1;else if(ntg==0)nb=0;
        for(int r=0;r<h->numGpus;++r){hipSetDevice(h->deviceIds[r]);if(h->d_local_state_slices[r]==nullptr&&lne>0){st=ROCQ_STATUS_INVALID_VALUE;break;} if(nb>0)hipLaunchKernelGGL(apply_S_kernel,dim3(nb),dim3(tpb),0,h->streams[r],h->d_local_state_slices[r],lnq,targetQubit); if(hipGetLastError()!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;} if(hipStreamSynchronize(h->streams[r])!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;}} return st;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}
rocqStatus_t rocsvApplyT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE; rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle); if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned cgq = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits; if (targetQubit >= cgq) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned lnq=h->numLocalQubitsPerGpu; size_t lne=1ULL<<lnq; rocqStatus_t st=ROCQ_STATUS_SUCCESS; unsigned tpb=256; size_t ntg=lne/2; unsigned nb=(ntg+tpb-1)/tpb; if(ntg>0&&nb==0)nb=1;else if(ntg==0)nb=0;
        for(int r=0;r<h->numGpus;++r){hipSetDevice(h->deviceIds[r]);if(h->d_local_state_slices[r]==nullptr&&lne>0){st=ROCQ_STATUS_INVALID_VALUE;break;} if(nb>0)hipLaunchKernelGGL(apply_T_kernel,dim3(nb),dim3(tpb),0,h->streams[r],h->d_local_state_slices[r],lnq,targetQubit); if(hipGetLastError()!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;} if(hipStreamSynchronize(h->streams[r])!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;}} return st;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyRx(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;

    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;

        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = local_slice_num_elements / 2;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1; else if (num_thread_groups == 0) num_blocks = 0;
        
        if (local_slice_num_elements == 0 && current_global_qubits > 0) return ROCQ_STATUS_INVALID_VALUE;

        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyRx hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && local_slice_num_elements > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}

            if (num_blocks > 0) {
                 hipLaunchKernelGGL(apply_Rx_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, targetQubitLocal, static_cast<float>(theta));
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyRx apply_Rx_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyRx hipStreamSynchronize"); break; }
        }
        return status;
    } else {
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
}

rocqStatus_t rocsvApplyRy(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE; rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle); if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned cgq = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits; if (targetQubit >= cgq) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned lnq=h->numLocalQubitsPerGpu; size_t lne=1ULL<<lnq; rocqStatus_t st=ROCQ_STATUS_SUCCESS; unsigned tpb=256; size_t ntg=lne/2; unsigned nb=(ntg+tpb-1)/tpb; if(ntg>0&&nb==0)nb=1;else if(ntg==0)nb=0;
        for(int r=0;r<h->numGpus;++r){hipSetDevice(h->deviceIds[r]);if(h->d_local_state_slices[r]==nullptr&&lne>0){st=ROCQ_STATUS_INVALID_VALUE;break;} if(nb>0)hipLaunchKernelGGL(apply_Ry_kernel,dim3(nb),dim3(tpb),0,h->streams[r],h->d_local_state_slices[r],lnq,targetQubit,static_cast<float>(theta)); if(hipGetLastError()!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;} if(hipStreamSynchronize(h->streams[r])!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;}} return st;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}
rocqStatus_t rocsvApplyRz(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE; rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle); if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned cgq = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits; if (targetQubit >= cgq) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned lnq=h->numLocalQubitsPerGpu; size_t lne=1ULL<<lnq; rocqStatus_t st=ROCQ_STATUS_SUCCESS; unsigned tpb=256; size_t ntg=lne/2; unsigned nb=(ntg+tpb-1)/tpb; if(ntg>0&&nb==0)nb=1;else if(ntg==0)nb=0;
        for(int r=0;r<h->numGpus;++r){hipSetDevice(h->deviceIds[r]);if(h->d_local_state_slices[r]==nullptr&&lne>0){st=ROCQ_STATUS_INVALID_VALUE;break;} if(nb>0)hipLaunchKernelGGL(apply_Rz_kernel,dim3(nb),dim3(tpb),0,h->streams[r],h->d_local_state_slices[r],lnq,targetQubit,static_cast<float>(theta)); if(hipGetLastError()!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;} if(hipStreamSynchronize(h->streams[r])!=hipSuccess){st=ROCQ_STATUS_HIP_ERROR;break;}} return st;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (controlQubit >= current_global_qubits || targetQubit >= current_global_qubits || controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
     if (current_global_qubits < 2) return ROCQ_STATUS_INVALID_VALUE;


    unsigned qubitIndices[2] = {controlQubit, targetQubit};
    if (are_qubits_local(h, qubitIndices, 2)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned controlQubitLocal = controlQubit; 
        unsigned targetQubitLocal = targetQubit;   

        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_kernel_threads = (local_num_qubits_for_kernel < 2) ? 0 : (local_slice_num_elements >> 2);
        unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
        if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1; if (num_kernel_threads == 0) num_blocks = 0;
        
        if (local_slice_num_elements == 0 && current_global_qubits > 0) return ROCQ_STATUS_INVALID_VALUE;


        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyCNOT hipSetDevice"); break; }

            if (h->d_local_state_slices[rank] == nullptr && local_slice_num_elements > 0) return ROCQ_STATUS_INVALID_VALUE;

            if (num_blocks > 0) {
                hipLaunchKernelGGL(apply_CNOT_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], 
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, controlQubitLocal, targetQubitLocal);
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyCNOT apply_CNOT_kernel"); break; }
            } else if (local_num_qubits_for_kernel < 2) {
                 // CNOT on <2 local qubits is a no-op for this slice or an error if control/target are out of bounds for 0/1 qubit system
                 // Assuming local_num_qubits_for_kernel is valid for the slice, this means no operation.
            }

            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyCNOT hipStreamSynchronize"); break; }
        }
        return status;
    } else {
        // std::cout << "rocsvApplyCNOT: Global gate - requires rocsvSwapIndexBits." << std::endl;
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
}

rocqStatus_t rocsvApplyCZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2) {
    // Placeholder - needs full refactoring like rocsvApplyCNOT
    return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplySWAP(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2) {
    // Placeholder - needs full refactoring like rocsvApplyCNOT
    return ROCQ_STATUS_NOT_IMPLEMENTED;
}

} // extern "C"

rocqStatus_t rocsvApplyFusedSingleQubitMatrix(rocsvHandle_t handle,
                                              unsigned targetQubit,
                                              const rocComplex* d_fusedMatrix) {
    if (!handle || !d_fusedMatrix) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (h->globalNumQubits == 0 && h->numGpus > 0) { // Check if state was allocated
        // If globalNumQubits is 0 but we have GPUs, it implies state not set up.
        return ROCQ_STATUS_INVALID_VALUE; 
    }
     if (h->globalNumQubits == 0 && h->numGpus == 0) { // Uninitialized handle
        return ROCQ_STATUS_FAILURE;
    }


    if (targetQubit >= h->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    bool is_local = are_qubits_local(h, &targetQubit, 1);

    if (is_local) {
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        for (int i = 0; i < h->numGpus; ++i) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[i]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyFusedSingleQubitMatrix hipSetDevice"); break; }


            unsigned localTargetQubit = targetQubit; 

            size_t total_slice_states = (h->numLocalQubitsPerGpu > 0 || h->numGpus == 1 || h->numLocalQubitsPerGpu == 0) ? (1ULL << h->numLocalQubitsPerGpu) : 0;
            if (total_slice_states == 0 && h->globalNumQubits > 0 && h->numGpus > 1) { status = ROCQ_STATUS_INVALID_VALUE; break; } // Should not happen if allocated
            if (h->d_local_state_slices[i] == nullptr && total_slice_states > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}


            unsigned threads_per_block = 256; 
            size_t num_thread_groups = (total_slice_states > 1) ? total_slice_states / 2 : 0;
            unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
            
            if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
            else if (num_thread_groups == 0 && total_slice_states > 0 && h->numLocalQubitsPerGpu > 0) num_blocks = 0; // e.g. 1 qubit, 1 group, but tpb > 1
            else if (h->numLocalQubitsPerGpu == 0 && targetQubitLocal == 0) num_blocks = (total_slice_states > 0) ? 1:0; // 0-qubit system
            else if (total_slice_states == 0) num_blocks = 0;


            if (num_blocks > 0 || (h->numLocalQubitsPerGpu == 0 && targetQubitLocal == 0 && total_slice_states > 0) ) {
                 hipLaunchKernelGGL(apply_single_qubit_generic_matrix_kernel,
                                   dim3(num_blocks ? num_blocks : 1), // Ensure at least 1 block if 0-qubit case selected
                                   dim3(threads_per_block),
                                   0,
                                   h->streams[i],
                                   h->d_local_state_slices[i],
                                   h->numLocalQubitsPerGpu, 
                                   localTargetQubit,        
                                   d_fusedMatrix);
                hipError_t hip_err = hipGetLastError();
                if (hip_err != hipSuccess) {
                    status = checkHipError(hip_err, "rocsvApplyFusedSingleQubitMatrix hipLaunchKernelGGL");
                    break; 
                }
            }
        } // End GPU loop

        if(status != ROCQ_STATUS_SUCCESS) return status; // Return early if error during kernel launch

        // Synchronize all streams after launching on all GPUs
        for (int i = 0; i < h->numGpus; ++i) {
            // hipSetDevice(h->deviceIds[i]); // Not strictly necessary if streams are correctly bound
            hipError_t sync_err = hipStreamSynchronize(h->streams[i]);
            if (sync_err != hipSuccess) {
                status = checkHipError(sync_err, "rocsvApplyFusedSingleQubitMatrix hipStreamSynchronize");
                // Potentially log this error, but try to sync other streams.
                // The first error encountered during kernel launch or earlier device set will be returned.
            }
        }
        return status;

    } else {
        // Global gate: requires communication via rocsvSwapIndexBits
        // Since rocsvSwapIndexBits is a stub, this path is not fully functional.
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }
}


// Helper function (conceptual) to reconstruct global index from rank and local index.
// This mapping assumes global index G = (rank_bits_as_MSB) | (local_index_bits_as_LSB).
// numLocalQubitsPerGpu: Number of qubits represented by local_idx.
static inline size_t rocquant_reconstruct_global_idx_from_slice_info( 
    int rank, 
    size_t local_idx, 
    unsigned numLocalQubitsPerGpu) {
    return (static_cast<size_t>(rank) << numLocalQubitsPerGpu) | local_idx;
}

// Helper function (conceptual) to get target rank and new local index from a global index.
// Inverse of rocquant_reconstruct_global_idx_from_slice_info.
static inline std::pair<int, size_t> rocquant_get_rank_and_local_idx_from_global( 
    size_t global_idx, 
    unsigned numLocalQubitsPerGpu) {
    int rank = static_cast<int>(global_idx >> numLocalQubitsPerGpu);
    size_t local_idx_mask = (1ULL << numLocalQubitsPerGpu) - 1;
    size_t local_idx = global_idx & local_idx_mask;
    return {rank, local_idx};
}

// Helper function (conceptual) to swap specified bits in a given number.
static inline size_t rocquant_swap_bits_in_value(size_t value, unsigned bit_pos1, unsigned bit_pos2) { 
    unsigned bit1_val = (value >> bit_pos1) & 1;
    unsigned bit2_val = (value >> bit_pos2) & 1;
    if (bit1_val != bit2_val) {
        value ^= (1ULL << bit_pos1); 
        value ^= (1ULL << bit_pos2); 
    }
    return value;
}


rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (qubit_idx1 == qubit_idx2) return ROCQ_STATUS_SUCCESS; 

    if (h->globalNumQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE; 
    }
    if (qubit_idx1 >= h->globalNumQubits || qubit_idx2 >= h->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (h->numGpus <= 1) return ROCQ_STATUS_SUCCESS; // No communication for single GPU 

    // NOTE ON TEMPORARY BUFFERS:
    // Ideally, temporary swap buffers (one per GPU, size of local slice) would be managed
    // within the rocsvInternalHandle (e.g., h->d_swap_buffers). However, due to persistent
    // tool issues preventing modifications to the handle structure and its management functions
    // (rocsvCreate, rocsvDestroy, rocsvAllocateDistributedState), this stub assumes that
    // such buffers would need to be allocated here, per call, or passed in if the signature allowed.
    // For this conceptual stub, we'll denote them as `d_temp_send_buffer_r` and `d_temp_recv_buffer_r`
    // which would need allocation and deallocation around the RCCL calls if not part of the handle.
    // This is a deviation from the preferred design.

    // --- Determine if communication is needed based on bit roles ---
    // Assumes fixed mapping: lower `h->numLocalQubitsPerGpu` bits are local, higher bits are slice-determining.
    bool is_q1_local_domain = (qubit_idx1 < h->numLocalQubitsPerGpu);
    bool is_q2_local_domain = (qubit_idx2 < h->numLocalQubitsPerGpu);

    if (is_q1_local_domain && is_q2_local_domain) {
        // Case 1: Both bits are local. Requires local shuffle on each GPU.
        // For each GPU r_src: Launch local_bit_swap_kernel(...);
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs local permutation kernel(s).
    } else if (!is_q1_local_domain && !is_q2_local_domain) {
        // Case 2: Both bits are slice-determining. Complex re-mapping of ranks.
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs different logic.
    }
    // Case 3: One local, one slice bit. Proceed with Alltoallv logic.

    // --- Stage 1: Prepare send buffers and calculate send/recv counts & displacements ---
    // This stage requires custom GPU kernels.
    // For each GPU `r_src` (from 0 to h->numGpus - 1):
    //   1. Allocate `d_temp_send_buffer_r` of size `h->localStateSizes[r_src]`.
    //   2. Launch `prepare_swap_data_kernel` on `h->streams[r_src]`:
    //      Inputs: `h->d_local_state_slices[r_src]`, `h->localStateSizes[r_src]`,
    //              `qubit_idx1`, `qubit_idx2`, `h->numLocalQubitsPerGpu`, `r_src`.
    //      Outputs: `d_temp_send_buffer_r` (packed data), `d_send_counts_for_r_src` (device array).
    //      Kernel logic: For each element, compute its new global index, target rank, and new local index.
    //                  Pack data into `d_temp_send_buffer_r` contiguously for each target rank.
    //                  Calculate `send_counts` for this `r_src` to all other ranks.
    //   3. Copy `d_send_counts_for_r_src` to host.
    // After loop:
    //   - Aggregate host send_counts to form global send/recv counts and displacements tables.
    //   - Copy these tables (per-GPU views) to device memory for RCCL.
    // std::cout << "SwapIndexBits: Stage 1 (Data prep) NOT IMPLEMENTED." << std::endl;


    // --- Stage 2: RCCL All-to-All Communication ---
    // rcclGroupStart();
    // for (int r = 0; r < h->numGpus; ++r) {
    //     hipSetDevice(h->deviceIds[r]);
    //     // Allocate d_temp_recv_buffer_r if not using local_state_slice directly as recv,
    //     // or if a separate buffer is needed before final permutation.
    //     // For in-place like operation with Alltoallv, send from d_temp_send_buffer_r, receive to d_local_state_slices[r].
    //     rcclResult_t err = rcclAlltoallv(
    //         /* (const void*) d_temp_send_buffer_r */ nullptr,       
    //         /* d_sCounts_r (device ptr) */ nullptr,                  
    //         /* d_sDispls_r (device ptr) */ nullptr,                  
    //         (void*) h->d_local_state_slices[r], 
    //         /* d_rCounts_r (device ptr) */ nullptr,                  
    //         /* d_rDispls_r (device ptr) */ nullptr,                  
    //         rcclFloatComplex, 
    //         h->comms[r], 
    //         h->streams[r]
    //     );
    //     // if (err != rcclSuccess) { /* handle error */ }
    // }
    // rcclGroupEnd();
    // std::cout << "SwapIndexBits: Stage 2 (RCCL Alltoallv) NOT IMPLEMENTED." << std::endl;

    // --- Cleanup Temporary Buffers ---
    // For each GPU `r_src`: Free `d_temp_send_buffer_r` (and `d_temp_recv_buffer_r` if used).
    // std::cout << "SwapIndexBits: Stage 2.5 (Temp buffer cleanup) NOT IMPLEMENTED." << std::endl;

    // --- Stage 3: (Optional) Final local permutation ---
    // If data in `h->d_local_state_slices[r]` is not in final sorted order by new local index,
    // a local permutation kernel would run here, possibly using an allocated temp buffer.
    // std::cout << "SwapIndexBits: Stage 3 (Final local permutation) NOT IMPLEMENTED." << std::endl;
    
    return ROCQ_STATUS_NOT_IMPLEMENTED;
}


// Helper function (conceptual) to reconstruct global index from rank and local index.
// This mapping assumes global index G = (rank_bits_as_MSB) | (local_index_bits_as_LSB).
// numLocalQubitsPerGpu: Number of qubits represented by local_idx.
// Example: For 8 total qubits, 2 slice qubits (4 GPUs), 6 local qubits.
// Rank 0: global indices 0 to 2^6-1
// Rank 1: global indices 2^6 to 2*(2^6)-1
// Global index g_idx = (rank << numLocalQubitsPerGpu) | local_idx;
static inline size_t rocquant_reconstruct_global_idx_from_slice_info( // Renamed to avoid potential conflicts
    int rank, 
    size_t local_idx, 
    unsigned numLocalQubitsPerGpu) {
    return (static_cast<size_t>(rank) << numLocalQubitsPerGpu) | local_idx;
}

// Helper function (conceptual) to get target rank and new local index from a global index.
// Inverse of reconstruct_global_idx_from_slice_info.
static inline std::pair<int, size_t> rocquant_get_rank_and_local_idx_from_global( // Renamed
    size_t global_idx, 
    unsigned numLocalQubitsPerGpu) {
    int rank = static_cast<int>(global_idx >> numLocalQubitsPerGpu);
    size_t local_idx_mask = (1ULL << numLocalQubitsPerGpu) - 1;
    size_t local_idx = global_idx & local_idx_mask;
    return {rank, local_idx};
}

// Helper function (conceptual) to swap specified bits in a given number.
static inline size_t rocquant_swap_bits_in_value(size_t value, unsigned bit_pos1, unsigned bit_pos2) { // Renamed
    unsigned bit1_val = (value >> bit_pos1) & 1;
    unsigned bit2_val = (value >> bit_pos2) & 1;
    if (bit1_val != bit2_val) {
        value ^= (1ULL << bit_pos1); // Flip bit at pos1
        value ^= (1ULL << bit_pos2); // Flip bit at pos2
    }
    return value;
}


rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (qubit_idx1 == qubit_idx2) return ROCQ_STATUS_SUCCESS; 

    if (h->globalNumQubits == 0) {
        // fprintf(stderr, "State vector not allocated or handle not initialized.\n");
        return ROCQ_STATUS_INVALID_VALUE; 
    }
    if (qubit_idx1 >= h->globalNumQubits || qubit_idx2 >= h->globalNumQubits) {
        // fprintf(stderr, "Qubit indices out of bounds.\n");
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (h->numGpus <= 1) return ROCQ_STATUS_SUCCESS; // No communication for single GPU 

    // NOTE: This function conceptually requires temporary swap buffers, one for each GPU,
    // ideally managed within the rocsvInternalHandle structure (e.g., as h->d_swap_buffers).
    // Due to persistent tool-related issues preventing modifications to the handle
    // and its management functions (rocsvCreate, rocsvDestroy, rocsvAllocateDistributedState),
    // these buffers are NOT currently part of the handle. A full implementation would
    // need to ensure these buffers are available, e.g., by allocating them here or ensuring
    // they are pre-allocated via the handle. For this stub, their existence and correct
    // sizing (equal to h->localStateSizes[rank]) is assumed for the conceptual explanation.
    // std::cout << "Info: rocsvSwapIndexBits assumes temporary swap buffers are available per GPU." << std::endl;


    // --- Determine if communication is needed based on bit roles ---
    // The state distribution scheme assumes that the `h->numLocalQubitsPerGpu` least significant
    // bits of a global state index correspond to the local index within a GPU slice,
    // and the remaining `h->numGlobalSliceQubits` most significant bits determine the GPU rank.
    // (This assumes `qubit_idxN` refers to its position in the global index, 0 to `globalNumQubits-1`).
    bool is_q1_local_domain = (qubit_idx1 < h->numLocalQubitsPerGpu);
    bool is_q2_local_domain = (qubit_idx2 < h->numLocalQubitsPerGpu);

    if (is_q1_local_domain && is_q2_local_domain) {
        // Case 1: Both bits are within the local domain of each GPU slice.
        // Requires a local permutation kernel on each GPU. No RCCL communication.
        // For each GPU r_src from 0 to numGpus-1:
        //   hipSetDevice(h->deviceIds[r_src]);
        //   Launch local_bit_swap_kernel(h->d_local_state_slices[r_src], h->localStateSizes[r_src],
        //                                qubit_idx1, qubit_idx2, h->streams[r_src]);
        // This kernel would permute elements within d_local_state_slices[r_src].
        // std::cout << "Info: Swapping two local-domain bits. Requires local permutation kernel (stubbed)." << std::endl;
        return ROCQ_STATUS_NOT_IMPLEMENTED; 
    } else if (!is_q1_local_domain && !is_q2_local_domain) {
        // Case 2: Both bits are within the slice-determining domain (i.e., both affect GPU rank).
        // This changes which ranks (GPUs) own which parts of the state vector based on these two bits.
        // This is a more complex metadata and data re-mapping than simple Alltoallv of current slices.
        // std::cout << "Info: Swapping two slice-domain bits. Complex rank re-mapping (stubbed)." << std::endl;
        return ROCQ_STATUS_NOT_IMPLEMENTED; 
    }
    // Case 3: One bit is local-domain, one is slice-domain. This is the primary target for Alltoallv.
    // std::cout << "Info: Swapping one local-domain bit with one slice-domain bit. Alltoallv pattern." << std::endl;


    // --- Stage 1: Prepare send buffers and calculate send/recv counts & displacements ---
    // This stage requires custom GPU kernels for efficiency.
    // For each GPU `r_src` (from 0 to h->numGpus - 1):
    //   1. Set device context: `hipSetDevice(h->deviceIds[r_src])`.
    //   2. Launch a kernel (e.g., `prepare_swap_data_kernel`) on `h->streams[r_src]`.
    //      This kernel takes:
    //        - Input: `h->d_local_state_slices[r_src]` (current data on this GPU).
    //        - Input: `h->localStateSizes[r_src]` (number of elements).
    //        - Input: `qubit_idx1`, `qubit_idx2` (global bit positions to swap).
    //        - Input: `h->numLocalQubitsPerGpu`.
    //        - Input: `r_src` (current GPU's rank).
    //      This kernel calculates for each local amplitude:
    //        a. Its original `global_idx` using `rocquant_reconstruct_global_idx_from_slice_info`.
    //        b. The `swapped_global_idx` using `rocquant_swap_bits_in_value`.
    //        c. The `target_rank` and `new_local_idx_on_target` from `swapped_global_idx`
    //           using `rocquant_get_rank_and_local_idx_from_global`.
    //      The kernel then needs to:
    //        - Count how many amplitudes are destined for each `target_rank`. This results in an array
    //          `d_send_counts_for_r_src[numGpus]` for the current `r_src`. (Can be done with atomics or a reduction).
    //        - Based on these counts, calculate send displacements for packing.
    //        - Write each amplitude from `h->d_local_state_slices[r_src]` into the correct position
    //          in the temporary swap buffer for this GPU (conceptually `h->d_swap_buffers[r_src]`).
    //          The data in this swap buffer must be packed: all data for target GPU 0, then all for target GPU 1, etc.
    //          The order within each block should correspond to `new_local_idx_on_target`.
    //
    //   3. After the kernel, `d_send_counts_for_r_src` is copied to a host array for `r_src`.
    //
    // After iterating all `r_src`:
    //   - The host now has all `send_counts`.
    //   - From this, derive `recv_counts` for each GPU.
    //   - Calculate send and receive displacements for each GPU.
    //   - Copy these per-GPU counts and displacements arrays to device memory for `rcclAlltoallv`.

    // std::cout << "SwapIndexBits: Stage 1 (Data preparation for Alltoallv) is complex and kernel-dependent, NOT IMPLEMENTED." << std::endl;


    // --- Stage 2: RCCL All-to-All Communication ---
    // This performs the actual data exchange between GPUs.
    // rcclGroupStart();
    // for (int r = 0; r < h->numGpus; ++r) {
    //     hipSetDevice(h->deviceIds[r]);
    //     // d_sCounts_r, d_sDispls_r, d_rCounts_r, d_rDispls_r are device pointers to arrays of size numGpus.
    //     // The send buffer is the temporary swap buffer for rank r.
    //     // The receive buffer is h->d_local_state_slices[r].
    //     rcclResult_t err = rcclAlltoallv(
    //         /* (const void*) h->d_swap_buffers[r] */ placeholder_send_buffer_ptr_r,       
    //         d_sCounts_r,                  
    //         d_sDispls_r,                  
    //         (void*) h->d_local_state_slices[r], 
    //         d_rCounts_r,                  
    //         d_rDispls_r,                  
    //         rcclFloatComplex,             // Assuming rocComplex is hipFloatComplex
    //         h->comms[r], 
    //         h->streams[r]
    //     );
    //     if (err != rcclSuccess) {
    //         // rcclGroupEnd(); 
    //         return checkRcclError(err, "rocsvSwapIndexBits rcclAlltoallv");
    //     }
    // }
    // rcclResult_t group_err = rcclGroupEnd();
    // if (group_err != rcclSuccess) return checkRcclError(group_err, "rocsvSwapIndexBits rcclGroupEnd");
    // std::cout << "SwapIndexBits: Stage 2 (RCCL Alltoallv) call is NOT IMPLEMENTED." << std::endl;


    // --- Stage 3: (Optional) Final local permutation ---
    // If Stage 1 did not perfectly order data in the swap buffers according to the final
    // `new_local_idx_on_target` for each destination block, or if Alltoallv itself doesn't
    // guarantee this final local order, a final local permutation kernel might be needed.
    // This kernel would read from `h->d_local_state_slices[r]` (where Alltoallv placed data)
    // and write back to `h->d_local_state_slices[r]` in the correct local index order,
    // possibly using the temporary swap buffer for rank r as working space.
    // std::cout << "SwapIndexBits: Stage 3 (Optional final local sort/permutation kernel) is NOT IMPLEMENTED." << std::endl;

    // --- Stage 4: Update qubit mapping metadata in handle (if dynamic) ---
    // If the definition of which global qubit indices are "slice-determining" versus "local-part"
    // can change due to such swaps (e.g., if the set of slice-determining bits is not fixed
    // to be the MSBs), the handle's metadata (e.g., `h->numLocalQubitsPerGpu`,
    // `h->numGlobalSliceQubits`, or a more explicit list/map of slice-determining bits) needs to be updated.
    // This stub assumes the primary effect is data movement, and that the interpretation of
    // `numLocalQubitsPerGpu` and `numGlobalSliceQubits` remains consistent with the bit positions.

    // std::cout << "Info: rocsvSwapIndexBits conceptual data redistribution complete." << std::endl;
    
    return ROCQ_STATUS_NOT_IMPLEMENTED; // Due to missing GPU kernels, detailed comms logic, and prerequisite buffer management issues.
}


// Helper function (conceptual) to reconstruct global index from rank and local index.
// This mapping assumes global index G = (rank_bits_as_MSB) | (local_index_bits_as_LSB).
// numLocalQubitsPerGpu: Number of qubits represented by local_idx.
// Example: For 8 total qubits, 2 slice qubits (4 GPUs), 6 local qubits.
// Rank 0: global indices 0 to 2^6-1
// Rank 1: global indices 2^6 to 2*(2^6)-1
// Global index g_idx = (rank << numLocalQubitsPerGpu) | local_idx;
static inline size_t rocquant_reconstruct_global_idx_from_slice_info( // Renamed to avoid potential conflicts
    int rank, 
    size_t local_idx, 
    unsigned numLocalQubitsPerGpu) {
    return (static_cast<size_t>(rank) << numLocalQubitsPerGpu) | local_idx;
}

// Helper function (conceptual) to get target rank and new local index from a global index.
// Inverse of reconstruct_global_idx_from_slice_info.
static inline std::pair<int, size_t> rocquant_get_rank_and_local_idx_from_global( // Renamed
    size_t global_idx, 
    unsigned numLocalQubitsPerGpu) {
    int rank = static_cast<int>(global_idx >> numLocalQubitsPerGpu);
    size_t local_idx_mask = (1ULL << numLocalQubitsPerGpu) - 1;
    size_t local_idx = global_idx & local_idx_mask;
    return {rank, local_idx};
}

// Helper function (conceptual) to swap specified bits in a given number.
static inline size_t rocquant_swap_bits_in_value(size_t value, unsigned bit_pos1, unsigned bit_pos2) { // Renamed
    unsigned bit1_val = (value >> bit_pos1) & 1;
    unsigned bit2_val = (value >> bit_pos2) & 1;
    if (bit1_val != bit2_val) {
        value ^= (1ULL << bit_pos1); // Flip bit at pos1
        value ^= (1ULL << bit_pos2); // Flip bit at pos2
    }
    return value;
}


rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (qubit_idx1 == qubit_idx2) return ROCQ_STATUS_SUCCESS; 

    if (h->globalNumQubits == 0) {
        // fprintf(stderr, "State vector not allocated or handle not initialized.\n");
        return ROCQ_STATUS_INVALID_VALUE; 
    }
    if (qubit_idx1 >= h->globalNumQubits || qubit_idx2 >= h->globalNumQubits) {
        // fprintf(stderr, "Qubit indices out of bounds.\n");
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (h->numGpus <= 1) return ROCQ_STATUS_SUCCESS; // No communication for single GPU 

    // NOTE: The following section assumes that `h->d_swap_buffers` (a std::vector<rocComplex*>) 
    // exists within the rocsvInternalHandle structure and is properly allocated for each GPU.
    // Each h->d_swap_buffers[rank] should point to a device buffer of size
    // h->localStateSizes[rank]. Attempts to modify rocsvInternalHandle, rocsvCreate, 
    // rocsvDestroy, and rocsvAllocateDistributedState to include and manage these buffers
    // encountered persistent tool-related diff application issues and could not be completed.
    // If these prerequisite changes were successful, checks for buffer validity would be essential here.
    /*
    if (h->d_swap_buffers.empty() || h->d_swap_buffers.size() != static_cast<size_t>(h->numGpus)) {
        // fprintf(stderr, "Swap buffers not allocated or incorrectly sized in the handle.\n");
        return ROCQ_STATUS_FAILURE; 
    }
    for(int i=0; i < h->numGpus; ++i) {
        if (h->d_swap_buffers.empty() || h->d_swap_buffers[i] == nullptr) { // Check individual buffers
            // fprintf(stderr, "Swap buffer for GPU %d is null.\n", i);
            return ROCQ_STATUS_FAILURE;
        }
    }
    */


    // --- Determine if communication is needed based on bit roles ---
    // The state distribution scheme assumes that the `h->numLocalQubitsPerGpu` least significant
    // bits of a global state index correspond to the local index within a GPU slice,
    // and the remaining `h->numGlobalSliceQubits` most significant bits determine the GPU rank.
    bool is_q1_local_domain = (qubit_idx1 < h->numLocalQubitsPerGpu);
    bool is_q2_local_domain = (qubit_idx2 < h->numLocalQubitsPerGpu);

    if (is_q1_local_domain && is_q2_local_domain) {
        // Case 1: Both bits are within the local domain of each GPU slice.
        // Requires a local permutation kernel on each GPU. No RCCL communication.
        // std::cout << "Info: Swapping two local-domain bits. Requires local permutation kernel (stubbed)." << std::endl;
        // For each GPU r_src from 0 to numGpus-1:
        //   hipSetDevice(h->deviceIds[r_src]);
        //   Launch local_bit_swap_kernel(h->d_local_state_slices[r_src], h->localStateSizes[r_src],
        //                                qubit_idx1, qubit_idx2, h->streams[r_src]);
        // This kernel would permute elements within d_local_state_slices[r_src].
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs local permutation kernel(s).
    } else if (!is_q1_local_domain && !is_q2_local_domain) {
        // Case 2: Both bits are within the slice-determining domain.
        // This effectively remaps which GPU rank corresponds to which segment of the original state vector.
        // Data might not physically move between GPUs if the logical remapping is handled by changing
        // how `h->comms[physical_gpu_id]` maps to effective `rank` in subsequent operations,
        // or it could involve a full data reshuffle if physical slices must match logical ranks.
        // This is generally more complex than the Alltoallv pattern for a single slice/local bit swap.
        // std::cout << "Info: Swapping two slice-domain bits. Complex rank re-mapping (stubbed)." << std::endl;
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs different logic (e.g., metadata update or specific comm pattern).
    }
    // Case 3: One bit is local-domain, one is slice-domain. This is the primary target for Alltoallv.
    // std::cout << "Info: Swapping one local-domain bit with one slice-domain bit. Alltoallv pattern." << std::endl;


    // --- Stage 1: Prepare send buffers and calculate send/recv counts & displacements ---
    // This stage requires custom GPU kernels for efficiency.
    // For each GPU `r_src` (from 0 to h->numGpus - 1):
    //   1. Set device context: `hipSetDevice(h->deviceIds[r_src])`.
    //   2. Launch a kernel (e.g., `prepare_swap_data_kernel`) on `h->streams[r_src]`.
    //      This kernel takes:
    //        - Input: `h->d_local_state_slices[r_src]` (current data on this GPU).
    //        - Input: `h->localStateSizes[r_src]` (number of elements).
    //        - Input: `qubit_idx1`, `qubit_idx2` (global bit positions to swap).
    //        - Input: `h->numLocalQubitsPerGpu`.
    //        - Input: `r_src` (current GPU's rank).
    //      This kernel calculates for each local amplitude:
    //        a. Its original `global_idx` using `rocquant_reconstruct_global_idx_from_slice_info`.
    //        b. The `swapped_global_idx` using `rocquant_swap_bits_in_value`.
    //        c. The `target_rank` and `new_local_idx_on_target` from `swapped_global_idx`
    //           using `rocquant_get_rank_and_local_idx_from_global`.
    //      The kernel then needs to:
    //        - Count how many amplitudes are destined for each `target_rank`. This results in an array
    //          `d_send_counts_for_r_src[numGpus]` for the current `r_src`. (Can be done with atomics or a reduction).
    //        - Based on these counts, calculate send displacements for packing.
    //        - Write each amplitude from `h->d_local_state_slices[r_src]` into the correct position
    //          in `h->d_swap_buffers[r_src]`. The data in `h->d_swap_buffers[r_src]` must be packed:
    //          all data for target GPU 0, then all for target GPU 1, etc. The order within each
    //          block (for a given target GPU) should correspond to `new_local_idx_on_target` to simplify
    //          the receive side or avoid a final local shuffle.
    //
    //   3. After the kernel, `d_send_counts_for_r_src` is copied to a host array for `r_src`.
    //
    // After iterating all `r_src`:
    //   - The host now has all `send_counts` (e.g., in a `std::vector<std::vector<int>> h_all_send_counts`).
    //   - From this, derive `recv_counts` for each GPU. For GPU `g_dst`, its `recv_counts_from_g_src` is
    //     `h_all_send_counts[g_src][g_dst]`.
    //   - Calculate send and receive displacements for each GPU.
    //   - Copy these per-GPU `send_counts`, `sdispls`, `recv_counts`, `rdispls` arrays to device memory
    //     for use in `rcclAlltoallv`.

    // std::cout << "SwapIndexBits: Stage 1 (Data preparation for Alltoallv) is complex and kernel-dependent, NOT IMPLEMENTED." << std::endl;


    // --- Stage 2: RCCL All-to-All Communication ---
    // This performs the actual data exchange between GPUs.
    // rcclGroupStart();
    // for (int r = 0; r < h->numGpus; ++r) {
    //     hipSetDevice(h->deviceIds[r]);
    //     // d_sCounts_r, d_sDispls_r, d_rCounts_r, d_rDispls_r are device pointers to arrays of size numGpus,
    //     // containing the send/recv counts and displacements for rank r.
    //     rcclResult_t err = rcclAlltoallv(
    //         h->d_swap_buffers[r],         // Send buffer (prepared in Stage 1)
    //         d_sCounts_r,                  
    //         d_sDispls_r,                  
    //         h->d_local_state_slices[r],   // Receive buffer (where data is placed)
    //         d_rCounts_r,                  
    //         d_rDispls_r,                  
    //         rcclFloatComplex,             // Assuming rocComplex is hipFloatComplex
    //         h->comms[r], 
    //         h->streams[r]
    //     );
    //     if (err != rcclSuccess) {
    //         // rcclGroupEnd(); // Attempt to end group on error.
    //         return checkRcclError(err, "rocsvSwapIndexBits rcclAlltoallv");
    //     }
    // }
    // rcclResult_t group_err = rcclGroupEnd();
    // if (group_err != rcclSuccess) return checkRcclError(group_err, "rocsvSwapIndexBits rcclGroupEnd");
    // std::cout << "SwapIndexBits: Stage 2 (RCCL Alltoallv) call is NOT IMPLEMENTED." << std::endl;


    // --- Stage 3: (Optional) Final local permutation ---
    // If Stage 1 did not perfectly order data in the swap buffers according to the final
    // `new_local_idx_on_target` for each destination block, or if Alltoallv itself doesn't
    // guarantee this final local order based on how displacements were calculated,
    // a final local permutation kernel might be needed here.
    // This kernel would read from `h->d_local_state_slices[r]` (where Alltoallv placed data)
    // and write back to `h->d_local_state_slices[r]` in the correct local index order,
    // possibly using `h->d_swap_buffers[r]` as temporary storage.
    // std::cout << "SwapIndexBits: Stage 3 (Optional final local sort/permutation kernel) is NOT IMPLEMENTED." << std::endl;

    // --- Stage 4: Update qubit mapping metadata in handle (if dynamic) ---
    // If the definition of which global qubit indices are "slice-determining" versus "local-part"
    // can change due to such swaps, the handle's metadata (e.g., `h->numLocalQubitsPerGpu`,
    // `h->numGlobalSliceQubits`, or a more explicit list of slice-determining bits) needs to be updated.
    // For this stub, we assume the primary effect is data movement, and metadata changes are
    // handled by the caller or a higher-level system if the mapping is dynamic.

    // std::cout << "Info: rocsvSwapIndexBits conceptual data redistribution complete." << std::endl;
    
    return ROCQ_STATUS_NOT_IMPLEMENTED; // Due to missing GPU kernels and detailed comms logic, and prerequisite buffer management issues.
}


// Helper function (conceptual) to reconstruct global index from rank and local index.
// This mapping assumes global index G = (rank_bits_as_MSB) | (local_index_bits_as_LSB).
// numLocalQubitsPerGpu: Number of qubits represented by local_idx.
// Example: For 8 total qubits, 2 slice qubits (4 GPUs), 6 local qubits.
// Rank 0: global indices 0 to 2^6-1
// Rank 1: global indices 2^6 to 2*(2^6)-1
// Global index g_idx = (rank << numLocalQubitsPerGpu) | local_idx;
static inline size_t rocquant_reconstruct_global_idx_from_slice_info( // Renamed to avoid potential conflicts
    int rank, 
    size_t local_idx, 
    unsigned numLocalQubitsPerGpu) {
    return (static_cast<size_t>(rank) << numLocalQubitsPerGpu) | local_idx;
}

// Helper function (conceptual) to get target rank and new local index from a global index.
// Inverse of reconstruct_global_idx_from_slice_info.
static inline std::pair<int, size_t> rocquant_get_rank_and_local_idx_from_global( // Renamed
    size_t global_idx, 
    unsigned numLocalQubitsPerGpu) {
    int rank = static_cast<int>(global_idx >> numLocalQubitsPerGpu);
    size_t local_idx_mask = (1ULL << numLocalQubitsPerGpu) - 1;
    size_t local_idx = global_idx & local_idx_mask;
    return {rank, local_idx};
}

// Helper function (conceptual) to swap specified bits in a given number.
static inline size_t rocquant_swap_bits_in_value(size_t value, unsigned bit_pos1, unsigned bit_pos2) { // Renamed
    unsigned bit1_val = (value >> bit_pos1) & 1;
    unsigned bit2_val = (value >> bit_pos2) & 1;
    if (bit1_val != bit2_val) {
        value ^= (1ULL << bit_pos1); // Flip bit at pos1
        value ^= (1ULL << bit_pos2); // Flip bit at pos2
    }
    return value;
}


rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (qubit_idx1 == qubit_idx2) return ROCQ_STATUS_SUCCESS; 

    if (h->globalNumQubits == 0) {
        // fprintf(stderr, "State vector not allocated or handle not initialized.\n");
        return ROCQ_STATUS_INVALID_VALUE; 
    }
    if (qubit_idx1 >= h->globalNumQubits || qubit_idx2 >= h->globalNumQubits) {
        // fprintf(stderr, "Qubit indices out of bounds.\n");
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (h->numGpus <= 1) return ROCQ_STATUS_SUCCESS; // No communication for single GPU 

    // NOTE: The following section assumes that `h->d_swap_buffers` exists and is properly allocated
    // for each GPU (i.e., `h->d_swap_buffers[rank]` is a device pointer to a buffer of size
    // `h->localStateSizes[rank]`). Modifications to rocsvInternalHandle, rocsvCreate, 
    // rocsvDestroy, and rocsvAllocateDistributedState to manage these buffers were attempted 
    // but failed due to persistent tool-related diff application issues.
    // If these were successful, uncommenting checks like the one below would be appropriate:
    /*
    if (h->d_swap_buffers.empty() || h->d_swap_buffers.size() != static_cast<size_t>(h->numGpus)) {
        // fprintf(stderr, "Swap buffers not allocated or incorrectly sized in the handle.\n");
        return ROCQ_STATUS_FAILURE; 
    }
    for(int i=0; i < h->numGpus; ++i) {
        if (h->d_swap_buffers[i] == nullptr) {
            // fprintf(stderr, "Swap buffer for GPU %d is null.\n", i);
            return ROCQ_STATUS_FAILURE;
        }
    }
    */


    // --- Determine if communication is needed based on bit roles ---
    // This logic assumes a fixed mapping: lower `numLocalQubitsPerGpu` are local,
    // higher bits up to `globalNumQubits` can be slice bits if `numGlobalSliceQubits > 0`.
    // A more robust implementation might involve a dynamic mapping of global qubit indices to their roles.
    
    // The problem defines slice bits implicitly as those that distinguish between GPU slices.
    // If numLocalQubitsPerGpu = N_L and numGlobalSliceQubits = N_S,
    // Total qubits = N_L + N_S.
    // Typically, global_idx = (rank_part << N_L) | local_part.
    // rank_part is formed by N_S bits. local_part is formed by N_L bits.
    // A qubit index `q` (from 0 to Total_qubits-1) is a "local domain bit" if `q < N_L`.
    // A qubit index `q` is a "slice domain bit" if `q >= N_L`.
    
    bool is_q1_local_domain = (qubit_idx1 < h->numLocalQubitsPerGpu);
    bool is_q2_local_domain = (qubit_idx2 < h->numLocalQubitsPerGpu);

    if (is_q1_local_domain && is_q2_local_domain) {
        // Both bits are within the local domain of each GPU slice.
        // Requires a local permutation kernel on each GPU. No RCCL communication.
        // std::cout << "Info: Swapping two local-domain bits. Requires local permutation kernel (stubbed)." << std::endl;
        // For each GPU r_src from 0 to numGpus-1:
        //   hipSetDevice(h->deviceIds[r_src]);
        //   Launch local_bit_swap_kernel(h->d_local_state_slices[r_src], h->localStateSizes[r_src],
        //                                qubit_idx1, qubit_idx2, h->streams[r_src]);
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs local kernel
    } else if (!is_q1_local_domain && !is_q2_local_domain) {
        // Both bits are within the slice-determining domain.
        // This changes which ranks (GPUs) own which parts of the state vector.
        // This is a more complex metadata and data re-mapping than simple Alltoallv of current slices.
        // std::cout << "Info: Swapping two slice-domain bits. Complex rank re-mapping (stubbed)." << std::endl;
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs different kind of logic
    }
    // Else, one bit is local-domain, one is slice-domain. This is the primary target for Alltoallv.
    // std::cout << "Info: Swapping one local-domain bit with one slice-domain bit. Alltoallv pattern." << std::endl;


    // --- Stage 1: Prepare send buffers and calculate send/recv counts & displacements ---
    // This stage involves complex GPU kernels. Outline:
    // For each GPU `r_src` from 0 to numGpus-1:
    //   hipSetDevice(h->deviceIds[r_src]);
    //   Launch a kernel (e.g., `prepare_alltoall_data_kernel`) on `h->streams[r_src]`:
    //     Input: `h->d_local_state_slices[r_src]`, `h->localStateSizes[r_src]`,
    //            `qubit_idx1`, `qubit_idx2` (global bit positions),
    //            `h->numLocalQubitsPerGpu`.
    //     Output: `h->d_swap_buffers[r_src]` (data packed for Alltoallv),
    //             `d_send_counts_for_this_gpu` (array of size numGpus, on device).
    //   The kernel logic for each element at `old_local_idx` in `d_local_state_slices[r_src]`:
    //       1. `current_global_idx = rocquant_reconstruct_global_idx_from_slice_info(r_src, old_local_idx, h->numLocalQubitsPerGpu)`.
    //       2. `swapped_global_idx = rocquant_swap_bits_in_value(current_global_idx, qubit_idx1, qubit_idx2)`.
    //       3. `target_info = rocquant_get_rank_and_local_idx_from_global(swapped_global_idx, h->numLocalQubitsPerGpu)`.
    //          `target_rank = target_info.first; new_local_idx_on_target = target_info.second;`
    //       4. This element is destined for `target_rank`. Store its value and `new_local_idx_on_target`.
    //       5. Atomically increment `d_send_counts_for_this_gpu[target_rank]`. (This is one way to get counts).
    //     After counts are known (e.g., via a separate prefix-scan over target_ranks or atomic ops):
    //       Calculate send displacements (`sdispls`) from `d_send_counts_for_this_gpu`.
    //       A second kernel pass or careful indexing in the first pass:
    //         Place element into `h->d_swap_buffers[r_src]` at an offset based on `target_rank`'s segment
    //         and its position within that segment (which should correspond to `new_local_idx_on_target`
    //         if the data is to be immediately usable or easily permuted on the target).
    //
    //   After the kernel(s), `d_send_counts_for_this_gpu` (device array) is copied to host for each GPU.
    //   These are aggregated to form global `h_send_counts_all_gpus` and `h_recv_counts_all_gpus` tables.
    //   `recv_counts[dest_gpu * numGpus + src_gpu] = send_counts[src_gpu * numGpus + dest_gpu]`.
    //   From these host tables, per-GPU device arrays for `send_counts`, `sdispls`, `recv_counts`, `rdispls` are created for Alltoallv.

    // std::cout << "SwapIndexBits: Stage 1 (Data preparation for Alltoallv) is complex and kernel-dependent, NOT IMPLEMENTED." << std::endl;


    // --- Stage 2: RCCL All-to-All Communication ---
    // rcclGroupStart();
    // for (int r = 0; r < h->numGpus; ++r) {
    //     hipSetDevice(h->deviceIds[r]);
    //     // d_sCounts_r, d_sDispls_r, d_rCounts_r, d_rDispls_r are device pointers for rank r's view of comms.
    //     // These would have been prepared based on the global count/displacement calculation from Stage 1.
    //     // The send buffer is h->d_swap_buffers[r]. The receive buffer is h->d_local_state_slices[r].
    //     rcclResult_t err = rcclAlltoallv(
    //         h->d_swap_buffers[r],       
    //         d_sCounts_r,                // Device array[numGpus] of send counts from rank r to all ranks
    //         d_sDispls_r,                // Device array[numGpus] of send displacements
    //         h->d_local_state_slices[r], 
    //         d_rCounts_r,                // Device array[numGpus] of receive counts for rank r from all ranks
    //         d_rDispls_r,                // Device array[numGpus] of receive displacements
    //         rcclFloatComplex,           // Assuming rocComplex is hipFloatComplex
    //         h->comms[r], 
    //         h->streams[r]
    //     );
    //     if (err != rcclSuccess) {
    //         // rcclGroupEnd(); // Attempt to end group on error.
    //         return checkRcclError(err, "rocsvSwapIndexBits rcclAlltoallv");
    //     }
    // }
    // rcclResult_t group_err = rcclGroupEnd();
    // if (group_err != rcclSuccess) return checkRcclError(group_err, "rocsvSwapIndexBits rcclGroupEnd");
    // std::cout << "SwapIndexBits: Stage 2 (RCCL Alltoallv) call is NOT IMPLEMENTED." << std::endl;


    // --- Stage 3: (Optional) Final local permutation ---
    // If the Alltoallv received data into `h->d_local_state_slices[r]` but it's grouped by source GPU
    // rather than sorted by the `new_local_idx_on_target`, a final local permutation kernel is needed.
    // This kernel would use `h->d_swap_buffers[r]` as temporary working space.
    // hipLaunchKernelGGL(final_local_sort_by_new_local_idx_kernel, ..., 
    //                    h->d_local_state_slices[r], h->d_swap_buffers[r], ...);
    // std::cout << "SwapIndexBits: Stage 3 (Optional final local sort kernel) is NOT IMPLEMENTED." << std::endl;


    // --- Stage 4: Update qubit mapping metadata in handle (IMPORTANT if roles of bits change) ---
    // This stub assumes the fixed bit position definitions for slice vs local domains do not change.
    // The data has been moved to reflect the swap of logical qubit roles at qubit_idx1 and qubit_idx2.
    // If the definition of which global bit positions are "slice-determining" can change,
    // then h->numGlobalSliceQubits and h->numLocalQubitsPerGpu (or a more detailed mapping
    // like `std::vector<unsigned> slice_determining_qubit_indices_sorted;`) would need an update here.

    // std::cout << "Info: rocsvSwapIndexBits conceptual data redistribution complete." << std::endl;
    
    return ROCQ_STATUS_NOT_IMPLEMENTED; // Due to missing GPU kernels and detailed comms logic.
}


// Helper function (conceptual) to reconstruct global index from rank and local index.
// This mapping assumes global index G = (rank_bits_as_MSB) | (local_index_bits_as_LSB).
// numLocalQubitsPerGpu: Number of qubits represented by local_idx.
// Example: For 8 total qubits, 2 slice qubits (4 GPUs), 6 local qubits.
// Rank 0: global indices 0 to 2^6-1
// Rank 1: global indices 2^6 to 2*(2^6)-1
// Global index g_idx = (rank << numLocalQubitsPerGpu) | local_idx;
static inline size_t reconstruct_global_idx_from_slice_info(
    int rank, 
    size_t local_idx, 
    unsigned numLocalQubitsPerGpu) {
    return (static_cast<size_t>(rank) << numLocalQubitsPerGpu) | local_idx;
}

// Helper function (conceptual) to get target rank and new local index from a global index.
// Inverse of reconstruct_global_idx_from_slice_info.
static inline std::pair<int, size_t> get_rank_and_local_idx_from_global(
    size_t global_idx, 
    unsigned numLocalQubitsPerGpu) {
    int rank = static_cast<int>(global_idx >> numLocalQubitsPerGpu);
    size_t local_idx_mask = (1ULL << numLocalQubitsPerGpu) - 1;
    size_t local_idx = global_idx & local_idx_mask;
    return {rank, local_idx};
}

// Helper function (conceptual) to swap specified bits in a given number.
static inline size_t swap_bits_in_value(size_t value, unsigned bit_pos1, unsigned bit_pos2) {
    unsigned bit1_val = (value >> bit_pos1) & 1;
    unsigned bit2_val = (value >> bit_pos2) & 1;
    if (bit1_val != bit2_val) {
        value ^= (1ULL << bit_pos1); // Flip bit at pos1
        value ^= (1ULL << bit_pos2); // Flip bit at pos2
    }
    return value;
}


rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (qubit_idx1 == qubit_idx2) return ROCQ_STATUS_SUCCESS; 

    if (h->globalNumQubits == 0) {
        // fprintf(stderr, "State vector not allocated or handle not initialized.\n");
        return ROCQ_STATUS_INVALID_VALUE; 
    }
    if (qubit_idx1 >= h->globalNumQubits || qubit_idx2 >= h->globalNumQubits) {
        // fprintf(stderr, "Qubit indices out of bounds.\n");
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (h->numGpus <= 1) return ROCQ_STATUS_SUCCESS; // No communication for single GPU 

    // Critical Check: Ensure buffers are notionally available.
    // Since previous steps to add d_swap_buffers to the handle and manage its allocation
    // via rocsvCreate/Destroy/AllocateDistributedState failed due to tool issues,
    // we proceed with a note that these buffers are assumed to exist and be correctly sized.
    // If this were fully implemented, we'd check:
    // if (h->d_swap_buffers.empty() || h->d_swap_buffers[0] == nullptr) {
    //     fprintf(stderr, "Swap buffers not allocated in the handle.\n");
    //     return ROCQ_STATUS_FAILURE;
    // }
    // For the purpose of this stub, we'll log this assumption.
    // std::cout << "Info: rocsvSwapIndexBits assumes d_swap_buffers are managed within the handle." << std::endl;


    // --- Determine if communication is needed ---
    // The distribution is based on the top `h->numGlobalSliceQubits` of the global qubit index.
    // `qubit_idxN` are global qubit indices.
    // Local part uses `h->numLocalQubitsPerGpu` qubits.
    // A qubit is a "slice bit" if its global index `q` satisfies `q >= h->numLocalQubitsPerGpu`.
    // A qubit is a "local bit" if its global index `q` satisfies `q < h->numLocalQubitsPerGpu`.
    // (This assumes slice bits are the most significant bits of the global index).
    // This interpretation needs to be consistent with how `numGlobalSliceQubits` and `numLocalQubitsPerGpu` are defined.
    // Let's refine:
    // A global qubit index `q` (from 0 to globalNumQubits-1) determines its role.
    // The mapping is typically: global_index's top `numGlobalSliceQubits` map to the rank.
    // The remaining `numLocalQubitsPerGpu` bits map to the local index on that rank.
    // So, a qubit `q_global` is a "slice-determining bit" if it's one of the bits used to form the rank index.
    // It's a "local-part bit" if it's one of the bits forming the local index within a slice.
    // This function is most impactful when swapping a slice-determining bit with a local-part bit.

    // Example: 5 qubits total (q4,q3,q2,q1,q0), 2 GPUs (numGlobalSliceQubits=1).
    // Slice bit: q4 (highest bit, index 4). Local bits: q3,q2,q1,q0 (indices 0-3).
    // numLocalQubitsPerGpu = 4.
    // If qubit_idx1 = q4 (a slice bit) and qubit_idx2 = q1 (a local bit), communication is needed.
    // If qubit_idx1 = q1 and qubit_idx2 = q2 (both local bits), only local shuffle on each GPU.
    // If qubit_idx1 = q4 and qubit_idx2 = q3 (assuming q3 is another slice bit if numGlobalSliceQubits > 1), complex.

    bool is_q1_slice_bit = (qubit_idx1 >= h->numLocalQubitsPerGpu); 
    bool is_q2_slice_bit = (qubit_idx2 >= h->numLocalQubitsPerGpu);
    // This definition of slice_bit vs local_bit assumes that the slice bits are contiguous
    // at the most significant end of the global qubit index range, which is a common convention.

    if (is_q1_slice_bit == is_q2_slice_bit) {
        // Both are slice bits OR both are local bits.
        // If both local: perform local shuffle on each GPU. Kernel needed. (Not implemented here)
        // If both slice bits: this is a permutation of GPU ranks themselves. Data might not move between
        // physical devices but logical role of d_local_state_slices[i] changes. (More complex, not Alltoallv pattern).
        // For now, this function will state it's for slice-local swaps primarily.
        // std::cout << "Info: rocsvSwapIndexBits currently expects one slice bit and one local bit for Alltoallv." << std::endl;
        // This could be ROCQ_STATUS_SUCCESS if local shuffle is implemented, or NOT_IMPLEMENTED.
        // For this stub, let's assume if no communication is needed, it's a success (NO-OP for this function's main purpose).
        // A dedicated local_swap_kernel would handle the both-local case.
        // The both-slice-bits case is more about re-interpreting which GPU has which part of the state.
        if (!is_q1_slice_bit) { // Both are local bits
            // std::cout << "Info: Swapping two local bits. Requires local permutation kernel (not implemented in this stub)." << std::endl;
            // TODO: Implement local permutation kernel for each GPU.
            // For each GPU i:
            //   hipSetDevice(h->deviceIds[i]);
            //   Launch local_bit_swap_kernel(h->d_local_state_slices[i], h->localStateSizes[i],
            //                                effective_local_q1, effective_local_q2, h->streams[i]);
            return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs local kernel
        } else { // Both are slice bits
            // std::cout << "Info: Swapping two slice-determining bits. This changes GPU rank mapping (not implemented in this stub)." << std::endl;
            return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs metadata update and potentially rank reordering logic.
        }
    }

    // --- Stage 1: Prepare send buffers and calculate send/recv counts & displacements ---
    // This stage involves complex GPU kernels. Outline:
    // For each GPU `r_src` from 0 to numGpus-1:
    //   hipSetDevice(h->deviceIds[r_src]);
    //   Launch a kernel (e.g., `prepare_alltoall_data_kernel`) on `h->streams[r_src]`:
    //     Input: `h->d_local_state_slices[r_src]`, `h->localStateSizes[r_src]`,
    //            `qubit_idx1`, `qubit_idx2`, `h->globalNumQubits`,
    //            `h->numLocalQubitsPerGpu`, `h->numGlobalSliceQubits`, `r_src`.
    //     Output: `h->d_swap_buffers[r_src]` (data packed for Alltoallv),
    //             `d_send_counts_for_this_gpu` (array of size numGpus, on device).
    //   The kernel logic:
    //     For each element at `old_local_idx` in `d_local_state_slices[r_src]`:
    //       1. `current_global_idx = reconstruct_global_idx_from_slice_info(r_src, old_local_idx, h->numLocalQubitsPerGpu)`.
    //       2. `swapped_global_idx = swap_bits_in_value(current_global_idx, qubit_idx1, qubit_idx2)`.
    //       3. `target_info = get_rank_and_local_idx_from_global(swapped_global_idx, h->numLocalQubitsPerGpu)`.
    //          `target_rank = target_info.first; new_local_idx = target_info.second;`
    //       4. This element is destined for `target_rank`. Store its value temporarily.
    //       5. Atomically increment `d_send_counts_for_this_gpu[target_rank]`.
    //     After all elements are processed for target ranks and counts:
    //       Calculate send displacements (`sdispls`) from `d_send_counts_for_this_gpu`.
    //       Re-iterate or use stored temporary data:
    //         Place element into `h->d_swap_buffers[r_src]` at `sdispls[target_rank] + (local_offset_within_target_block)`.
    //         The `local_offset_within_target_block` must ensure that amplitudes arriving at the
    //         target GPU are placed in an order that allows easy reconstruction or direct use.
    //         Typically, this means they should be ordered by their `new_local_idx`. This might
    //         require another sort or careful calculation of the write position.
    //
    //   After the kernel, copy `d_send_counts_for_this_gpu` from device to host for each GPU.
    //   Aggregate these into global `h_send_counts_all_gpus` and `h_recv_counts_all_gpus` tables.
    //   Calculate global send/recv displacements (`h_sdispls_all_gpus`, `h_rdispls_all_gpus`).
    //   Copy these per-GPU counts and displacements back to device memory for Alltoallv.

    // std::cout << "SwapIndexBits: Stage 1 (Data preparation kernel for Alltoallv) is required but not implemented." << std::endl;


    // --- Stage 2: RCCL All-to-All Communication ---
    // Example using rcclGroupStart/End. Ensure RCCL calls are on their respective streams.
    // rcclGroupStart();
    // for (int r = 0; r < h->numGpus; ++r) {
    //     hipSetDevice(h->deviceIds[r]);
    //     // d_scounts, d_sdispls, d_rcounts, d_rdispls are device pointers for rank r's view of comms.
    //     // These would have been prepared based on the global count/displacement calculation from Stage 1.
    //     rcclResult_t err = rcclAlltoallv(
    //         h->d_swap_buffers[r],       // Send buffer for this rank
    //         d_sCounts_r,                // Device array of send counts from rank r to all ranks
    //         d_sDispls_r,                // Device array of send displacements
    //         h->d_local_state_slices[r], // Receive buffer for this rank (data written here)
    //         d_rCounts_r,                // Device array of receive counts for rank r from all ranks
    //         d_rDispls_r,                // Device array of receive displacements
    //         rcclFloatComplex,           // Assuming rocComplex is hipFloatComplex, maps to rcclFloatComplex
    //         h->comms[r], 
    //         h->streams[r]
    //     );
    //     if (err != rcclSuccess) {
    //         // Potentially call rcclGroupAbort() or handle error then break.
    //         // For now, just return error.
    //         // rcclGroupEnd(); // Attempt to end group on error.
    //         return checkRcclError(err, "rocsvSwapIndexBits rcclAlltoallv");
    //     }
    // }
    // rcclResult_t group_err = rcclGroupEnd();
    // if (group_err != rcclSuccess) return checkRcclError(group_err, "rocsvSwapIndexBits rcclGroupEnd");
    // std::cout << "SwapIndexBits: Stage 2 (RCCL Alltoallv) is required but not implemented." << std::endl;


    // --- Stage 3: (Optional) Final local permutation if needed ---
    // If the Alltoallv and data preparation didn't place data in the final sorted order
    // (by new_local_idx on the target GPU), a final local permutation kernel would run here.
    // This kernel would use `h->d_swap_buffers[r]` as temporary storage if needed.
    // hipLaunchKernelGGL(final_local_sort_kernel, ..., h->d_local_state_slices[r], h->d_swap_buffers[r], ...);
    // std::cout << "SwapIndexBits: Stage 3 (Optional final local sort kernel) is required but not implemented." << std::endl;


    // --- Stage 4: Update qubit mapping metadata in handle (IMPORTANT) ---
    // The logical roles of qubit_idx1 and qubit_idx2 have swapped.
    // If the handle stores a dynamic map of global_qubit_idx_to_its_current_role (e.g. slice vs local bit position),
    // that map needs to be updated.
    // If the scheme is fixed (e.g. top N bits are always slice bits), then `globalNumQubits`,
    // `numLocalQubitsPerGpu`, `numGlobalSliceQubits` might not change their values, but the
    // *interpretation* of which original qubit is where has changed. Gate application logic
    // that uses these global indices must be aware.
    // This stub assumes the *meaning* of the fixed bit positions (slice vs local) doesn't change,
    // only the data location corresponding to the *original* global qubit indices.
    // So, no change to h->numGlobalSliceQubits etc. for this specific operation's scope.
    // The change is that data associated with global_idx `X` (where bit `qubit_idx1` was, say, 0)
    // is now at a new global_idx `Y` (where bit `qubit_idx1` is now, say, 1, and bit `qubit_idx2` changed too).

    // std::cout << "Info: rocsvSwapIndexBits data redistribution complete (conceptually)." << std::endl;
    
    return ROCQ_STATUS_NOT_IMPLEMENTED; // Due to missing GPU kernels and detailed comms logic.
}


// Helper function (conceptual) to reconstruct global index from rank and local index
// This depends on the specific mapping strategy (e.g., top bits for rank).
// For this example, assume top numGlobalSliceQubits determine rank.
static inline size_t reconstruct_global_idx(int rank, size_t local_idx, 
                                            unsigned num_local_qubits_on_gpu, 
                                            unsigned num_global_slice_qubits) {
    // Example: if num_global_slice_qubits determines the rank, and local_idx are the lower bits.
    // This assumes local_idx corresponds to states after slice bits are factored out.
    // global_idx = (static_cast<size_t>(rank) << num_local_qubits_on_gpu) | local_idx;
    // This is a common way: rank bits are MSBs of global index portion determining distribution.
    // Let global_qubits = num_local_qubits_on_gpu + num_global_slice_qubits
    // slice_mask = (1ULL << num_global_slice_qubits) -1 ; shifted to top
    // rank effectively is (global_idx_full >> num_local_qubits_on_gpu) & slice_mask_for_rank_indices
    // local_idx maps to the (2^num_local_qubits_on_gpu) states.
    // A common convention is that the global state vector index `G` is split such that
    // G = rank * (2^L) + local_idx, where L = numLocalQubitsPerGpu.
    // Or, G = (local_idx << S) | rank, if rank bits are the lowest S bits.
    // Or, G's top S bits are rank, bottom L bits are local_idx. G = (rank << L) | local_idx. This is typical.
    return (static_cast<size_t>(rank) << num_local_qubits_on_gpu) | local_idx;
}

// Helper function (conceptual) to swap bits in a global index
static inline size_t swap_global_idx_bits(size_t global_idx, unsigned bit_a, unsigned bit_b) {
    unsigned val_a = (global_idx >> bit_a) & 1;
    unsigned val_b = (global_idx >> bit_b) & 1;
    if (val_a != val_b) {
        global_idx ^= (1ULL << bit_a);
        global_idx ^= (1ULL << bit_b);
    }
    return global_idx;
}

// Helper function (conceptual) to get target rank and new local index from a swapped global index
static inline std::pair<int, size_t> get_target_rank_and_local_idx_from_swapped(
    size_t swapped_global_idx, 
    unsigned num_local_qubits_on_gpu,
    unsigned num_global_slice_qubits // Total number of bits determining rank
    ) {
    // Based on the convention: G = (rank << L) | local_idx
    int target_rank = static_cast<int>(swapped_global_idx >> num_local_qubits_on_gpu);
    size_t target_local_idx_mask = (1ULL << num_local_qubits_on_gpu) - 1;
    size_t target_local_idx = swapped_global_idx & target_local_idx_mask;
    return {target_rank, target_local_idx};
}


// Kernel for preparing data for Alltoall communication (conceptual)
// This kernel would run on each GPU.
// It calculates where each local amplitude should go after the bit swap.
// It then shuffles the data from d_local_slice into d_swap_buffer,
// arranging it into contiguous blocks, one for each target GPU.
// It also calculates send_counts: how many amplitudes are being sent to each target GPU.
//
// __global__ void prepare_swap_buffers_kernel(
//     rocComplex* d_local_slice,      // Input: current GPU's slice of the state vector
//     rocComplex* d_swap_buffer,      // Output: data shuffled and packed for Alltoall
//     int* d_send_counts,             // Output: array of size numGpus, d_send_counts[target_gpu] = count
//     size_t local_slice_size,        // Input: number of elements in d_local_slice
//     unsigned qubit_idx1,            // Input: first global qubit index to swap
//     unsigned qubit_idx2,            // Input: second global qubit index to swap
//     unsigned globalNumQubits,       // Input: total qubits in simulation
//     unsigned numLocalQubitsPerGpu,  // Input: number of local qubits per GPU slice
//     unsigned numGlobalSliceQubits,  // Input: number of bits determining the GPU slice/rank
//     int current_rank,               // Input: rank of the current GPU
//     int numGpus                     // Input: total number of GPUs
//     // Potentially also pass precomputed offsets for writing into d_swap_buffer (derived from a prefix sum of send_counts)
// ) {
//     size_t current_local_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (current_local_idx >= local_slice_size) return;

//     // 1. Reconstruct global index
//     size_t global_idx = reconstruct_global_idx(current_rank, current_local_idx, numLocalQubitsPerGpu, numGlobalSliceQubits);
    
//     // 2. Swap bits in global index
//     size_t swapped_global_idx = swap_global_idx_bits(global_idx, qubit_idx1, qubit_idx2);

//     // 3. Determine target rank and new local index from swapped global index
//     std::pair<int, size_t> target_info = get_target_rank_and_local_idx_from_swapped(
//         swapped_global_idx, numLocalQubitsPerGpu, numGlobalSliceQubits
//     );
//     int target_rank = target_info.first;
//     size_t new_local_idx_on_target = target_info.second; // This new_local_idx is crucial for final placement.

//     // 4. Atomically increment send_count for the target_rank.
//     //    This gives us how many elements are going to each GPU.
//     //    This needs to be done carefully. A common pattern is:
//     //    a) Each thread calculates its target_rank.
//     //    b) A separate reduction/scan computes send_counts per target_rank.
//     //    c) This can also be done by having each thread atomicAdd to d_send_counts[target_rank].
//     // For this conceptual kernel, let's assume d_send_counts are pre-calculated by a prior pass or atomicAdds.
//     // (atomicInc(&d_send_counts[target_rank], numGpus-1) or similar if it's an array of counters)
//     // Actually, d_send_counts is calculated on host or via reduction after this kernel pass for target_rank.
//     // This kernel's first job is to determine target_rank for each element.
//     // Let's assume a different kernel `find_targets_and_counts_kernel` populates `d_target_ranks_per_element` and `d_send_counts`.
//     // Then, this kernel `prepare_swap_buffers_kernel` uses those.

//     // 5. Calculate write position in d_swap_buffer.
//     //    This requires knowing the offset for `target_rank` data, and the `new_local_idx_on_target`
//     //    is NOT directly the offset within that block if Alltoallv is used.
//     //    Alltoallv means we pack elements destined for GPU_X contiguously.
//     //    So, we need to know, for this `current_local_idx`, how many other elements from *this source GPU*
//     //    are also going to `target_rank` AND come *before* this element in some ordering (e.g. original local_idx order).
//     //
//     //    This implies a local scan/prefix sum on counts of elements going to each target_rank,
//     //    or a more complex addressing scheme.
//     //
//     //    Simplified approach:
//     //    - Assume `d_send_offsets_for_target_ranks` (size numGpus) is precomputed on host/device (prefix sum of d_send_counts).
//     //    - Assume `d_within_target_rank_offsets` (size local_slice_size) is precomputed, storing the
//     //      local serial number of this element among all elements from current_rank going to target_rank.
//     //    size_t write_offset_in_swap_buffer = d_send_offsets_for_target_ranks[target_rank] + d_within_target_rank_offsets[current_local_idx];
//     //    d_swap_buffer[write_offset_in_swap_buffer] = d_local_slice[current_local_idx];

//     // Given tool limits, we cannot fully implement this kernel.
//     // The key is that d_swap_buffer must be filled such that all data for GPU 0 is first,
//     // then all data for GPU 1, and so on. The order within these blocks should correspond
//     // to the final order expected on the target GPU (i.e., by new_local_idx_on_target).
//     // This usually means sorting or a careful scatter operation.
// }


rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle,
                                unsigned qubit_idx1,
                                unsigned qubit_idx2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);

    if (qubit_idx1 == qubit_idx2) return ROCQ_STATUS_SUCCESS; 

    if (h->globalNumQubits == 0) return ROCQ_STATUS_INVALID_VALUE; // State not allocated
    if (qubit_idx1 >= h->globalNumQubits || qubit_idx2 >= h->globalNumQubits) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    if (h->numGpus <= 1) return ROCQ_STATUS_SUCCESS; // No communication for single GPU or uninitialized handle

    // Further checks: ensure d_local_state_slices and d_swap_buffers are allocated
    if (h->d_local_state_slices.empty() || h->d_swap_buffers.empty() || 
        h->d_local_state_slices[0] == nullptr || h->d_swap_buffers[0] == nullptr) { // Basic check
        return ROCQ_STATUS_FAILURE; // Not properly allocated
    }
    
    // TODO: Add logic to check if the swap is between a slice-determining bit and a local bit.
    // If both bits are local to all GPUs (i.e., not among numGlobalSliceQubits),
    // then only a local permutation kernel is needed on each GPU, no RCCL communication.
    // If both bits are slice-determining bits, this changes ranks of entire slices,
    // also complex communication.
    // This function is primarily designed for swapping one slice bit with one local bit.
    // For now, we proceed assuming communication is necessary.

    // --- Stage 1: Prepare send buffers and calculate send/recv counts & displacements ---
    // This stage is highly complex and typically involves one or more custom GPU kernels.
    // Each GPU (rank `r_src`):
    // 1. Iterates through its `h->d_local_state_slices[r_src]`.
    // 2. For each amplitude at `old_local_idx`:
    //    a. Reconstructs its `current_global_idx`.
    //    b. Calculates the `swapped_global_idx` by permuting `qubit_idx1` and `qubit_idx2`.
    //    c. Determines the `target_rank` and `new_local_idx_on_target` from `swapped_global_idx`.
    // 3. Populates `h->d_swap_buffers[r_src]` by scattering its local amplitudes according
    //    to their `target_rank` and `new_local_idx_on_target`. Data for each target rank
    //    must be contiguous. This requires knowing how many elements go to each target rank
    //    (send_counts) and their displacements in the swap buffer.
    // 4. Calculates `send_counts_for_gpu_r_src[target_rank]` (number of amplitudes r_src sends to target_rank).
    //
    // This usually requires:
    //    - A kernel to determine target_rank and new_local_idx for each amplitude.
    //    - A device-wide sort or scan operation (e.g., using rocThrust or custom kernels) to calculate
    //      send_counts and displacements for packing data into the swap buffer correctly.
    //    - A scatter kernel to write data from local_slice to swap_buffer in packed format.

    // Placeholder for Stage 1:
    // For each GPU r_src from 0 to numGpus-1:
    //   hipSetDevice(h->deviceIds[r_src]);
    //   hipLaunchKernelGGL(prepare_swap_buffers_kernel, ...,
    //                      h->d_local_state_slices[r_src], h->d_swap_buffers[r_src],
    //                      d_send_counts_for_r_src, h->localStateSizes[r_src],
    //                      qubit_idx1, qubit_idx2, h->globalNumQubits,
    //                      h->numLocalQubitsPerGpu, h->numGlobalSliceQubits,
    //                      r_src, h->numGpus);
    //   // After kernel, copy d_send_counts_for_r_src to host to build global communication plan for Alltoallv
    // End For
    // Then, calculate recv_counts and displacements for all GPUs.
    // The recv_counts for GPU `i` from GPU `j` is send_counts for GPU `j` to GPU `i`.

    // std::cout << "SwapIndexBits: Data preparation kernel and count calculation needed here." << std::endl;
    // std::cout << "This involves complex GPU-side logic for data shuffling into send buffers." << std::endl;


    // --- Stage 2: RCCL All-to-All Communication ---
    // `rcclAlltoallv` is the most suitable primitive.
    // Each GPU `r` will call:
    // rcclAlltoallv(h->d_swap_buffers[r],      /* device ptr to send_counts_for_gpu_r */,
    //               /* device ptr to sdispls_for_gpu_r */,
    //               h->d_local_state_slices[r], /* device ptr to recv_counts_for_gpu_r */,
    //               /* device ptr to rdispls_for_gpu_r */,
    //               RCCL_HIP_FLOAT_COMPLEX, // Assuming this is the RCCL data type for rocComplex
    //               h->comms[r], h->streams[r]);
    //
    // This call happens inside a rcclGroupStart/End block if performed individually per rank.
    // Or, if rcclAlltoallv itself is not group-safe for multiple distinct communicators on the same process
    // (which it should be, as comms[r] is specific), then direct calls are fine.

    // Placeholder for Stage 2:
    // rcclGroupStart();
    // for (int r = 0; r < h->numGpus; ++r) {
    //     hipSetDevice(h->deviceIds[r]);
    //     // Populate d_send_counts_r, d_sdispls_r, d_recv_counts_r, d_rdispls_r on device r
    //     // For example, send_counts_for_gpu_r[k] is how much r sends to k.
    //     // recv_counts_for_gpu_r[k] is how much r receives from k.
    //     rcclResult_t err = rcclAlltoallv(
    //         h->d_swap_buffers[r],         /* send data buffer (already prepared) */
    //         d_send_counts_r_gpu,          /* array of send counts from this rank r to all other ranks */
    //         d_sdispls_r_gpu,              /* array of send displacements */
    //         h->d_local_state_slices[r],   /* receive data buffer */
    //         d_recv_counts_r_gpu,          /* array of receive counts by this rank r from all other ranks */
    //         d_rdispls_r_gpu,              /* array of receive displacements */
    //         rcclFloatComplex,             // Or the appropriate RCCL type for rocComplex
    //         h->comms[r], 
    //         h->streams[r]
    //     );
    //     if (err != rcclSuccess) { /* handle error, potentially abort group */ }
    // }
    // rcclGroupEnd();
    // std::cout << "SwapIndexBits: RCCL Alltoallv communication needed here." << std::endl;

    // --- Stage 3: (Optional) Final local permutation ---
    // After Alltoallv, data in h->d_local_state_slices[r] is now on the correct GPU,
    // but might be grouped by which source GPU it came from.
    // If the `prepare_swap_buffers_kernel` and Alltoallv parameters were set up perfectly,
    // the data might already be in the final correct local order (ordered by `new_local_idx_on_target`).
    // If not, a final local shuffle kernel would be needed on each GPU:
    // hipLaunchKernelGGL(permute_received_data_to_final_order_kernel, ...,
    //                    h->d_local_state_slices[r], h->d_swap_buffers[r] /*as temp*/, ...);
    // std::cout << "SwapIndexBits: Optional final local permutation kernel needed here." << std::endl;


    // --- Stage 4: Update qubit mapping metadata in handle (if necessary) ---
    // If the swap operation changes which global qubit indices are considered "slice-determining"
    // versus "local", then `h->numGlobalSliceQubits` and `h->numLocalQubitsPerGpu`
    // (or a more flexible structure storing the actual indices) would need to be updated.
    // For this example, we assume a fixed mapping initially, and this function
    // just permutes data according to that fixed mapping. A more advanced system
    // might update the mapping itself.
    // E.g. if qubit_idx1 was a slice bit and qubit_idx2 was a local bit:
    // The bit previously at qubit_idx1 (now at qubit_idx2's old position) might become local.
    // The bit previously at qubit_idx2 (now at qubit_idx1's old position) might become a slice bit.
    // This means the logical interpretation of the distributed state changes.
    // The handle's `numGlobalSliceQubits` and `numLocalQubitsPerGpu` might need adjustment,
    // or better, the specific set of global indices that form the "slice index" needs to be updated.
    // For simplicity, this current stub does not modify these, assuming they describe the
    // *current* fixed distribution scheme. The data is permuted *within* this scheme.

    return ROCQ_STATUS_NOT_IMPLEMENTED;
}
