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
};

// Helper to check HIP errors and convert to rocqStatus_t
rocqStatus_t checkHipError(hipError_t err, const char* operation = "") {
    if (err != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

// Helper to check rocBLAS errors and convert to rocqStatus_t
rocqStatus_t checkRocblasError(rocblas_status err, const char* operation = "") {
    if (err != rocblas_status_success) {
        return ROCQ_STATUS_FAILURE;
    }
    return ROCQ_STATUS_SUCCESS;
}

// Helper to check RCCL errors and convert to rocqStatus_t
rocqStatus_t checkRcclError(rcclResult_t err, const char* operation = "") {
    if (err != rcclSuccess) {
        return ROCQ_STATUS_RCCL_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

static bool are_qubits_local(rocsvInternalHandle* h, const unsigned* qubitIndices, unsigned numTargetQubits) {
    if (!h || h->numGpus == 0) {
        return false; 
    }
    if (h->numGpus == 1) {
        return true;
    }
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

// Measurement kernels
__global__ void collapse_state_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, int measuredOutcome);
__global__ void renormalize_state_kernel(rocComplex* state, unsigned numQubits, double d_sum_sq_mag_inv_sqrt);

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

__global__ void reduce_block_sums_to_slice_total_probs_kernel(
    const double* d_block_partial_probs,
    unsigned num_blocks_from_previous_kernel,
    double* d_slice_total_probs_out);

__global__ void reduce_block_sums_to_slice_total_sum_sq_mag_kernel(
    const double* d_block_sum_sq_mag_in,
    unsigned num_blocks_from_previous_kernel,
    double* d_slice_total_sum_sq_mag_out);


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
    internal_handle->localRank = -1;
    internal_handle->localStateSize = 0;
    internal_handle->numGpus = 0;
    internal_handle->globalNumQubits = 0;
    internal_handle->numLocalQubitsPerGpu = 0;
    internal_handle->numGlobalSliceQubits = 0;

    hipError_t hip_err;
    rocblas_status blas_err;
    rcclResult_t rccl_err_unused;
    int device_count = 0;
    hip_err = hipGetDeviceCount(&device_count);
    if (hip_err != hipSuccess) { delete internal_handle; return checkHipError(hip_err, "rocsvCreate hipGetDeviceCount"); }
    if (device_count <= 0) { delete internal_handle; return ROCQ_STATUS_FAILURE; }
    internal_handle->numGpus = device_count;

    try {
        internal_handle->deviceIds.resize(internal_handle->numGpus);
        internal_handle->streams.resize(internal_handle->numGpus);
        internal_handle->blasHandles.resize(internal_handle->numGpus);
        internal_handle->comms.resize(internal_handle->numGpus);
        internal_handle->d_local_state_slices.resize(internal_handle->numGpus, nullptr);
        internal_handle->localStateSizes.resize(internal_handle->numGpus, 0);
        internal_handle->d_swap_buffers.resize(internal_handle->numGpus, nullptr);
    } catch (const std::bad_alloc& e) { delete internal_handle; return ROCQ_STATUS_ALLOCATION_FAILED; }

    rcclUniqueId uniqueId;
    if (internal_handle->numGpus > 0) {
        if (rcclGetUniqueId(&uniqueId) != rcclSuccess) { delete internal_handle; return ROCQ_STATUS_RCCL_ERROR; }
    }

    for (int i = 0; i < internal_handle->numGpus; ++i) {
        internal_handle->deviceIds[i] = i;
        hip_err = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err != hipSuccess) { /* cleanup */ delete internal_handle; return checkHipError(hip_err, "rocsvCreate hipSetDevice");}
        hip_err = hipStreamCreate(&internal_handle->streams[i]);
        if (hip_err != hipSuccess) { /* cleanup */ delete internal_handle; return checkHipError(hip_err, "rocsvCreate hipStreamCreate");}
        blas_err = rocblas_create_handle(&internal_handle->blasHandles[i]);
        if (blas_err != rocblas_status_success) { /* cleanup */ delete internal_handle; return checkRocblasError(blas_err, "rocsvCreate rocblas_create_handle");}
        blas_err = rocblas_set_stream(internal_handle->blasHandles[i], internal_handle->streams[i]);
        if (blas_err != rocblas_status_success) { /* cleanup */ delete internal_handle; return checkRocblasError(blas_err, "rocsvCreate rocblas_set_stream");}
        rccl_err_unused = rcclCommInitRank(&internal_handle->comms[i], internal_handle->numGpus, uniqueId, i);
        if (rccl_err_unused != rcclSuccess) { /* cleanup */ delete internal_handle; return checkRcclError(rccl_err_unused, "rocsvCreate rcclCommInitRank");}
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
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    rocqStatus_t first_error_status = ROCQ_STATUS_SUCCESS;
    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_destroy = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_destroy != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) first_error_status = ROCQ_STATUS_HIP_ERROR;
        if (internal_handle->comms[i]) if (rcclCommDestroy(internal_handle->comms[i]) != rcclSuccess && first_error_status == ROCQ_STATUS_SUCCESS) first_error_status = ROCQ_STATUS_RCCL_ERROR;
        if (internal_handle->blasHandles[i]) if (rocblas_destroy_handle(internal_handle->blasHandles[i]) != rocblas_status_success && first_error_status == ROCQ_STATUS_SUCCESS) first_error_status = ROCQ_STATUS_FAILURE;
        if (internal_handle->streams[i]) if (hipStreamDestroy(internal_handle->streams[i]) != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) first_error_status = ROCQ_STATUS_HIP_ERROR;
        if (internal_handle->d_local_state_slices[i]) if (hipFree(internal_handle->d_local_state_slices[i]) != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) first_error_status = ROCQ_STATUS_HIP_ERROR;
        internal_handle->d_local_state_slices[i] = nullptr;
        if (internal_handle->d_swap_buffers[i]) if (hipFree(internal_handle->d_swap_buffers[i]) != hipSuccess && first_error_status == ROCQ_STATUS_SUCCESS) first_error_status = ROCQ_STATUS_HIP_ERROR;
        internal_handle->d_swap_buffers[i] = nullptr;
    }
    internal_handle->deviceIds.clear(); internal_handle->streams.clear(); internal_handle->blasHandles.clear(); internal_handle->comms.clear();
    internal_handle->d_local_state_slices.clear(); internal_handle->localStateSizes.clear(); internal_handle->d_swap_buffers.clear();
    delete internal_handle;
    return first_error_status;
}

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle, unsigned numQubits, rocComplex** d_state) {
    if (!handle || !d_state || numQubits > 60) {
        if (numQubits == 0 && d_state == nullptr) return ROCQ_STATUS_INVALID_VALUE;
        if (numQubits > 60) return ROCQ_STATUS_INVALID_VALUE;
    }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    if (internal_handle->numGpus == 0 || internal_handle->deviceIds.empty() || internal_handle->streams.empty()) return ROCQ_STATUS_FAILURE;
    hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[0]);
    if (hip_err_set != hipSuccess) return checkHipError(hip_err_set, "rocsvAllocateState hipSetDevice");
    size_t num_elements = 1ULL << numQubits;
    size_t size_bytes = num_elements * sizeof(rocComplex);
    if (internal_handle->d_local_state_slices[0] != nullptr) { hipFree(internal_handle->d_local_state_slices[0]); internal_handle->d_local_state_slices[0] = nullptr; }
    if (internal_handle->d_swap_buffers[0] != nullptr) { hipFree(internal_handle->d_swap_buffers[0]); internal_handle->d_swap_buffers[0] = nullptr; }
    hipError_t err = hipMalloc(&internal_handle->d_local_state_slices[0], size_bytes);
    if (err != hipSuccess) { *d_state = nullptr; internal_handle->d_local_state_slices[0] = nullptr; return checkHipError(err, "rocsvAllocateState hipMalloc"); }
    hipError_t err_swap = hipMalloc(&internal_handle->d_swap_buffers[0], size_bytes);
    if (err_swap != hipSuccess) { hipFree(internal_handle->d_local_state_slices[0]); internal_handle->d_local_state_slices[0] = nullptr; *d_state = nullptr; return checkHipError(err_swap, "rocsvAllocateState hipMalloc swap_buffer"); }
    internal_handle->localStateSizes[0] = num_elements; *d_state = internal_handle->d_local_state_slices[0];
    internal_handle->globalNumQubits = numQubits; internal_handle->numGlobalSliceQubits = 0; internal_handle->numLocalQubitsPerGpu = numQubits; internal_handle->localStateSize = num_elements;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle, unsigned totalNumQubits) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    if (internal_handle->numGpus == 0) return ROCQ_STATUS_FAILURE;
    if ((internal_handle->numGpus > 1) && ((internal_handle->numGpus & (internal_handle->numGpus - 1)) != 0)) return ROCQ_STATUS_INVALID_VALUE;
    if (totalNumQubits == 0 && internal_handle->numGpus > 1) return ROCQ_STATUS_INVALID_VALUE;
    if (totalNumQubits > 60) return ROCQ_STATUS_INVALID_VALUE;
    unsigned num_global_slice_qubits = 0;
    if (internal_handle->numGpus > 1) num_global_slice_qubits = static_cast<unsigned>(std::log2(internal_handle->numGpus));
    if (totalNumQubits < num_global_slice_qubits && internal_handle->numGpus > 1) return ROCQ_STATUS_INVALID_VALUE;
    internal_handle->globalNumQubits = totalNumQubits; internal_handle->numGlobalSliceQubits = num_global_slice_qubits;
    internal_handle->numLocalQubitsPerGpu = (totalNumQubits >= num_global_slice_qubits) ? (totalNumQubits - num_global_slice_qubits) : 0;
    if (internal_handle->numGpus == 1) { internal_handle->numLocalQubitsPerGpu = totalNumQubits; internal_handle->numGlobalSliceQubits = 0; }
    size_t sliceNumElements = 1ULL << internal_handle->numLocalQubitsPerGpu;
    size_t sliceSizeBytes = sliceNumElements * sizeof(rocComplex);
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvAllocateDistributedState hipSetDevice"); /* cleanup */ return status; }
        if (internal_handle->d_local_state_slices[i] != nullptr) { hipFree(internal_handle->d_local_state_slices[i]); internal_handle->d_local_state_slices[i] = nullptr; }
        if (internal_handle->d_swap_buffers[i] != nullptr) { hipFree(internal_handle->d_swap_buffers[i]); internal_handle->d_swap_buffers[i] = nullptr; }
        if (sliceSizeBytes > 0) {
            hipError_t err_alloc = hipMalloc(&internal_handle->d_local_state_slices[i], sliceSizeBytes);
            if (err_alloc != hipSuccess) { status = checkHipError(err_alloc, "rocsvAllocateDistributedState hipMalloc slice"); /* cleanup */ return status;}
            internal_handle->localStateSizes[i] = sliceNumElements;
            hipError_t err_alloc_swap = hipMalloc(&internal_handle->d_swap_buffers[i], sliceSizeBytes);
            if (err_alloc_swap != hipSuccess) { hipFree(internal_handle->d_local_state_slices[i]); internal_handle->d_local_state_slices[i] = nullptr; status = checkHipError(err_alloc_swap, "rocsvAllocateDistributedState hipMalloc swap_buffer"); /* cleanup */ return status;}
        } else { internal_handle->d_local_state_slices[i] = nullptr; internal_handle->d_swap_buffers[i] = nullptr; internal_handle->localStateSizes[i] = 0;}
    }
    if (internal_handle->numGpus == 1) internal_handle->localStateSize = sliceNumElements;
    return status;
}

rocqStatus_t rocsvInitializeState(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits) {
    if (!handle || !d_state || numQubits > 60 ) { if (numQubits == 0 && d_state == nullptr) return ROCQ_STATUS_INVALID_VALUE; if (numQubits > 60) return ROCQ_STATUS_INVALID_VALUE; }
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    if (internal_handle->numGpus == 0 || internal_handle->streams.empty() || internal_handle->d_local_state_slices.empty() || internal_handle->d_local_state_slices[0] != d_state) return ROCQ_STATUS_INVALID_VALUE;
    hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[0]);
    if (hip_err_set != hipSuccess) return checkHipError(hip_err_set, "rocsvInitializeState hipSetDevice");
    size_t num_elements = 1ULL << numQubits;
    if (internal_handle->localStateSizes[0] != num_elements) { if (!(numQubits == 0 && internal_handle->localStateSizes[0] == 1 && num_elements == 1)) return ROCQ_STATUS_INVALID_VALUE; }
    if (num_elements == 0 && numQubits > 0) return ROCQ_STATUS_INVALID_VALUE;
    hipError_t err = hipMemsetAsync(internal_handle->d_local_state_slices[0], 0, num_elements * sizeof(rocComplex), internal_handle->streams[0]);
    if (err != hipSuccess) return checkHipError(err, "rocsvInitializeState hipMemsetAsync");
    if (num_elements > 0) { 
        rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
        err = hipMemcpyAsync(internal_handle->d_local_state_slices[0], &zero_state_amplitude, sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->streams[0]);
        if (err != hipSuccess) return checkHipError(err, "rocsvInitializeState hipMemcpyAsync for first element");
    }
    err = hipStreamSynchronize(internal_handle->streams[0]);
    return checkHipError(err, "rocsvInitializeState hipStreamSynchronize");
}

rocqStatus_t rocsvInitializeDistributedState(rocsvHandle_t handle) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* internal_handle = static_cast<rocsvInternalHandle*>(handle);
    bool is_zero_qubit_single_gpu_case = (internal_handle->globalNumQubits == 0 && internal_handle->numGpus == 1 && internal_handle->numLocalQubitsPerGpu == 0);
    if (internal_handle->numGpus == 0 || (internal_handle->globalNumQubits == 0 && !is_zero_qubit_single_gpu_case)) return ROCQ_STATUS_FAILURE;
    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_set != hipSuccess) return checkHipError(hip_err_set, "rocsvInitializeDistributedState hipSetDevice");
        if (internal_handle->localStateSizes[i] > 0 && internal_handle->d_local_state_slices[i] == nullptr ) return ROCQ_STATUS_INVALID_VALUE;
        if (internal_handle->localStateSizes[i] > 0) {
            hipError_t hip_err_memset = hipMemsetAsync(internal_handle->d_local_state_slices[i], 0, internal_handle->localStateSizes[i] * sizeof(rocComplex), internal_handle->streams[i]);
            if (hip_err_memset != hipSuccess) return checkHipError(hip_err_memset, "rocsvInitializeDistributedState hipMemsetAsync");
        }
    }
    if (internal_handle->numGpus > 0) {
        hipError_t hip_err_set_dev0 = hipSetDevice(internal_handle->deviceIds[0]);
        if (hip_err_set_dev0 != hipSuccess) return checkHipError(hip_err_set_dev0, "rocsvInitializeDistributedState hipSetDevice for dev0");
        if (internal_handle->d_local_state_slices[0] != nullptr && internal_handle->localStateSizes[0] > 0) {
            rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
            hipError_t hip_err_memcpy = hipMemcpyAsync(internal_handle->d_local_state_slices[0], &zero_state_amplitude, sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->streams[0]);
            if (hip_err_memcpy != hipSuccess) return checkHipError(hip_err_memcpy, "rocsvInitializeDistributedState hipMemcpyAsync for zero state");
        } else if (is_zero_qubit_single_gpu_case && internal_handle->d_local_state_slices[0] != nullptr && internal_handle->localStateSizes[0] == 1) {
             rocComplex zero_state_amplitude = make_hipFloatComplex(1.0f, 0.0f);
             hipError_t hip_err_memcpy = hipMemcpyAsync(internal_handle->d_local_state_slices[0], &zero_state_amplitude, sizeof(rocComplex), hipMemcpyHostToDevice, internal_handle->streams[0]);
            if (hip_err_memcpy != hipSuccess) return checkHipError(hip_err_memcpy, "rocsvInitializeDistributedState hipMemcpyAsync for 0-qubit state");
        }
    }
    for (int i = 0; i < internal_handle->numGpus; ++i) {
        hipError_t hip_err_set = hipSetDevice(internal_handle->deviceIds[i]);
        if (hip_err_set != hipSuccess) status = checkHipError(hip_err_set, "rocsvInitializeDistributedState hipSetDevice for sync");
        hipError_t hip_err_sync = hipStreamSynchronize(internal_handle->streams[i]);
        if (hip_err_sync != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = checkHipError(hip_err_sync, "rocsvInitializeDistributedState hipStreamSynchronize");
    }
    return status;
}

rocqStatus_t rocsvApplyMatrix(rocsvHandle_t handle, rocComplex* d_state_legacy, unsigned globalNumQubits_param, const unsigned* qubitIndices, unsigned numTargetQubits, const rocComplex* matrixDevice, unsigned matrixDim) {
    if (!handle || !qubitIndices || !matrixDevice) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0 || h->d_local_state_slices.empty()) return ROCQ_STATUS_FAILURE;
    unsigned currentGlobalNumQubits = (h->globalNumQubits > 0) ? h->globalNumQubits : globalNumQubits_param;
    if (currentGlobalNumQubits == 0 && numTargetQubits > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (currentGlobalNumQubits == 0 && numTargetQubits == 0 && matrixDim == 1) return ROCQ_STATUS_SUCCESS;
    if (numTargetQubits == 0 && matrixDim == 1) return ROCQ_STATUS_SUCCESS;
    if (numTargetQubits == 0 && matrixDim != 1) return ROCQ_STATUS_INVALID_VALUE;
    if (matrixDim != (1U << numTargetQubits)) return ROCQ_STATUS_INVALID_VALUE;
    for (unsigned i = 0; i < numTargetQubits; ++i) {
        if (qubitIndices[i] >= currentGlobalNumQubits) return ROCQ_STATUS_INVALID_VALUE;
        for (unsigned j = i + 1; j < numTargetQubits; ++j) if (qubitIndices[i] == qubitIndices[j]) return ROCQ_STATUS_INVALID_VALUE;
    }
    hipError_t hip_err; rocqStatus_t status = ROCQ_STATUS_SUCCESS; unsigned threads_per_block = 256;
    if (are_qubits_local(h, qubitIndices, numTargetQubits)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        std::vector<unsigned> local_qubit_indices_vec(qubitIndices, qubitIndices + numTargetQubits);
        for (int rank = 0; rank < h->numGpus; ++rank) {
            hip_err = hipSetDevice(h->deviceIds[rank]);
            if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocsvApplyMatrix hipSetDevice (Local)"); break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0 ) { status = ROCQ_STATUS_INVALID_VALUE; break; }
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
                else if (num_thread_groups == 0) { if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0; else num_blocks = 0;}
                if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                    hipLaunchKernelGGL(apply_single_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], current_local_slice_ptr, local_num_qubits_for_kernel, targetQubitLocal, matrixDevice);
                    if (hipGetLastError() != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; break;}
                }
            } else if (numTargetQubits == 2) {
                unsigned q0_local = local_qubit_indices_vec[0]; unsigned q1_local = local_qubit_indices_vec[1];
                size_t num_thread_groups = (local_num_qubits_for_kernel >=2) ? local_slice_num_elements / 4 : 0;
                unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
                if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1; else if (num_thread_groups == 0) num_blocks = 0;
                if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                    hipLaunchKernelGGL(apply_two_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], current_local_slice_ptr, local_num_qubits_for_kernel, q0_local, q1_local, matrixDevice);
                    if (hipGetLastError() != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; break;}
                }
            } else if (numTargetQubits == 3 || numTargetQubits == 4) {
                hipMalloc(&d_target_indices_gpu, numTargetQubits * sizeof(unsigned)); hipMemcpyAsync(d_target_indices_gpu, local_qubit_indices_vec.data(), numTargetQubits * sizeof(unsigned), hipMemcpyHostToDevice, h->streams[rank]);
                size_t m_val = numTargetQubits; size_t num_kernel_threads = (local_num_qubits_for_kernel < m_val) ? 0 : (local_slice_num_elements >> m_val);
                unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
                if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1; if (num_kernel_threads == 0) num_blocks = 0;
                if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                    if (numTargetQubits == 3) hipLaunchKernelGGL(apply_three_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], current_local_slice_ptr, local_num_qubits_for_kernel, d_target_indices_gpu, matrixDevice);
                    else hipLaunchKernelGGL(apply_four_qubit_generic_matrix_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank], current_local_slice_ptr, local_num_qubits_for_kernel, d_target_indices_gpu, matrixDevice);
                    if (hipGetLastError() != hipSuccess) status = ROCQ_STATUS_HIP_ERROR;
                }
                hipStreamSynchronize(h->streams[rank]); hipFree(d_target_indices_gpu); if(status != ROCQ_STATUS_SUCCESS) break;
            } else if (numTargetQubits >= 5) { 
                const unsigned MAX_M = 10; if (numTargetQubits > MAX_M) { status = ROCQ_STATUS_NOT_IMPLEMENTED; break; }
                if (h->localStateSizes[rank] == 0) continue;
                unsigned m = numTargetQubits; std::vector<unsigned> h_sorted_idx(local_qubit_indices_vec); std::sort(h_sorted_idx.begin(), h_sorted_idx.end());
                unsigned* d_idx_gs = nullptr; rocComplex* d_tmp_in = nullptr; rocComplex* d_tmp_out = nullptr; bool gs_err = false;
                if(hipMalloc(&d_idx_gs,m*sizeof(unsigned))!=hipSuccess){status=ROCQ_STATUS_ALLOCATION_FAILED;gs_err=true;}
                else if(hipMemcpyAsync(d_idx_gs,h_sorted_idx.data(),m*sizeof(unsigned),hipMemcpyHostToDevice,h->streams[rank])!=hipSuccess){status=ROCQ_STATUS_HIP_ERROR;gs_err=true;}
                if(!gs_err && hipMalloc(&d_tmp_in,matrixDim*sizeof(rocComplex))!=hipSuccess){status=ROCQ_STATUS_ALLOCATION_FAILED;gs_err=true;}
                if(!gs_err && hipMalloc(&d_tmp_out,matrixDim*sizeof(rocComplex))!=hipSuccess){status=ROCQ_STATUS_ALLOCATION_FAILED;gs_err=true;}
                if(gs_err){if(d_idx_gs)hipFree(d_idx_gs);if(d_tmp_in)hipFree(d_tmp_in);if(d_tmp_out)hipFree(d_tmp_out);break;}
                rocblas_float_complex alpha={1,0}, beta={0,0}; unsigned n_non_tgt_q=local_num_qubits_for_kernel-m; size_t n_non_tgt_cfg=1ULL<<n_non_tgt_q; if(local_num_qubits_for_kernel<m)n_non_tgt_cfg=0;
                unsigned gs_tpb=256; if(matrixDim<gs_tpb&&matrixDim>0)gs_tpb=matrixDim; else if(matrixDim==0)gs_tpb=1; unsigned gs_nb=(matrixDim+gs_tpb-1)/gs_tpb; if(matrixDim==0)gs_nb=0; else if(gs_nb==0&&matrixDim>0)gs_nb=1;
                for(size_t j=0;j<n_non_tgt_cfg;++j){
                    if(status!=ROCQ_STATUS_SUCCESS)break; size_t base_idx_nt=0; unsigned cur_nt_pos=0;
                    for(unsigned bi=0;bi<local_num_qubits_for_kernel;++bi){bool is_tgt=false; for(unsigned k=0;k<m;++k)if(h_sorted_idx[k]==bi){is_tgt=true;break;} if(!is_tgt){if(((j>>cur_nt_pos)&1))base_idx_nt|=(1ULL<<bi);cur_nt_pos++;}}
                    if(gs_nb>0){
                        hipLaunchKernelGGL(gather_elements_kernel_v2,dim3(gs_nb),dim3(gs_tpb),0,h->streams[rank],d_tmp_in,current_local_slice_ptr,d_idx_gs,m,base_idx_nt); if(hipGetLastError()!=hipSuccess){status=ROCQ_STATUS_HIP_ERROR;break;}
                        if(rocblas_cgemv(h->blasHandles[rank],rocblas_operation_none,matrixDim,matrixDim,&alpha,(const rocblas_float_complex*)matrixDevice,matrixDim,(const rocblas_float_complex*)d_tmp_in,1,&beta,(rocblas_float_complex*)d_tmp_out)!=rocblas_status_success){status=ROCQ_STATUS_FAILURE;break;}
                        hipLaunchKernelGGL(scatter_elements_kernel_v2,dim3(gs_nb),dim3(gs_tpb),0,h->streams[rank],current_local_slice_ptr,d_tmp_out,d_idx_gs,m,base_idx_nt); if(hipGetLastError()!=hipSuccess){status=ROCQ_STATUS_HIP_ERROR;break;}
                    }
                } 
                hipStreamSynchronize(h->streams[rank]); hipFree(d_idx_gs);hipFree(d_tmp_in);hipFree(d_tmp_out); if(status!=ROCQ_STATUS_SUCCESS)break;
            } else { if(numTargetQubits==0&&matrixDim==1)status=ROCQ_STATUS_NOT_IMPLEMENTED; else status=ROCQ_STATUS_INVALID_VALUE; break;}
            if (numTargetQubits < 3 && numTargetQubits > 0) { hip_err = hipStreamSynchronize(h->streams[rank]); if (hip_err != hipSuccess && status == ROCQ_STATUS_SUCCESS) { status = ROCQ_STATUS_HIP_ERROR; break; }}
        } 
    } else {
        // Non-local gate application: requires communication.
        // Conceptual plan for rocsvApplyMatrix to handle global gates:
        // 1. Identify which of the `qubitIndices` are non-local (slice-determining bits, i.e., >= h->numLocalQubitsPerGpu)
        //    and which are local.
        // 2. Create a temporary list of target qubits for the local operation. This list will initially
        //    contain the original local target qubits.
        // 3. For each non-local target qubit `q_global`:
        //    a. Select a "temporary" local qubit index `q_temp_local` that is currently not
        //       among the target qubits (neither original local nor already used as temporary).
        //       This requires careful management if available local qubit indices are scarce.
        //       If no suitable temporary local qubit index is available, this strategy might fail or require
        //       more complex chained swaps.
        //    b. Call `rocsvSwapIndexBits(handle, q_global, q_temp_local)` to move the logical state of
        //       `q_global` into the local position `q_temp_local`.
        //    c. Record this swap: e.g., add `(q_global, q_temp_local)` to a list of performed swaps.
        //    d. Add `q_temp_local` to the list of target qubits for the upcoming local gate application.
        //       The original `q_global` is effectively replaced by `q_temp_local` for the matrix operation.
        // 4. Permute the `matrixDevice`: The provided `matrixDevice` is defined with respect to the
        //    original `qubitIndices`. Since some of these qubits have been swapped to new (temporary)
        //    positions for the local operation, the matrix itself must be permuted to match this
        //    new basis ordering. This is a significant operation involving row/column permutations
        //    of `matrixDevice` based on the swaps performed. A new device matrix, `d_permuted_matrix`,
        //    would be created.
        //    Alternatively, if the gate kernels (e.g. apply_multi_qubit_generic_matrix_kernel)
        //    could take a permutation map for their target indices relative to the matrix's canonical
        //    ordering, this could avoid explicit matrix permutation. This is more advanced.
        // 5. All target qubits for the operation are now effectively local (original local ones + temporary
        //    local ones that hold the state of original global ones).
        //    Call the local gate application logic (i.e., the code currently within the
        //    `if (are_qubits_local(...))` block) using:
        //    - The updated list of purely local target qubit indices.
        //    - The (conceptually) permuted `d_permuted_matrix`.
        // 6. Reverse Swaps: After the local gate application, undo all swaps performed in step 3b
        //    by calling `rocsvSwapIndexBits` for each recorded swap, in the reverse order.
        //    E.g., if swapped (q_g1, q_t1) then (q_g2, q_t2), unswap by calling for (q_g2, q_t2) then (q_g1, q_t1).
        //
        // Challenges:
        // - Finding available temporary local qubits.
        // - Performing the matrix permutation correctly and efficiently.
        // - The performance overhead of multiple `rocsvSwapIndexBits` calls.
        //
        // Due to these complexities, this full orchestration is not implemented here.
        // Users requiring global gates must currently use rocsvSwapIndexBits manually
        // to bring non-local qubits into the local domain before calling local gate functions.
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

    if (numQubits != h->globalNumQubits && h->globalNumQubits > 0) return ROCQ_STATUS_INVALID_VALUE;
    unsigned current_global_qubits = h->globalNumQubits;
    if (qubitToMeasure >= current_global_qubits && !(current_global_qubits == 0 && qubitToMeasure == 0)) {
         return ROCQ_STATUS_INVALID_VALUE;
    }

    unsigned KERNEL_BLOCK_SIZE = 256;

    if (h->numGpus == 1) {
        hipSetDevice(h->deviceIds[0]);
        hipStream_t current_stream = h->streams[0];
        rocComplex* current_d_state_slice = h->d_local_state_slices[0];
        unsigned num_qubits_in_slice = h->numLocalQubitsPerGpu;
        size_t slice_num_elements = h->localStateSizes[0];

        if (d_state != nullptr && d_state != current_d_state_slice) return ROCQ_STATUS_INVALID_VALUE;
        if (numQubits != current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;

        double h_prob0_sum_slice = 0.0;
        double h_prob1_sum_slice = 0.0;

        unsigned num_kernel_blocks = 0;
        if (slice_num_elements > 0) {
            num_kernel_blocks = (slice_num_elements + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
            if (num_kernel_blocks == 0) num_kernel_blocks = 1;
        }

        if (num_kernel_blocks > 0) {
            double* d_block_probs = nullptr;
            hipMalloc(&d_block_probs, num_kernel_blocks * 2 * sizeof(double));
            hipMemsetAsync(d_block_probs, 0, num_kernel_blocks * 2 * sizeof(double), current_stream);
            size_t shared_mem_size = KERNEL_BLOCK_SIZE * 2 * sizeof(double);
            hipLaunchKernelGGL(calculate_local_slice_probabilities_kernel,
                               dim3(num_kernel_blocks), dim3(KERNEL_BLOCK_SIZE), shared_mem_size, current_stream,
                               current_d_state_slice, slice_num_elements,
                               num_qubits_in_slice, qubitToMeasure,
                               d_block_probs);
            if (hipGetLastError() != hipSuccess) { if(d_block_probs) hipFree(d_block_probs); return ROCQ_STATUS_HIP_ERROR; }

            double* d_slice_probs_out = nullptr;
            hipMalloc(&d_slice_probs_out, 2 * sizeof(double));
            hipMemsetAsync(d_slice_probs_out, 0, 2*sizeof(double), current_stream);

            unsigned num_blocks_final_reduction = (num_kernel_blocks + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
            if(num_blocks_final_reduction == 0 && num_kernel_blocks > 0) num_blocks_final_reduction = 1;

            if(num_blocks_final_reduction > 0) {
                hipLaunchKernelGGL(reduce_block_sums_to_slice_total_probs_kernel,
                                dim3(num_blocks_final_reduction), dim3(KERNEL_BLOCK_SIZE), KERNEL_BLOCK_SIZE * 2 * sizeof(double), current_stream,
                                d_block_probs, num_kernel_blocks, d_slice_probs_out);
                if (hipGetLastError() != hipSuccess) { hipFree(d_block_probs); hipFree(d_slice_probs_out); return ROCQ_STATUS_HIP_ERROR; }
            }

            double h_slice_probs[2];
            hipMemcpy(h_slice_probs, d_slice_probs_out, 2 * sizeof(double), hipMemcpyDeviceToHost);
            h_prob0_sum_slice = h_slice_probs[0];
            h_prob1_sum_slice = h_slice_probs[1];
            if(d_block_probs) hipFree(d_block_probs);
            if(d_slice_probs_out) hipFree(d_slice_probs_out);

        } else if (num_qubits_in_slice == 0 && qubitToMeasure == 0 && slice_num_elements == 1) {
            rocComplex h_amp;
            hipMemcpy(&h_amp, current_d_state_slice, sizeof(rocComplex), hipMemcpyDeviceToHost);
            h_prob0_sum_slice = (double)h_amp.x * h_amp.x + (double)h_amp.y * h_amp.y;
            h_prob1_sum_slice = 0.0;
        }

        double prob0 = h_prob0_sum_slice;
        double prob1 = h_prob1_sum_slice;
        double total_prob_check_s = prob0 + prob1;

        if (fabs(total_prob_check_s) < 1e-12) { prob0 = 0.5; prob1 = 0.5;}
        else if (fabs(total_prob_check_s - 1.0) > 1e-9) { prob0 /= total_prob_check_s; prob1 = 1.0 - prob0;}
        if (prob0 < 0.0) prob0 = 0.0; if (prob0 > 1.0) prob0 = 1.0;
        prob1 = 1.0 - prob0;

        static bool seeded_s = false; if (!seeded_s) { srand((unsigned int)time(NULL)+1); seeded_s = true; }
        double rand_val_s = (double)rand() / RAND_MAX;

        if (rand_val_s < prob0) { *h_outcome = 0; *h_probability = prob0; }
        else { *h_outcome = 1; *h_probability = prob1; }

        unsigned threads_per_block_m = 256;
        unsigned num_blocks_m = (slice_num_elements + threads_per_block_m - 1) / threads_per_block_m;
        if (num_blocks_m == 0 && slice_num_elements > 0) num_blocks_m = 1;

        if (num_blocks_m > 0 && slice_num_elements > 0) {
            hipLaunchKernelGGL(collapse_state_kernel, dim3(num_blocks_m), dim3(threads_per_block_m), 0, current_stream,
                               current_d_state_slice, num_qubits_in_slice, qubitToMeasure, *h_outcome);
            if (hipGetLastError() != hipSuccess) return ROCQ_STATUS_HIP_ERROR;
        }

        double h_sum_sq_mag_s = 0.0;
        unsigned num_kernel_blocks_ssq = 0;
        if (slice_num_elements > 0) {
            num_kernel_blocks_ssq = (slice_num_elements + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
            if (num_kernel_blocks_ssq == 0) num_kernel_blocks_ssq = 1;
        }

        if (num_kernel_blocks_ssq > 0) {
            double* d_block_ssq_s = nullptr;
            hipMalloc(&d_block_ssq_s, num_kernel_blocks_ssq * sizeof(double));
            hipMemsetAsync(d_block_ssq_s, 0, num_kernel_blocks_ssq * sizeof(double), current_stream);
            size_t shared_mem_size_ssq = KERNEL_BLOCK_SIZE * sizeof(double);
            hipLaunchKernelGGL(calculate_local_slice_sum_sq_mag_kernel,
                           dim3(num_kernel_blocks_ssq), dim3(KERNEL_BLOCK_SIZE), shared_mem_size_ssq, current_stream,
                           current_d_state_slice, slice_num_elements, d_block_ssq_s);
            if (hipGetLastError() != hipSuccess) { if(d_block_ssq_s) hipFree(d_block_ssq_s); return ROCQ_STATUS_HIP_ERROR; }

            double* d_slice_ssq_out = nullptr;
            hipMalloc(&d_slice_ssq_out, sizeof(double));
            hipMemsetAsync(d_slice_ssq_out, 0, sizeof(double), current_stream);

            unsigned num_blocks_final_ssq_reduc = (num_kernel_blocks_ssq + KERNEL_BLOCK_SIZE-1) / KERNEL_BLOCK_SIZE;
            if(num_blocks_final_ssq_reduc == 0 && num_kernel_blocks_ssq > 0) num_blocks_final_ssq_reduc = 1;

            if(num_blocks_final_ssq_reduc > 0){
                hipLaunchKernelGGL(reduce_block_sums_to_slice_total_sum_sq_mag_kernel,
                                dim3(num_blocks_final_ssq_reduc), dim3(KERNEL_BLOCK_SIZE), KERNEL_BLOCK_SIZE * sizeof(double), current_stream,
                                d_block_ssq_s, num_kernel_blocks_ssq, d_slice_ssq_out);
                if (hipGetLastError() != hipSuccess) { hipFree(d_block_ssq_s); hipFree(d_slice_ssq_out); return ROCQ_STATUS_HIP_ERROR; }
            }
            hipMemcpy(&h_sum_sq_mag_s, d_slice_ssq_out, sizeof(double), hipMemcpyDeviceToHost);
            if(d_block_ssq_s) hipFree(d_block_ssq_s);
            if(d_slice_ssq_out) hipFree(d_slice_ssq_out);
        } else if (slice_num_elements == 1) {
             rocComplex h_amp_collapsed;
             hipMemcpy(&h_amp_collapsed, current_d_state_slice, sizeof(rocComplex), hipMemcpyDeviceToHost);
             h_sum_sq_mag_s = (double)h_amp_collapsed.x * h_amp_collapsed.x + (double)h_amp_collapsed.y * h_amp_collapsed.y;
        }

        if (fabs(h_sum_sq_mag_s) > 1e-12) {
            double norm_factor = 1.0 / sqrt(fabs(h_sum_sq_mag_s));
            if (num_blocks_m > 0 && slice_num_elements > 0) {
                hipLaunchKernelGGL(renormalize_state_kernel, dim3(num_blocks_m), dim3(threads_per_block_m), 0, current_stream,
                                   current_d_state_slice, num_qubits_in_slice, norm_factor);
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

    std::vector<double*> d_slice_probs_all_gpus(h->numGpus, nullptr);

    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;

        unsigned num_kernel_blocks_stage1 = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
        if (num_kernel_blocks_stage1 == 0 && h->localStateSizes[i] > 0) num_kernel_blocks_stage1 = 1;
        if (num_kernel_blocks_stage1 == 0) continue;

        double* d_block_partial_probs_gpu_i = nullptr;
        hipMalloc(&d_block_partial_probs_gpu_i, num_kernel_blocks_stage1 * 2 * sizeof(double));
        hipMemsetAsync(d_block_partial_probs_gpu_i, 0, num_kernel_blocks_stage1 * 2 * sizeof(double), h->streams[i]);

        size_t shared_mem_size_stage1 = KERNEL_BLOCK_SIZE * 2 * sizeof(double);
        hipLaunchKernelGGL(calculate_local_slice_probabilities_kernel,
                           dim3(num_kernel_blocks_stage1), dim3(KERNEL_BLOCK_SIZE), shared_mem_size_stage1, h->streams[i],
                           h->d_local_state_slices[i], h->localStateSizes[i],
                           h->numLocalQubitsPerGpu, local_target_qubit,
                           d_block_partial_probs_gpu_i);
        if (hipGetLastError() != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; if(d_block_partial_probs_gpu_i) hipFree(d_block_partial_probs_gpu_i); goto mgpu_measure_cleanup_probs_rccl; }

        hipMalloc(&d_slice_probs_all_gpus[i], 2 * sizeof(double));
        hipMemsetAsync(d_slice_probs_all_gpus[i], 0, 2 * sizeof(double), h->streams[i]);

        unsigned num_kernel_blocks_stage2 = (num_kernel_blocks_stage1 + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
        if (num_kernel_blocks_stage2 == 0 && num_kernel_blocks_stage1 > 0) num_kernel_blocks_stage2 = 1;

        if (num_kernel_blocks_stage2 > 0) {
             hipLaunchKernelGGL(reduce_block_sums_to_slice_total_probs_kernel,
                               dim3(num_kernel_blocks_stage2), dim3(KERNEL_BLOCK_SIZE), KERNEL_BLOCK_SIZE * 2 * sizeof(double), h->streams[i],
                               d_block_partial_probs_gpu_i, num_kernel_blocks_stage1, d_slice_probs_all_gpus[i]);
            if (hipGetLastError() != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; if(d_block_partial_probs_gpu_i) hipFree(d_block_partial_probs_gpu_i); goto mgpu_measure_cleanup_probs_rccl; }
        }
        if(d_block_partial_probs_gpu_i) hipFree(d_block_partial_probs_gpu_i);
    }
    if(status != ROCQ_STATUS_SUCCESS) goto mgpu_measure_cleanup_probs_rccl;

    for (int i = 0; i < h->numGpus; ++i) { // Sync before AllReduce
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0 || d_slice_probs_all_gpus[i] == nullptr) continue;
        hipStreamSynchronize(h->streams[i]);
    }

    rcclGroupStart();
    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0 || d_slice_probs_all_gpus[i] == nullptr) continue;
        rcclAllReduce(d_slice_probs_all_gpus[i], d_slice_probs_all_gpus[i], 2, rcclDouble, rcclSum, h->comms[i], h->streams[i]);
    }
    rcclGroupEnd();

    for (int i = 0; i < h->numGpus; ++i) { // Sync AllReduce
        hipSetDevice(h->deviceIds[i]);
         if (h->localStateSizes[i] == 0 || d_slice_probs_all_gpus[i] == nullptr) continue;
        if(hipStreamSynchronize(h->streams[i]) != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = ROCQ_STATUS_HIP_ERROR;
    }
     if(status != ROCQ_STATUS_SUCCESS) goto mgpu_measure_cleanup_probs_rccl;

    double h_global_probs[2] = {0.0, 0.0};
    int first_participating_gpu = -1;
    for(int i=0; i < h->numGpus; ++i) {
        if (d_slice_probs_all_gpus[i] != nullptr) {
            first_participating_gpu = i;
            break;
        }
    }
    if (first_participating_gpu != -1) {
        hipSetDevice(h->deviceIds[first_participating_gpu]);
        hipMemcpy(&h_global_probs[0], d_slice_probs_all_gpus[first_participating_gpu], 2 * sizeof(double), hipMemcpyDeviceToHost);
        global_prob0 = h_global_probs[0];
        global_prob1 = h_global_probs[1];
    }

mgpu_measure_cleanup_probs_rccl:
    for(int i=0; i<h->numGpus; ++i) if(d_slice_probs_all_gpus[i]) { hipSetDevice(h->deviceIds[i]); hipFree(d_slice_probs_all_gpus[i]); d_slice_probs_all_gpus[i] = nullptr;}
    if(status != ROCQ_STATUS_SUCCESS) return status;

    double total_prob_check_m = global_prob0 + global_prob1;
    if (fabs(total_prob_check_m) < 1e-12) { global_prob0 = 0.5; global_prob1 = 0.5;}
    else if (fabs(total_prob_check_m - 1.0) > 1e-9) { global_prob0 /= total_prob_check_m; global_prob1 = 1.0 - global_prob0;}
    if (global_prob0 < 0.0) global_prob0 = 0.0; if (global_prob0 > 1.0) global_prob0 = 1.0;
    global_prob1 = 1.0 - global_prob0;

    static bool seeded_m = false; if (!seeded_m) { srand((unsigned int)time(NULL)+2); seeded_m = true; }
    double rand_val_m = (double)rand() / RAND_MAX;

    if (rand_val_m < global_prob0) { *h_outcome = 0; *h_probability = global_prob0; }
    else { *h_outcome = 1; *h_probability = global_prob1; }

    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;
        unsigned num_blocks_collapse = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
        if (num_blocks_collapse == 0 && h->localStateSizes[i] > 0) num_blocks_collapse = 1;
        if (num_blocks_collapse > 0) {
            hipLaunchKernelGGL(collapse_state_kernel, dim3(num_blocks_collapse), dim3(KERNEL_BLOCK_SIZE), 0, h->streams[i],
                               h->d_local_state_slices[i], h->numLocalQubitsPerGpu, local_target_qubit, *h_outcome);
            if(hipGetLastError() != hipSuccess) {status = ROCQ_STATUS_HIP_ERROR; goto mgpu_measure_cleanup_renorm_rccl;}
        }
    }
    for (int i = 0; i < h->numGpus; ++i) { // Sync collapse
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;
        if(hipStreamSynchronize(h->streams[i]) != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = ROCQ_STATUS_HIP_ERROR;
    }
    if(status != ROCQ_STATUS_SUCCESS) goto mgpu_measure_cleanup_renorm_rccl;

    double global_sum_sq_mag_collapsed = 0.0;
    std::vector<double*> d_slice_sum_sq_mag_all_gpus(h->numGpus, nullptr);

    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0) continue;
        unsigned num_kernel_blocks_s1 = (h->localStateSizes[i] + KERNEL_BLOCK_SIZE -1) / KERNEL_BLOCK_SIZE;
        if (num_kernel_blocks_s1 == 0 && h->localStateSizes[i] > 0) num_kernel_blocks_s1 = 1;
        if (num_kernel_blocks_s1 == 0) continue;

        double* d_block_ssq_gpu_i = nullptr;
        hipMalloc(&d_block_ssq_gpu_i, num_kernel_blocks_s1 * sizeof(double));
        hipMemsetAsync(d_block_ssq_gpu_i, 0, num_kernel_blocks_s1 * sizeof(double), h->streams[i]);
        size_t shared_mem_size_ssq = KERNEL_BLOCK_SIZE * sizeof(double);
        hipLaunchKernelGGL(calculate_local_slice_sum_sq_mag_kernel,
                           dim3(num_kernel_blocks_s1), dim3(KERNEL_BLOCK_SIZE), shared_mem_size_ssq, h->streams[i],
                           h->d_local_state_slices[i], h->localStateSizes[i], d_block_ssq_gpu_i);
        if(hipGetLastError() != hipSuccess) {status = ROCQ_STATUS_HIP_ERROR; if(d_block_ssq_gpu_i) hipFree(d_block_ssq_gpu_i); goto mgpu_measure_cleanup_renorm_rccl;}

        hipMalloc(&d_slice_sum_sq_mag_all_gpus[i], sizeof(double));
        hipMemsetAsync(d_slice_sum_sq_mag_all_gpus[i], 0, sizeof(double), h->streams[i]);
        unsigned num_kernel_blocks_s2 = (num_kernel_blocks_s1 + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
        if(num_kernel_blocks_s2 == 0 && num_kernel_blocks_s1 > 0) num_kernel_blocks_s2 = 1;

        if(num_kernel_blocks_s2 > 0) {
            hipLaunchKernelGGL(reduce_block_sums_to_slice_total_sum_sq_mag_kernel,
                            dim3(num_kernel_blocks_s2), dim3(KERNEL_BLOCK_SIZE), KERNEL_BLOCK_SIZE * sizeof(double), h->streams[i],
                            d_block_ssq_gpu_i, num_kernel_blocks_s1, d_slice_sum_sq_mag_all_gpus[i]);
            if(hipGetLastError() != hipSuccess) {status = ROCQ_STATUS_HIP_ERROR; if(d_block_ssq_gpu_i) hipFree(d_block_ssq_gpu_i); goto mgpu_measure_cleanup_renorm_rccl;}
        }
        if(d_block_ssq_gpu_i) hipFree(d_block_ssq_gpu_i);
    }
     if(status != ROCQ_STATUS_SUCCESS) goto mgpu_measure_cleanup_renorm_rccl;

    for (int i = 0; i < h->numGpus; ++i) { // Sync before AllReduce
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0 || d_slice_sum_sq_mag_all_gpus[i] == nullptr) continue;
        hipStreamSynchronize(h->streams[i]);
    }

    rcclGroupStart();
    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0 || d_slice_sum_sq_mag_all_gpus[i] == nullptr) continue;
        rcclAllReduce(d_slice_sum_sq_mag_all_gpus[i], d_slice_sum_sq_mag_all_gpus[i], 1, rcclDouble, rcclSum, h->comms[i], h->streams[i]);
    }
    rcclGroupEnd();
    for (int i = 0; i < h->numGpus; ++i) { // Sync AllReduce
        hipSetDevice(h->deviceIds[i]);
        if (h->localStateSizes[i] == 0 || d_slice_sum_sq_mag_all_gpus[i] == nullptr) continue;
        if(hipStreamSynchronize(h->streams[i]) != hipSuccess && status == ROCQ_STATUS_SUCCESS) status = ROCQ_STATUS_HIP_ERROR;
    }
    if(status != ROCQ_STATUS_SUCCESS) goto mgpu_measure_cleanup_renorm_rccl;

    first_participating_gpu = -1;
    for(int i=0; i < h->numGpus; ++i) {
        if (d_slice_sum_sq_mag_all_gpus[i] != nullptr) {
            first_participating_gpu = i;
            break;
        }
    }
    if (first_participating_gpu != -1) {
        hipSetDevice(h->deviceIds[first_participating_gpu]);
        hipMemcpy(&global_sum_sq_mag_collapsed, d_slice_sum_sq_mag_all_gpus[first_participating_gpu], sizeof(double), hipMemcpyDeviceToHost);
    } else {
        global_sum_sq_mag_collapsed = 0.0;
    }

mgpu_measure_cleanup_renorm_rccl:
    for(int i=0; i<h->numGpus; ++i) if(d_slice_sum_sq_mag_all_gpus[i]) {hipSetDevice(h->deviceIds[i]); hipFree(d_slice_sum_sq_mag_all_gpus[i]); d_slice_sum_sq_mag_all_gpus[i] = nullptr;}
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
                 if(hipGetLastError() != hipSuccess) {status = ROCQ_STATUS_HIP_ERROR; goto mgpu_measure_final_sync_rccl;}
            }
        }
    }

mgpu_measure_final_sync_rccl:
    for (int i = 0; i < h->numGpus; ++i) {
        hipSetDevice(h->deviceIds[i]);
        hipError_t err_sync = hipStreamSynchronize(h->streams[i]);
        if (err_sync != hipSuccess && status == ROCQ_STATUS_SUCCESS) {
            status = checkHipError(err_sync, "rocsvMeasure multi-GPU final sync");
        }
    }
    return status;
}

// API Functions for Specific Gates (Unchanged from previous step, ensure they are present)
// ... rocsvApplyX, Y, Z, H, S, T, Rx, Ry, Rz ...
// ... rocsvApplyCNOT, CZ, SWAP ...
// ... rocsvApplyFusedSingleQubitMatrix ...
// ... rocsvSwapIndexBits ...
// ... Host-side helper functions ...

rocqStatus_t rocsvApplyX(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyX hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                 hipLaunchKernelGGL(apply_X_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank],
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, targetQubitLocal);
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyX apply_X_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyX hipStreamSynchronize"); break; }
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyY(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyY hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                 hipLaunchKernelGGL(apply_Y_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank],
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, targetQubitLocal);
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyY apply_Y_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyY hipStreamSynchronize"); break; }
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for(int rank=0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyZ hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if(num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_Z_kernel,dim3(num_blocks),dim3(threads_per_block),0,h->streams[rank],h->d_local_state_slices[rank],local_num_qubits_for_kernel,targetQubitLocal);
                if(hipGetLastError()!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
            }
            if(hipStreamSynchronize(h->streams[rank])!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyH(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for(int rank=0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyH hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if(num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_H_kernel,dim3(num_blocks),dim3(threads_per_block),0,h->streams[rank],h->d_local_state_slices[rank],local_num_qubits_for_kernel,targetQubitLocal);
                if(hipGetLastError()!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
            }
            if(hipStreamSynchronize(h->streams[rank])!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyS(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for(int rank=0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyS hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if(num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_S_kernel,dim3(num_blocks),dim3(threads_per_block),0,h->streams[rank],h->d_local_state_slices[rank],local_num_qubits_for_kernel,targetQubitLocal);
                if(hipGetLastError()!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
            }
            if(hipStreamSynchronize(h->streams[rank])!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for(int rank=0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyT hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if(num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_T_kernel,dim3(num_blocks),dim3(threads_per_block),0,h->streams[rank],h->d_local_state_slices[rank],local_num_qubits_for_kernel,targetQubitLocal);
                if(hipGetLastError()!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
            }
            if(hipStreamSynchronize(h->streams[rank])!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyRx(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyRx hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                 hipLaunchKernelGGL(apply_Rx_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank],
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, targetQubitLocal, static_cast<float>(theta));
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyRx apply_Rx_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyRx hipStreamSynchronize"); break; }
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyRy(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 0;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for(int rank=0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyRy hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if(num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_Ry_kernel,dim3(num_blocks),dim3(threads_per_block),0,h->streams[rank],h->d_local_state_slices[rank],local_num_qubits_for_kernel,targetQubitLocal,static_cast<float>(theta));
                if(hipGetLastError()!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
            }
            if(hipStreamSynchronize(h->streams[rank])!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyRz(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit, double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (targetQubit >= current_global_qubits) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 && targetQubit > 0) return ROCQ_STATUS_INVALID_VALUE;
    if (are_qubits_local(h, &targetQubit, 1)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS;
        unsigned threads_per_block = 256;
        size_t num_thread_groups = (local_slice_num_elements > 0) ? local_slice_num_elements / 2 : 0;
        unsigned num_blocks = (num_thread_groups + threads_per_block - 1) / threads_per_block;
        if (num_thread_groups > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_thread_groups == 0) {
             if (local_num_qubits_for_kernel == 0 && targetQubitLocal == 0 && local_slice_num_elements == 1) num_blocks = 1;
             else if (local_num_qubits_for_kernel == 1 && local_slice_num_elements == 2) num_blocks = 1;
             else num_blocks = 0;
        }
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for(int rank=0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyRz hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if(num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_Rz_kernel,dim3(num_blocks),dim3(threads_per_block),0,h->streams[rank],h->d_local_state_slices[rank],local_num_qubits_for_kernel,targetQubitLocal,static_cast<float>(theta));
                if(hipGetLastError()!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
            }
            if(hipStreamSynchronize(h->streams[rank])!=hipSuccess){status = ROCQ_STATUS_HIP_ERROR; break;}
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (controlQubit >= current_global_qubits || targetQubit >= current_global_qubits || controlQubit == targetQubit) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits < 2 && (controlQubit > 0 || targetQubit > 0)) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0 ) return ROCQ_STATUS_INVALID_VALUE;
    unsigned qubitIndices[2] = {controlQubit, targetQubit};
    if (are_qubits_local(h, qubitIndices, 2)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned controlQubitLocal = controlQubit; unsigned targetQubitLocal = targetQubit;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS; unsigned threads_per_block = 256;
        size_t num_kernel_threads = (local_num_qubits_for_kernel < 2) ? 0 : (1ULL << (local_num_qubits_for_kernel - 2));
        unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
        if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_kernel_threads == 0 && local_num_qubits_for_kernel >=2) num_blocks = 1;
        else if (num_kernel_threads == 0) num_blocks = 0;
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyCNOT hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_CNOT_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank],
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, controlQubitLocal, targetQubitLocal);
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyCNOT apply_CNOT_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyCNOT hipStreamSynchronize"); break; }
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplyCZ(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (qubit1 >= current_global_qubits || qubit2 >= current_global_qubits || qubit1 == qubit2) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits < 2 && (qubit1 > 0 || qubit2 > 0)) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0) return ROCQ_STATUS_INVALID_VALUE;
    unsigned qubitIndices[2] = {qubit1, qubit2};
    if (are_qubits_local(h, qubitIndices, 2)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned local_q1 = qubit1; unsigned local_q2 = qubit2;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS; unsigned threads_per_block = 256;
        size_t num_kernel_threads = (local_num_qubits_for_kernel < 2) ? 0 : (1ULL << (local_num_qubits_for_kernel - 2));
        unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
        if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_kernel_threads == 0 && local_num_qubits_for_kernel >=2) num_blocks = 1;
        else if (num_kernel_threads == 0) num_blocks = 0;
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplyCZ hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_CZ_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank],
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, local_q1, local_q2);
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplyCZ apply_CZ_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplyCZ hipStreamSynchronize"); break; }
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

rocqStatus_t rocsvApplySWAP(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubit1, unsigned qubit2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocsvInternalHandle* h = static_cast<rocsvInternalHandle*>(handle);
    if (h->numGpus == 0) return ROCQ_STATUS_FAILURE;
    unsigned current_global_qubits = (h->globalNumQubits > 0) ? h->globalNumQubits : numQubits;
    if (qubit1 >= current_global_qubits || qubit2 >= current_global_qubits || qubit1 == qubit2) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits < 2 && (qubit1 > 0 || qubit2 > 0)) return ROCQ_STATUS_INVALID_VALUE;
    if (current_global_qubits == 0) return ROCQ_STATUS_INVALID_VALUE;
    unsigned qubitIndices[2] = {qubit1, qubit2};
    if (are_qubits_local(h, qubitIndices, 2)) {
        unsigned local_num_qubits_for_kernel = h->numLocalQubitsPerGpu;
        size_t local_slice_num_elements = (1ULL << local_num_qubits_for_kernel);
        unsigned local_q1_kernel = qubit1; unsigned local_q2_kernel = qubit2;
        rocqStatus_t status = ROCQ_STATUS_SUCCESS; unsigned threads_per_block = 256;
        size_t num_kernel_threads = (local_num_qubits_for_kernel < 2) ? 0 : (1ULL << (local_num_qubits_for_kernel - 2));
        unsigned num_blocks = (num_kernel_threads + threads_per_block - 1) / threads_per_block;
        if (num_kernel_threads > 0 && num_blocks == 0) num_blocks = 1;
        else if (num_kernel_threads == 0 && local_num_qubits_for_kernel >=2) num_blocks = 1;
        else if (num_kernel_threads == 0) num_blocks = 0;
        if (local_slice_num_elements == 0 && current_global_qubits > 0 && h->numGpus > 0) num_blocks = 0;
        for (int rank = 0; rank < h->numGpus; ++rank) {
            hipError_t hip_err_set = hipSetDevice(h->deviceIds[rank]);
            if (hip_err_set != hipSuccess) { status = checkHipError(hip_err_set, "rocsvApplySWAP hipSetDevice"); break; }
            if (h->d_local_state_slices[rank] == nullptr && h->localStateSizes[rank] > 0) { status = ROCQ_STATUS_INVALID_VALUE; break; }
            if (h->localStateSizes[rank] != local_slice_num_elements && h->numGpus > 1 && h->globalNumQubits > 0) {status = ROCQ_STATUS_INVALID_VALUE; break;}
            if (num_blocks > 0 && h->localStateSizes[rank] > 0) {
                hipLaunchKernelGGL(apply_SWAP_kernel, dim3(num_blocks), dim3(threads_per_block), 0, h->streams[rank],
                                   h->d_local_state_slices[rank], local_num_qubits_for_kernel, local_q1_kernel, local_q2_kernel);
                hipError_t hip_err_kernel = hipGetLastError();
                if (hip_err_kernel != hipSuccess) { status = checkHipError(hip_err_kernel, "rocsvApplySWAP apply_SWAP_kernel"); break; }
            }
            hipError_t hip_err_sync = hipStreamSynchronize(h->streams[rank]);
            if (hip_err_sync != hipSuccess) { status = checkHipError(hip_err_sync, "rocsvApplySWAP hipStreamSynchronize"); break; }
        }
        return status;
    } else return ROCQ_STATUS_NOT_IMPLEMENTED;
}

} // extern "C"
