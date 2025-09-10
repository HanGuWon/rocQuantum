#include "rocquantum/hipStateVec.h"
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

// Forward declare all kernels with batchSize parameter
__global__ void apply_X_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, size_t batchSize);
__global__ void apply_H_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, size_t batchSize);
__global__ void apply_CNOT_kernel(rocComplex* state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit, size_t batchSize);
// ... other kernel declarations with batchSize ...

struct rocsvInternalHandle {
    hipStream_t streams[1];
    size_t batchSize = 1;
};

rocqStatus_t rocsvCreate(rocsvHandle_t* handle) {
    *handle = new rocsvInternalHandle();
    hipStreamCreate(&((*handle)->streams[0]));
    (*handle)->batchSize = 1;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvDestroy(rocsvHandle_t handle) {
    if (handle) {
        hipStreamDestroy(handle->streams[0]);
        delete handle;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle, unsigned numQubits, rocComplex** d_state, size_t batchSize) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    handle->batchSize = batchSize;
    size_t num_elements_per_state = 1ULL << numQubits;
    size_t total_elements = batchSize * num_elements_per_state;
    if (hipMalloc(d_state, total_elements * sizeof(rocComplex)) != hipSuccess) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    return ROCQ_STATUS_SUCCESS;
}

// --- Gate Implementations (Updated for Batching) ---

// Example: rocsvApplyH
rocqStatus_t rocsvApplyH(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t total_threads = handle->batchSize * (num_elements_per_state / 2);
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(apply_H_kernel, dim3(blocks), dim3(threads_per_block), 0, handle->streams[0],
                       d_state, numQubits, targetQubit, handle->batchSize);
    return ROCQ_STATUS_SUCCESS;
}

// Example: rocsvApplyCNOT
rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit) {
    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t total_threads = handle->batchSize * (num_elements_per_state / 4);
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(apply_CNOT_kernel, dim3(blocks), dim3(threads_per_block), 0, handle->streams[0],
                       d_state, numQubits, controlQubit, targetQubit, handle->batchSize);
    return ROCQ_STATUS_SUCCESS;
}

// ... ALL OTHER rocsvApply... functions must be updated similarly ...
// (The full file would show updates for X, Y, Z, S, T, rotations, CZ, SWAP, CRX, etc.)

rocqStatus_t rocsvGetStateVectorFull(rocsvHandle_t handle, rocComplex* d_state, rocComplex* h_state) {
    size_t num_elements_per_state = 1ULL << handle->numQubits;
    size_t total_elements = handle->batchSize * num_elements_per_state;
    if (hipMemcpy(h_state, d_state, total_elements * sizeof(rocComplex), hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetStateVectorSlice(rocsvHandle_t handle, rocComplex* d_state, rocComplex* h_state, unsigned batch_index) {
    size_t num_elements_per_state = 1ULL << handle->numQubits;
    size_t offset = batch_index * num_elements_per_state;
    rocComplex* d_state_slice = d_state + offset;
    if (hipMemcpy(h_state, d_state_slice, num_elements_per_state * sizeof(rocComplex), hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}
