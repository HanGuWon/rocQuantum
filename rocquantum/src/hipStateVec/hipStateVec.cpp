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
    unsigned numQubits = 0;
    rocComplex* d_state = nullptr;
    bool ownsState = false;
};

namespace {
__host__ __device__ inline rocComplex make_unit_complex(double real, double imag) {
#ifdef ROCQ_PRECISION_DOUBLE
    return rocComplex{real, imag};
#else
    return rocComplex{static_cast<float>(real), static_cast<float>(imag)};
#endif
}

inline rocComplex* resolve_state_pointer(rocsvInternalHandle* handle, rocComplex* external) {
    if (external) {
        return external;
    }
    return handle ? handle->d_state : nullptr;
}
} // namespace

rocqStatus_t rocsvCreate(rocsvHandle_t* handle) {
    *handle = new rocsvInternalHandle();
    hipStreamCreate(&((*handle)->streams[0]));
    (*handle)->batchSize = 1;
    (*handle)->numQubits = 0;
    (*handle)->d_state = nullptr;
    (*handle)->ownsState = false;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvDestroy(rocsvHandle_t handle) {
    if (handle) {
        rocsvFreeState(handle);
        hipStreamDestroy(handle->streams[0]);
        delete handle;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle, unsigned numQubits, rocComplex** d_state, size_t batchSize) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    handle->batchSize = batchSize;
    handle->numQubits = numQubits;

    if (handle->d_state && handle->ownsState) {
        hipFree(handle->d_state);
        handle->d_state = nullptr;
        handle->ownsState = false;
    }

    size_t num_elements_per_state = 1ULL << numQubits;
    size_t total_elements = batchSize * num_elements_per_state;
    rocComplex* allocated_ptr = nullptr;
    if (hipMalloc(&allocated_ptr, total_elements * sizeof(rocComplex)) != hipSuccess) {
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }

    if (d_state) {
        *d_state = allocated_ptr;
    }

    handle->d_state = allocated_ptr;
    handle->ownsState = true;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvFreeState(rocsvHandle_t handle) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    if (handle->d_state && handle->ownsState) {
        hipFree(handle->d_state);
    }
    handle->d_state = nullptr;
    handle->ownsState = false;
    handle->numQubits = 0;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvInitializeState(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* target_state = resolve_state_pointer(handle, d_state);
    if (!target_state) return ROCQ_STATUS_INVALID_VALUE;

    size_t total_elements = handle->batchSize * (1ULL << numQubits);
    if (hipMemset(target_state, 0, total_elements * sizeof(rocComplex)) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    rocComplex one = make_unit_complex(1.0, 0.0);
    if (hipMemcpy(target_state, &one, sizeof(rocComplex), hipMemcpyHostToDevice) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    handle->numQubits = numQubits;
    return ROCQ_STATUS_SUCCESS;
}

// --- Gate Implementations (Updated for Batching) ---

// Example: rocsvApplyH
rocqStatus_t rocsvApplyH(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned targetQubit) {
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t total_threads = handle->batchSize * (num_elements_per_state / 2);
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(apply_H_kernel, dim3(blocks), dim3(threads_per_block), 0, handle->streams[0],
                       state, numQubits, targetQubit, handle->batchSize);
    return ROCQ_STATUS_SUCCESS;
}

// Example: rocsvApplyCNOT
rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned controlQubit, unsigned targetQubit) {
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t total_threads = handle->batchSize * (num_elements_per_state / 4);
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(apply_CNOT_kernel, dim3(blocks), dim3(threads_per_block), 0, handle->streams[0],
                       state, numQubits, controlQubit, targetQubit, handle->batchSize);
    return ROCQ_STATUS_SUCCESS;
}

// ... ALL OTHER rocsvApply... functions must be updated similarly ...
// (The full file would show updates for X, Y, Z, S, T, rotations, CZ, SWAP, CRX, etc.)

rocqStatus_t rocsvGetStateVectorFull(rocsvHandle_t handle, rocComplex* d_state, rocComplex* h_state) {
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    size_t num_elements_per_state = 1ULL << handle->numQubits;
    size_t total_elements = handle->batchSize * num_elements_per_state;
    if (hipMemcpy(h_state, state, total_elements * sizeof(rocComplex), hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetStateVectorSlice(rocsvHandle_t handle, rocComplex* d_state, rocComplex* h_state, unsigned batch_index) {
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    size_t num_elements_per_state = 1ULL << handle->numQubits;
    size_t offset = batch_index * num_elements_per_state;
    rocComplex* d_state_slice = state + offset;
    if (hipMemcpy(h_state, d_state_slice, num_elements_per_state * sizeof(rocComplex), hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}
