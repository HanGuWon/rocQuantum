#include "rocquantum/hipStateVec.h"

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// Kernels implemented in the corresponding .hip translation units.
__global__ void apply_single_qubit_matrix_kernel(rocComplex* state,
                                                 unsigned numQubits,
                                                 unsigned targetQubit,
                                                 rocComplex m00,
                                                 rocComplex m01,
                                                 rocComplex m10,
                                                 rocComplex m11,
                                                 size_t batchSize);

__global__ void apply_controlled_single_qubit_matrix_kernel(rocComplex* state,
                                                            unsigned numQubits,
                                                            unsigned controlQubit,
                                                            unsigned targetQubit,
                                                            rocComplex m00,
                                                            rocComplex m01,
                                                            rocComplex m10,
                                                            rocComplex m11,
                                                            size_t batchSize);

__global__ void apply_CNOT_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned controlQubit,
                                  unsigned targetQubit,
                                  size_t batchSize);

__global__ void apply_CZ_kernel(rocComplex* state,
                                unsigned numQubits,
                                unsigned controlQubit,
                                unsigned targetQubit,
                                size_t batchSize);

__global__ void apply_SWAP_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned qubitA,
                                  unsigned qubitB,
                                  size_t batchSize);

__global__ void apply_multi_controlled_x_kernel(rocComplex* state,
                                                unsigned numQubits,
                                                unsigned long long controlMask,
                                                unsigned targetQubit,
                                                size_t batchSize);

__global__ void apply_CSWAP_kernel(rocComplex* state,
                                   unsigned numQubits,
                                   unsigned controlQubit,
                                   unsigned targetQubit1,
                                   unsigned targetQubit2,
                                   size_t batchSize);

struct rocsvInternalHandle {
    hipStream_t streams[1];
    size_t batchSize = 1;
    unsigned numQubits = 0;
    rocComplex* d_state = nullptr;
    bool ownsState = false;
};

namespace {

__host__ __device__ inline rocComplex make_complex(double real, double imag) {
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

inline rocqStatus_t check_last_hip_error() {
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "hipStateVec kernel launch failed: " << hipGetErrorString(err) << '\n';
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

inline bool validate_qubit_index(unsigned qubit, unsigned numQubits) {
    return qubit < numQubits;
}

rocqStatus_t launch_single_qubit_matrix(rocsvInternalHandle* handle,
                                        rocComplex* state,
                                        unsigned numQubits,
                                        unsigned targetQubit,
                                        const rocComplex& m00,
                                        const rocComplex& m01,
                                        const rocComplex& m10,
                                        const rocComplex& m11) {
    if (!validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t pairs_per_state = elements_per_state >> 1;
    const size_t total_pairs = handle->batchSize * pairs_per_state;
    if (total_pairs == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_pairs + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_single_qubit_matrix_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       targetQubit,
                       m00,
                       m01,
                       m10,
                       m11,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t launch_controlled_single_qubit_matrix(rocsvInternalHandle* handle,
                                                   rocComplex* state,
                                                   unsigned numQubits,
                                                   unsigned controlQubit,
                                                   unsigned targetQubit,
                                                   const rocComplex& m00,
                                                   const rocComplex& m01,
                                                   const rocComplex& m10,
                                                   const rocComplex& m11) {
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit, numQubits) ||
        controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t pairs_per_state = elements_per_state >> 1;
    const size_t total_pairs = handle->batchSize * pairs_per_state;
    if (total_pairs == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_pairs + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_controlled_single_qubit_matrix_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit,
                       m00,
                       m01,
                       m10,
                       m11,
                       handle->batchSize);
    return check_last_hip_error();
}

} // namespace

rocqStatus_t rocsvCreate(rocsvHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
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

rocqStatus_t rocsvAllocateState(rocsvHandle_t handle,
                                unsigned numQubits,
                                rocComplex** d_state,
                                size_t batchSize) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    handle->batchSize = batchSize > 0 ? batchSize : 1;
    handle->numQubits = numQubits;

    if (handle->d_state && handle->ownsState) {
        hipFree(handle->d_state);
        handle->d_state = nullptr;
        handle->ownsState = false;
    }

    const size_t num_elements_per_state = 1ULL << numQubits;
    const size_t total_elements = handle->batchSize * num_elements_per_state;
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

rocqStatus_t rocsvInitializeState(rocsvHandle_t handle,
                                  rocComplex* d_state,
                                  unsigned numQubits) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* target_state = resolve_state_pointer(handle, d_state);
    if (!target_state) return ROCQ_STATUS_INVALID_VALUE;

    const size_t total_elements = handle->batchSize * (1ULL << numQubits);
    if (hipMemset(target_state, 0, total_elements * sizeof(rocComplex)) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    const rocComplex one = make_complex(1.0, 0.0);
    if (hipMemcpy(target_state, &one, sizeof(rocComplex), hipMemcpyHostToDevice) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }

    handle->numQubits = numQubits;
    return ROCQ_STATUS_SUCCESS;
}

// --- Single-qubit named gates ------------------------------------------------

rocqStatus_t rocsvApplyH(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const rocComplex m00 = make_complex(inv_sqrt2, 0.0);
    const rocComplex m01 = make_complex(inv_sqrt2, 0.0);
    const rocComplex m10 = make_complex(inv_sqrt2, 0.0);
    const rocComplex m11 = make_complex(-inv_sqrt2, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, m00, m01, m10, m11);
}

rocqStatus_t rocsvApplyX(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex one = make_complex(1.0, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, zero, one, one, zero);
}

rocqStatus_t rocsvApplyY(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex minus_i = make_complex(0.0, -1.0);
    const rocComplex plus_i = make_complex(0.0, 1.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, zero, minus_i, plus_i, zero);
}

rocqStatus_t rocsvApplyZ(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex minus_one = make_complex(-1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, minus_one);
}

rocqStatus_t rocsvApplyS(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex plus_i = make_complex(0.0, 1.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, plus_i);
}

rocqStatus_t rocsvApplySdg(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex minus_i = make_complex(0.0, -1.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, minus_i);
}

rocqStatus_t rocsvApplyT(rocsvHandle_t handle,
                         rocComplex* d_state,
                         unsigned numQubits,
                         unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    constexpr double pi = 3.14159265358979323846;
    const double phase = pi / 4.0;
    const rocComplex one = make_complex(1.0, 0.0);
    const rocComplex zero = make_complex(0.0, 0.0);
    const rocComplex t_phase = make_complex(std::cos(phase), std::sin(phase));
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit, one, zero, zero, t_phase);
}

// --- Parametrised single-qubit rotations ------------------------------------

rocqStatus_t rocsvApplyRx(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned targetQubit,
                          double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_i_s = make_complex(0.0, -std::sin(half));
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit,
                                      c, minus_i_s, minus_i_s, c);
}

rocqStatus_t rocsvApplyRy(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned targetQubit,
                          double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_s = make_complex(-std::sin(half), 0.0);
    const rocComplex plus_s = make_complex(std::sin(half), 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit,
                                      c, minus_s, plus_s, c);
}

rocqStatus_t rocsvApplyRz(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned targetQubit,
                          double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex phase_pos = make_complex(std::cos(half), std::sin(half));
    const rocComplex phase_neg = make_complex(std::cos(half), -std::sin(half));
    const rocComplex zero = make_complex(0.0, 0.0);
    return launch_single_qubit_matrix(handle, state, numQubits, targetQubit,
                                      phase_neg, zero, zero, phase_pos);
}

// --- Two-qubit named gates ---------------------------------------------------

rocqStatus_t rocsvApplyCNOT(rocsvHandle_t handle,
                            rocComplex* d_state,
                            unsigned numQubits,
                            unsigned controlQubit,
                            unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit, numQubits) ||
        controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 4) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_CNOT_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplyCZ(rocsvHandle_t handle,
                          rocComplex* d_state,
                          unsigned numQubits,
                          unsigned controlQubit,
                          unsigned targetQubit) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit, numQubits) ||
        controlQubit == targetQubit) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 4) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_CZ_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplySWAP(rocsvHandle_t handle,
                            rocComplex* d_state,
                            unsigned numQubits,
                            unsigned qubitA,
                            unsigned qubitB) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(qubitA, numQubits) ||
        !validate_qubit_index(qubitB, numQubits) ||
        qubitA == qubitB) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 4) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t groups_per_state = elements_per_state >> 2;
    const size_t total_groups = handle->batchSize * groups_per_state;
    if (total_groups == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_groups + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_SWAP_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       qubitA,
                       qubitB,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplyCRX(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned controlQubit,
                           unsigned targetQubit,
                           double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_i_s = make_complex(0.0, -std::sin(half));
    return launch_controlled_single_qubit_matrix(handle, state, numQubits, controlQubit, targetQubit,
                                                 c, minus_i_s, minus_i_s, c);
}

rocqStatus_t rocsvApplyCRY(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned controlQubit,
                           unsigned targetQubit,
                           double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex c = make_complex(std::cos(half), 0.0);
    const rocComplex minus_s = make_complex(-std::sin(half), 0.0);
    const rocComplex plus_s = make_complex(std::sin(half), 0.0);
    return launch_controlled_single_qubit_matrix(handle, state, numQubits, controlQubit, targetQubit,
                                                 c, minus_s, plus_s, c);
}

rocqStatus_t rocsvApplyCRZ(rocsvHandle_t handle,
                           rocComplex* d_state,
                           unsigned numQubits,
                           unsigned controlQubit,
                           unsigned targetQubit,
                           double theta) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const double half = theta / 2.0;
    const rocComplex phase_pos = make_complex(std::cos(half), std::sin(half));
    const rocComplex phase_neg = make_complex(std::cos(half), -std::sin(half));
    const rocComplex zero = make_complex(0.0, 0.0);
    return launch_controlled_single_qubit_matrix(handle, state, numQubits, controlQubit, targetQubit,
                                                 phase_neg, zero, zero, phase_pos);
}

// --- Multi-controlled gates --------------------------------------------------

rocqStatus_t rocsvApplyMultiControlledX(rocsvHandle_t handle,
                                        rocComplex* d_state,
                                        unsigned numQubits,
                                        const unsigned* controlQubits,
                                        unsigned numControlQubits,
                                        unsigned targetQubit) {
    if (!handle || !controlQubits || numControlQubits == 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(targetQubit, numQubits)) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (numControlQubits > 63) {
        return ROCQ_STATUS_NOT_IMPLEMENTED; // Needs 64-bit mask extension.
    }

    unsigned long long mask = 0ULL;
    for (unsigned i = 0; i < numControlQubits; ++i) {
        unsigned ctrl = controlQubits[i];
        if (!validate_qubit_index(ctrl, numQubits) || ctrl == targetQubit) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        mask |= (1ULL << ctrl);
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 2) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_multi_controlled_x_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       mask,
                       targetQubit,
                       handle->batchSize);
    return check_last_hip_error();
}

rocqStatus_t rocsvApplyCSWAP(rocsvHandle_t handle,
                             rocComplex* d_state,
                             unsigned numQubits,
                             unsigned controlQubit,
                             unsigned targetQubit1,
                             unsigned targetQubit2) {
    if (!handle) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    if (!validate_qubit_index(controlQubit, numQubits) ||
        !validate_qubit_index(targetQubit1, numQubits) ||
        !validate_qubit_index(targetQubit2, numQubits) ||
        controlQubit == targetQubit1 ||
        controlQubit == targetQubit2 ||
        targetQubit1 == targetQubit2) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t elements_per_state = 1ULL << numQubits;
    if (elements_per_state < 8) {
        return ROCQ_STATUS_SUCCESS;
    }

    const size_t total_states = handle->batchSize * elements_per_state;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((total_states + threads_per_block - 1) / threads_per_block);

    hipLaunchKernelGGL(apply_CSWAP_kernel,
                       dim3(blocks > 0 ? blocks : 1),
                       dim3(threads_per_block),
                       0,
                       handle->streams[0],
                       state,
                       numQubits,
                       controlQubit,
                       targetQubit1,
                       targetQubit2,
                       handle->batchSize);
    return check_last_hip_error();
}

// --- State-vector readback helpers ------------------------------------------

rocqStatus_t rocsvGetStateVectorFull(rocsvHandle_t handle,
                                     rocComplex* d_state,
                                     rocComplex* h_state) {
    if (!handle || !h_state) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;
    const size_t elements_per_state = 1ULL << handle->numQubits;
    const size_t total_elements = handle->batchSize * elements_per_state;
    if (hipMemcpy(h_state,
                  state,
                  total_elements * sizeof(rocComplex),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocsvGetStateVectorSlice(rocsvHandle_t handle,
                                      rocComplex* d_state,
                                      rocComplex* h_state,
                                      unsigned batch_index) {
    if (!handle || !h_state) return ROCQ_STATUS_INVALID_VALUE;
    rocComplex* state = resolve_state_pointer(handle, d_state);
    if (!state) return ROCQ_STATUS_INVALID_VALUE;

    const size_t elements_per_state = 1ULL << handle->numQubits;
    if (batch_index >= handle->batchSize) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    const size_t offset = batch_index * elements_per_state;
    const rocComplex* slice_ptr = state + offset;
    if (hipMemcpy(h_state,
                  slice_ptr,
                  elements_per_state * sizeof(rocComplex),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return ROCQ_STATUS_HIP_ERROR;
    }
    return ROCQ_STATUS_SUCCESS;
}
