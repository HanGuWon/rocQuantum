// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

#include "hipDensityMat.hpp"
#include "hipDensityMat_internal.hpp"

#include <hip/hip_complex.h>
#include <stdexcept>
#include <cstdint>
#include <cmath> // For sqrt
#include <vector> // For host-side reduction

// Helper to check HIP API calls for errors.
inline hipDensityMatStatus_t check_hip_error(hipError_t err) {
    if (err != hipSuccess) {
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

/**
 * @brief GPU kernel to apply a single-qubit Kraus operator: ρ' = KρK†.
 */
__global__ void apply_single_qubit_kraus_kernel(
    hipComplex* rho_out,
    const hipComplex* rho_in,
    const hipComplex* K,
    int num_qubits,
    int target_qubit)
{
    const int64_t dim = 1LL << num_qubits;
    const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dim || col >= dim) return;

    const int64_t low_mask = (1LL << target_qubit) - 1;
    const int64_t high_mask = ~((1LL << (target_qubit + 1)) - 1);
    const int64_t row_low = row & low_mask, col_low = col & low_mask;
    const int64_t row_high = row & high_mask, col_high = col & high_mask;
    const int row_t = (row >> target_qubit) & 1, col_t = (col >> target_qubit) & 1;

    const int64_t k0 = row_high | (0 << target_qubit) | row_low;
    const int64_t k1 = row_high | (1 << target_qubit) | row_low;
    const int64_t l0 = col_high | (0 << target_qubit) | col_low;
    const int64_t l1 = col_high | (1 << target_qubit) | col_low;

    hipComplex result = make_hipFloatComplex(0.0f, 0.0f);
    for (int kt = 0; kt < 2; ++kt) {
        for (int lt = 0; lt < 2; ++lt) {
            const int64_t k = (kt == 0) ? k0 : k1;
            const int64_t l = (lt == 0) ? l0 : l1;
            hipComplex K_rowt_kt = K[row_t * 2 + kt];
            hipComplex rho_kl = rho_in[k * dim + l];
            hipComplex K_colt_lt_conj = hipConjf(K[col_t * 2 + lt]);
            hipComplex term = hipCmulf(K_rowt_kt, rho_kl);
            term = hipCmulf(term, K_colt_lt_conj);
            result = hipCaddf(result, term);
        }
    }
    rho_out[row * dim + col] = result;
}

/**
 * @brief GPU kernel for element-wise addition: target += source.
 */
__global__ void accumulate_kernel(hipComplex* target, const hipComplex* source, int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        target[idx] = hipCaddf(target[idx], source[idx]);
    }
}

/**
 * @brief GPU kernel to compute partial sums for Tr(Oρ) for a single block.
 */
__global__ void expectation_value_kernel(
    const hipComplex* rho,
    double* partial_sums,
    int num_qubits,
    int target_qubit,
    hipDensityMatPauli_t pauli_op)
{
    extern __shared__ double sdata[];

    const int64_t dim = 1LL << num_qubits;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const int64_t i_start = blockIdx.x * block_size + tid;
    const int64_t stride = gridDim.x * block_size;

    double thread_sum = 0.0;

    for (int64_t i = i_start; i < dim; i += stride) {
        double term = 0.0;
        int64_t bit_mask = 1LL << target_qubit;
        int64_t i_flipped = i ^ bit_mask;

        switch (pauli_op) {
            case HIPDENSITYMAT_PAULI_Z: {
                double sign = ((i & bit_mask) == 0) ? 1.0 : -1.0;
                term = sign * rho[i * dim + i].x;
                break;
            }
            case HIPDENSITYMAT_PAULI_X: {
                term = rho[i_flipped * dim + i].x;
                break;
            }
            case HIPDENSITYMAT_PAULI_Y: {
                double sign = ((i & bit_mask) == 0) ? 1.0 : -1.0;
                term = -sign * rho[i_flipped * dim + i].y;
                break;
            }
        }
        thread_sum += term;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}


hipDensityMatStatus_t hipDensityMatCreateState(hipDensityMatState_t* state, int num_qubits) {
    if (state == nullptr || num_qubits <= 0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    
    hipDensityMatState* internal_state = nullptr;
    try {
        internal_state = new hipDensityMatState();
    } catch (const std::bad_alloc&) {
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    internal_state->num_qubits_ = num_qubits;
    int64_t dim = 1LL << num_qubits;
    internal_state->num_elements_ = dim * dim;
    size_t size_bytes = internal_state->num_elements_ * sizeof(hipComplex);

    if (hipMalloc(&internal_state->device_data_, size_bytes) != hipSuccess) {
        delete internal_state;
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }
    if (hipMemset(internal_state->device_data_, 0, size_bytes) != hipSuccess) {
        hipFree(internal_state->device_data_);
        delete internal_state;
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    hipComplex val_one = make_hipFloatComplex(1.0f, 0.0f);
    if (hipMemcpy(internal_state->device_data_, &val_one, sizeof(hipComplex), hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(internal_state->device_data_);
        delete internal_state;
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }
    internal_state->stream_ = 0;
    *state = internal_state;
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t hipDensityMatDestroyState(hipDensityMatState_t state) {
    if (state == nullptr) return HIPDENSITYMAT_STATUS_SUCCESS;
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (internal_state->device_data_ != nullptr) {
        hipFree(internal_state->device_data_);
    }
    delete internal_state;
    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t hipDensityMatApplyKrausOperator(
    hipDensityMatState_t state,
    int target_qubit,
    const hipComplex* kraus_matrix_host)
{
    if (state == nullptr || kraus_matrix_host == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    const int64_t dim = 1LL << internal_state->num_qubits_;
    const size_t size_bytes = internal_state->num_elements_ * sizeof(hipComplex);
    
    hipComplex* kraus_matrix_device = nullptr;
    hipComplex* rho_out_device = nullptr;
    hipError_t hip_err;

    hip_err = hipMalloc(&kraus_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMalloc(&rho_out_device, size_bytes);
    if (hip_err != hipSuccess) {
        hipFree(kraus_matrix_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMemcpy(kraus_matrix_device, kraus_matrix_host, 4 * sizeof(hipComplex), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) {
        hipFree(kraus_matrix_device);
        hipFree(rho_out_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((dim + blockDim.x - 1) / blockDim.x, (dim + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim, blockDim, 0, internal_state->stream_,
        rho_out_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, rho_out_device, size_bytes, hipMemcpyDeviceToDevice);
    
    hipFree(kraus_matrix_device);
    hipFree(rho_out_device);
    return check_hip_error(hip_err);
}

hipDensityMatStatus_t hipDensityMatApplyBitFlipChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (probability < 0.0 || probability > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    const int64_t dim = 1LL << internal_state->num_qubits_;
    const size_t size_bytes = internal_state->num_elements_ * sizeof(hipComplex);
    hipError_t hip_err;

    hipComplex* accumulator_rho_device = nullptr;
    hipComplex* temp_rho_device = nullptr;
    hipComplex* kraus_matrix_device = nullptr;

    hip_err = hipMalloc(&accumulator_rho_device, size_bytes);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMalloc(&temp_rho_device, size_bytes);
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }
    hip_err = hipMalloc(&kraus_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); hipFree(temp_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }

    hipMemset(accumulator_rho_device, 0, size_bytes);

    float p0 = sqrt(1.0 - probability);
    hipComplex K0[4] = { make_hipFloatComplex(p0, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
                         make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p0, 0.0f) };
    
    hipMemcpy(kraus_matrix_device, K0, sizeof(K0), hipMemcpyHostToDevice);
    
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((dim + blockDim2D.x - 1) / blockDim2D.x, (dim + blockDim2D.y - 1) / blockDim2D.y);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);

    dim3 blockDim1D(256);
    dim3 gridDim1D((internal_state->num_elements_ + blockDim1D.x - 1) / blockDim1D.x);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    float p1 = sqrt(probability);
    hipComplex K1[4] = { make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p1, 0.0f),
                         make_hipFloatComplex(p1, 0.0f), make_hipFloatComplex(0.0f, 0.0f) };

    hipMemcpy(kraus_matrix_device, K1, sizeof(K1), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, accumulator_rho_device, size_bytes, hipMemcpyDeviceToDevice);

    hipFree(accumulator_rho_device);
    hipFree(temp_rho_device);
    hipFree(kraus_matrix_device);

    return check_hip_error(hip_err);
}

hipDensityMatStatus_t hipDensityMatApplyPhaseFlipChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (probability < 0.0 || probability > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    const int64_t dim = 1LL << internal_state->num_qubits_;
    const size_t size_bytes = internal_state->num_elements_ * sizeof(hipComplex);
    hipError_t hip_err;

    hipComplex* accumulator_rho_device = nullptr;
    hipComplex* temp_rho_device = nullptr;
    hipComplex* kraus_matrix_device = nullptr;

    hip_err = hipMalloc(&accumulator_rho_device, size_bytes);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMalloc(&temp_rho_device, size_bytes);
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }
    hip_err = hipMalloc(&kraus_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); hipFree(temp_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }

    hipMemset(accumulator_rho_device, 0, size_bytes);

    float p0 = sqrt(1.0 - probability);
    hipComplex K0[4] = { make_hipFloatComplex(p0, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
                         make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p0, 0.0f) };
    
    hipMemcpy(kraus_matrix_device, K0, sizeof(K0), hipMemcpyHostToDevice);
    
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((dim + blockDim2D.x - 1) / blockDim2D.x, (dim + blockDim2D.y - 1) / blockDim2D.y);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);

    dim3 blockDim1D(256);
    dim3 gridDim1D((internal_state->num_elements_ + blockDim1D.x - 1) / blockDim1D.x);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    float p1 = sqrt(probability);
    hipComplex K1[4] = { make_hipFloatComplex(p1, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
                         make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(-p1, 0.0f) };

    hipMemcpy(kraus_matrix_device, K1, sizeof(K1), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, accumulator_rho_device, size_bytes, hipMemcpyDeviceToDevice);

    hipFree(accumulator_rho_device);
    hipFree(temp_rho_device);
    hipFree(kraus_matrix_device);

    return check_hip_error(hip_err);
}

hipDensityMatStatus_t hipDensityMatApplyDepolarizingChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double probability)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (probability < 0.0 || probability > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    const int64_t dim = 1LL << internal_state->num_qubits_;
    const size_t size_bytes = internal_state->num_elements_ * sizeof(hipComplex);
    hipError_t hip_err;

    hipComplex* accumulator_rho_device = nullptr;
    hipComplex* temp_rho_device = nullptr;
    hipComplex* kraus_matrix_device = nullptr;

    hip_err = hipMalloc(&accumulator_rho_device, size_bytes);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMalloc(&temp_rho_device, size_bytes);
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }
    hip_err = hipMalloc(&kraus_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); hipFree(temp_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }

    hipMemset(accumulator_rho_device, 0, size_bytes);

    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((dim + blockDim2D.x - 1) / blockDim2D.x, (dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 blockDim1D(256);
    dim3 gridDim1D((internal_state->num_elements_ + blockDim1D.x - 1) / blockDim1D.x);

    float p0_factor = sqrt(1.0 - probability);
    hipComplex K0[4] = { make_hipFloatComplex(p0_factor, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
                         make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p0_factor, 0.0f) };
    hipMemcpy(kraus_matrix_device, K0, sizeof(K0), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    float p_factor = sqrt(probability / 3.0);
    
    hipComplex K1[4] = { make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(p_factor, 0.0f),
                         make_hipFloatComplex(p_factor, 0.0f), make_hipFloatComplex(0.0f, 0.0f) };
    hipMemcpy(kraus_matrix_device, K1, sizeof(K1), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    hipComplex K2[4] = { make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(0.0f, -p_factor),
                         make_hipFloatComplex(0.0f, p_factor), make_hipFloatComplex(0.0f, 0.0f) };
    hipMemcpy(kraus_matrix_device, K2, sizeof(K2), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    hipComplex K3[4] = { make_hipFloatComplex(p_factor, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
                         make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(-p_factor, 0.0f) };
    hipMemcpy(kraus_matrix_device, K3, sizeof(K3), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, accumulator_rho_device, size_bytes, hipMemcpyDeviceToDevice);

    hipFree(accumulator_rho_device);
    hipFree(temp_rho_device);
    hipFree(kraus_matrix_device);

    return check_hip_error(hip_err);
}

hipDensityMatStatus_t hipDensityMatComputeExpectation(
    hipDensityMatState_t state,
    int target_qubit,
    hipDensityMatPauli_t pauli_op,
    double* result_host)
{
    if (state == nullptr || result_host == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    const int64_t dim = 1LL << internal_state->num_qubits_;
    hipError_t hip_err;

    const int block_size = 256;
    const int num_blocks = std::min((int)((dim + block_size - 1) / block_size), 2048);
    
    double* partial_sums_device = nullptr;
    size_t partial_sums_size = num_blocks * sizeof(double);
    hip_err = hipMalloc(&partial_sums_device, partial_sums_size);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;

    hipLaunchKernelGGL(expectation_value_kernel,
        num_blocks,
        block_size,
        block_size * sizeof(double),
        internal_state->stream_,
        static_cast<const hipComplex*>(internal_state->device_data_),
        partial_sums_device,
        internal_state->num_qubits_,
        target_qubit,
        pauli_op);

    hip_err = hipGetLastError();
    if (hip_err != hipSuccess) {
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err != hipSuccess) {
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    std::vector<double> partial_sums_host(num_blocks);
    hip_err = hipMemcpy(partial_sums_host.data(), partial_sums_device, partial_sums_size, hipMemcpyDeviceToHost);
    if (hip_err != hipSuccess) {
        hipFree(partial_sums_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += partial_sums_host[i];
    }
    
    *result_host = total_sum;

    hipFree(partial_sums_device);

    return HIPDENSITYMAT_STATUS_SUCCESS;
}

hipDensityMatStatus_t hipDensityMatApplyAmplitudeDampingChannel(
    hipDensityMatState_t state,
    int target_qubit,
    double gamma)
{
    if (state == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    if (gamma < 0.0 || gamma > 1.0) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    const int64_t dim = 1LL << internal_state->num_qubits_;
    const size_t size_bytes = internal_state->num_elements_ * sizeof(hipComplex);
    hipError_t hip_err;

    hipComplex* accumulator_rho_device = nullptr;
    hipComplex* temp_rho_device = nullptr;
    hipComplex* kraus_matrix_device = nullptr;

    hip_err = hipMalloc(&accumulator_rho_device, size_bytes);
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMalloc(&temp_rho_device, size_bytes);
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }
    hip_err = hipMalloc(&kraus_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) { hipFree(accumulator_rho_device); hipFree(temp_rho_device); return HIPDENSITYMAT_STATUS_ALLOC_FAILED; }

    hipMemset(accumulator_rho_device, 0, size_bytes);

    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((dim + blockDim2D.x - 1) / blockDim2D.x, (dim + blockDim2D.y - 1) / blockDim2D.y);
    dim3 blockDim1D(256);
    dim3 gridDim1D((internal_state->num_elements_ + blockDim1D.x - 1) / blockDim1D.x);

    hipComplex K0[4] = { make_hipFloatComplex(1.0f, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
                         make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(sqrt(1.0 - gamma), 0.0f) };
    
    hipMemcpy(kraus_matrix_device, K0, sizeof(K0), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    hipComplex K1[4] = { make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(sqrt(gamma), 0.0f),
                         make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(0.0f, 0.0f) };

    hipMemcpy(kraus_matrix_device, K1, sizeof(K1), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim2D, blockDim2D, 0, internal_state->stream_,
        temp_rho_device, static_cast<const hipComplex*>(internal_state->device_data_),
        kraus_matrix_device, internal_state->num_qubits_, target_qubit);
    hipLaunchKernelGGL(accumulate_kernel, gridDim1D, blockDim1D, 0, internal_state->stream_,
        accumulator_rho_device, temp_rho_device, internal_state->num_elements_);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, accumulator_rho_device, size_bytes, hipMemcpyDeviceToDevice);

    hipFree(accumulator_rho_device);
    hipFree(temp_rho_device);
    hipFree(kraus_matrix_device);

    return check_hip_error(hip_err);
}

hipDensityMatStatus_t hipDensityMatApplyGate(
    hipDensityMatState_t state,
    int target_qubit,
    const hipComplex* gate_matrix_host)
{
    if (state == nullptr || gate_matrix_host == nullptr) return HIPDENSITYMAT_STATUS_INVALID_VALUE;
    
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    if (target_qubit < 0 || target_qubit >= internal_state->num_qubits_) return HIPDENSITYMAT_STATUS_INVALID_VALUE;

    const int64_t dim = 1LL << internal_state->num_qubits_;
    const size_t size_bytes = internal_state->num_elements_ * sizeof(hipComplex);
    
    hipComplex* gate_matrix_device = nullptr;
    hipComplex* rho_out_device = nullptr;
    hipError_t hip_err;

    hip_err = hipMalloc(&gate_matrix_device, 4 * sizeof(hipComplex));
    if (hip_err != hipSuccess) return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    hip_err = hipMalloc(&rho_out_device, size_bytes);
    if (hip_err != hipSuccess) {
        hipFree(gate_matrix_device);
        return HIPDENSITYMAT_STATUS_ALLOC_FAILED;
    }

    hip_err = hipMemcpy(gate_matrix_device, gate_matrix_host, 4 * sizeof(hipComplex), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) {
        hipFree(gate_matrix_device);
        hipFree(rho_out_device);
        return HIPDENSITYMAT_STATUS_EXECUTION_FAILED;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((dim + blockDim.x - 1) / blockDim.x, (dim + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(apply_single_qubit_kraus_kernel, gridDim, blockDim, 0, internal_state->stream_,
        rho_out_device, static_cast<const hipComplex*>(internal_state->device_data_),
        gate_matrix_device, internal_state->num_qubits_, target_qubit);

    hip_err = hipGetLastError();
    if (hip_err == hipSuccess) hip_err = hipStreamSynchronize(internal_state->stream_);
    if (hip_err == hipSuccess) hip_err = hipMemcpy(internal_state->device_data_, rho_out_device, size_bytes, hipMemcpyDeviceToDevice);
    
    hipFree(gate_matrix_device);
    hipFree(rho_out_device);
    return check_hip_error(hip_err);
}

hipDensityMatStatus_t hipDensityMatApplyChannel(hipDensityMatState_t state, int target_qubit, const void* channel_params) {
    return HIPDENSITYMAT_STATUS_NOT_IMPLEMENTED;
}
