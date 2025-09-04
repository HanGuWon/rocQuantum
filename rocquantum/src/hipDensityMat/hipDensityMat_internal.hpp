// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifndef HIPDENSITYMAT_INTERNAL_HPP
#define HIPDENSITYMAT_INTERNAL_HPP

#include <hip/hip_runtime.h>
#include <cstdint> // For int64_t

/**
 * @brief Internal C++ structure that the opaque hipDensityMatState_t handle points to.
 *
 * This class holds all the necessary information for managing a density matrix
 * on the GPU.
 */
struct hipDensityMatState {
    // The number of qubits represented by the density matrix.
    int num_qubits_;

    // The total number of complex elements in the density matrix.
    // For N qubits, this is (2^N * 2^N) = 2^(2N).
    int64_t num_elements_;

    // A void pointer to the allocated device memory for the density matrix.
    // This will be cast to the appropriate complex type (e.g., hipComplex*).
    void* device_data_;

    // HIP stream for asynchronous execution of kernels.
    hipStream_t stream_;
};

#endif // HIPDENSITYMAT_INTERNAL_HPP
