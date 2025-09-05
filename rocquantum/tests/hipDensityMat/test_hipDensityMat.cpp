#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector>
#include "hipDensityMat.hpp"
#include "hipDensityMat_internal.hpp" // For accessing internal state for verification

// Helper function to copy density matrix from device to host for verification
void get_density_matrix_from_device(hipDensityMatState_t state, std::vector<hipComplex>& host_matrix) {
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    const int64_t num_elements = internal_state->num_elements_;
    host_matrix.resize(num_elements);
    hipMemcpy(host_matrix.data(), internal_state->device_data_, num_elements * sizeof(hipComplex), hipMemcpyDeviceToHost);
}

// Helper function to set the density matrix on the device from a host matrix
void set_density_matrix_on_device(hipDensityMatState_t state, const std::vector<hipComplex>& host_matrix) {
    hipDensityMatState* internal_state = static_cast<hipDensityMatState*>(state);
    const int64_t num_elements = internal_state->num_elements_;
    hipMemcpy(internal_state->device_data_, host_matrix.data(), num_elements * sizeof(hipComplex), hipMemcpyHostToDevice);
}

TEST(HipDensityMatCNOT, CNOT_FlipsTargetWhenControlIsOne) {
    const int num_qubits = 2;
    const int64_t dim = 1 << num_qubits;
    const int64_t num_elements = dim * dim;

    hipDensityMatState_t state;
    hipDensityMatStatus_t status = hipDensityMatCreateState(&state, num_qubits);
    ASSERT_EQ(status, HIPDENSITYMAT_STATUS_SUCCESS);

    // Initialize state to |10><10|
    // |10> is basis state 2. The density matrix is rho[2][2] = 1.
    std::vector<hipComplex> host_rho(num_elements, make_hipFloatComplex(0.0f, 0.0f));
    host_rho[2 * dim + 2] = make_hipFloatComplex(1.0f, 0.0f);

    set_density_matrix_on_device(state, host_rho);

    // Apply CNOT(0, 1)
    status = hipDensityMatApplyCNOT(state, 0, 1);
    ASSERT_EQ(status, HIPDENSITYMAT_STATUS_SUCCESS);

    // Get result back
    std::vector<hipComplex> result_rho(num_elements);
    get_density_matrix_from_device(state, result_rho);

    // Expected state is |11><11|
    // |11> is basis state 3. The density matrix should have rho[3][3] = 1.
    for (int64_t i = 0; i < num_elements; ++i) {
        if (i == (3 * dim + 3)) {
            EXPECT_FLOAT_EQ(result_rho[i].x, 1.0f);
            EXPECT_FLOAT_EQ(result_rho[i].y, 0.0f);
        } else {
            EXPECT_FLOAT_EQ(result_rho[i].x, 0.0f);
            EXPECT_FLOAT_EQ(result_rho[i].y, 0.0f);
        }
    }

    hipDensityMatDestroyState(state);
}

TEST(HipDensityMatCNOT, CNOT_DoesNothingWhenControlIsZero) {
    const int num_qubits = 2;
    const int64_t dim = 1 << num_qubits;
    const int64_t num_elements = dim * dim;

    hipDensityMatState_t state;
    hipDensityMatStatus_t status = hipDensityMatCreateState(&state, num_qubits);
    ASSERT_EQ(status, HIPDENSITYMAT_STATUS_SUCCESS);

    // Initialize state to |01><01|
    // |01> is basis state 1. The density matrix is rho[1][1] = 1.
    std::vector<hipComplex> host_rho(num_elements, make_hipFloatComplex(0.0f, 0.0f));
    host_rho[1 * dim + 1] = make_hipFloatComplex(1.0f, 0.0f);

    set_density_matrix_on_device(state, host_rho);

    // Apply CNOT(0, 1)
    status = hipDensityMatApplyCNOT(state, 0, 1);
    ASSERT_EQ(status, HIPDENSITYMAT_STATUS_SUCCESS);

    // Get result back
    std::vector<hipComplex> result_rho(num_elements);
    get_density_matrix_from_device(state, result_rho);

    // Expected state is |01><01| (no change)
    for (int64_t i = 0; i < num_elements; ++i) {
        if (i == (1 * dim + 1)) {
            EXPECT_FLOAT_EQ(result_rho[i].x, 1.0f);
            EXPECT_FLOAT_EQ(result_rho[i].y, 0.0f);
        } else {
            EXPECT_FLOAT_EQ(result_rho[i].x, 0.0f);
            EXPECT_FLOAT_EQ(result_rho[i].y, 0.0f);
        }
    }

    hipDensityMatDestroyState(state);
}

TEST(HipDensityMatControlledGate, CZGateOnPlusPlusState) {
    const int num_qubits = 2;
    const int control_qubit = 0;
    const int target_qubit = 1;
    const int64_t dim = 1 << num_qubits;
    const int64_t num_elements = dim * dim;

    hipDensityMatState_t state;
    hipDensityMatStatus_t status = hipDensityMatCreateState(&state, num_qubits);
    ASSERT_EQ(status, HIPDENSITYMAT_STATUS_SUCCESS);

    // Initialize state to |+>|+>
    // |+> = 1/sqrt(2) * (|0> + |1>)
    // |+>|+> = 0.5 * (|00> + |01> + |10> + |11>)
    // The density matrix rho = |psi><psi| is a 4x4 matrix with all elements equal to 0.25
    std::vector<hipComplex> host_rho(num_elements, make_hipFloatComplex(0.25f, 0.0f));
    set_density_matrix_on_device(state, host_rho);

    // Define Z gate matrix on host
    hipComplex z_gate_host[4] = {
        make_hipFloatComplex(1.0f, 0.0f), make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f), make_hipFloatComplex(-1.0f, 0.0f)
    };

    // Allocate and copy Z gate to device
    hipComplex* z_gate_device;
    hipError_t hip_err = hipMalloc(&z_gate_device, 4 * sizeof(hipComplex));
    ASSERT_EQ(hip_err, hipSuccess);
    hip_err = hipMemcpy(z_gate_device, z_gate_host, 4 * sizeof(hipComplex), hipMemcpyHostToDevice);
    ASSERT_EQ(hip_err, hipSuccess);

    // Apply Controlled-Z gate
    status = hipDensityMatApplyControlledGate(state, control_qubit, target_qubit, z_gate_device);
    ASSERT_EQ(status, HIPDENSITYMAT_STATUS_SUCCESS);

    // Get result back
    std::vector<hipComplex> result_rho(num_elements);
    get_density_matrix_from_device(state, result_rho);

    // Expected state after CZ on |+>|+> is 0.5 * (|00> + |01> + |10> - |11>)
    // Expected rho' = |psi'><psi'| where |psi'> = 0.5 * (|0>+|1>+|2>-|3>).
    // rho' = 0.25 * [[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float expected_val = 0.25f;
            bool i_is_11 = (i == 3);
            bool j_is_11 = (j == 3);
            if (i_is_11 != j_is_11) { // XOR logic for sign flip
                expected_val = -0.25f;
            }
            EXPECT_NEAR(result_rho[i * dim + j].x, expected_val, 1e-6);
            EXPECT_NEAR(result_rho[i * dim + j].y, 0.0f, 1e-6);
        }
    }

    // Cleanup
    hipFree(z_gate_device);
    hipDensityMatDestroyState(state);
}