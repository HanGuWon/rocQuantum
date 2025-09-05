#include <pybind11/pybind11>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector>
#include <stdexcept>

#include "hipDensityMat.hpp"

namespace py = pybind11;

// Helper to check HIP API calls for errors.
#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error("HIP error (" + std::to_string(err) + "): " + hipGetErrorString(err)); \
    } \
} while (0)

// Helper to check hipDensityMat API calls for errors.
#define HIPDENSITYMAT_CHECK(cmd) do { \
    hipDensityMatStatus_t status = cmd; \
    if (status != HIPDENSITYMAT_STATUS_SUCCESS) { \
        throw std::runtime_error("hipDensityMat error: " + std::to_string(status)); \
    } \
} while (0)


PYBIND11_MODULE(rocq_hip, m) {
    m.doc() = "Python bindings for the rocQuantum hipDensityMat library";

    py::class_<hipDensityMatState>(m, "DensityMatrixState")
        .def(py::init([](int num_qubits) {
            hipDensityMatState_t state_handle;
            HIPDENSITYMAT_CHECK(hipDensityMatCreateState(&state_handle, num_qubits));
            // We cast the opaque pointer to the internal struct pointer for pybind11
            return static_cast<hipDensityMatState*>(state_handle);
        }), py::arg("num_qubits"), "Create a new density matrix state.")
        .def("__del__", [](hipDensityMatState* state) {
            // The handle is the same as the pointer to the internal struct
            hipDensityMatDestroyState(state);
        })
        .def("apply_gate", [](hipDensityMatState* state, py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> gate_matrix, int target_qubit, bool adjoint) {
            if (gate_matrix.ndim() != 2 || gate_matrix.shape(0) != 2 || gate_matrix.shape(1) != 2) {
                throw std::invalid_argument("Gate matrix must be a 2x2 NumPy array.");
            }
            
            auto mat_unchecked = gate_matrix.unchecked<2>();
            hipComplex gate_host[4];
            gate_host[0] = reinterpret_cast<hipComplex(&)>(mat_unchecked(0, 0));
            gate_host[1] = reinterpret_cast<hipComplex(&)>(mat_unchecked(0, 1));
            gate_host[2] = reinterpret_cast<hipComplex(&)>(mat_unchecked(1, 0));
            gate_host[3] = reinterpret_cast<hipComplex(&)>(mat_unchecked(1, 1));

            if (adjoint) {
                // Conjugate transpose
                gate_host[1] = hipConjf(gate_host[1]);
                gate_host[2] = hipConjf(gate_host[2]);
                std::swap(gate_host[1], gate_host[2]);
            }

            HIPDENSITYMAT_CHECK(hipDensityMatApplyGate(state, target_qubit, gate_host));
        }, py::arg("gate_matrix"), py::arg("target_qubit"), py::arg("adjoint") = false, "Apply a single-qubit gate. If adjoint is true, applies the conjugate transpose.")
        .def("apply_cnot", [](hipDensityMatState* state, int control_qubit, int target_qubit) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyCNOT(state, control_qubit, target_qubit));
        }, py::arg("control_qubit"), py::arg("target_qubit"), "Apply a CNOT gate.")
        .def("apply_controlled_gate", [](hipDensityMatState* state, py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> gate_matrix, int control_qubit, int target_qubit) {
            if (gate_matrix.ndim() != 2 || gate_matrix.shape(0) != 2 || gate_matrix.shape(1) != 2) {
                throw std::invalid_argument("Gate matrix must be a 2x2 NumPy array.");
            }
            
            hipComplex* gate_matrix_device = nullptr;
            HIP_CHECK(hipMalloc(&gate_matrix_device, 4 * sizeof(hipComplex)));
            HIP_CHECK(hipMemcpy(gate_matrix_device, gate_matrix.data(), 4 * sizeof(hipComplex), hipMemcpyHostToDevice));
            
            hipDensityMatStatus_t status = hipDensityMatApplyControlledGate(state, control_qubit, target_qubit, gate_matrix_device);
            
            HIP_CHECK(hipFree(gate_matrix_device));
            HIPDENSITYMAT_CHECK(status);
        }, py::arg("gate_matrix"), py::arg("control_qubit"), py::arg("target_qubit"), "Apply a controlled single-qubit gate.")
        .def("compute_expectation", [](hipDensityMatState* state, hipDensityMatPauli_t pauli_op, int target_qubit) {
            double result;
            HIPDENSITYMAT_CHECK(hipDensityMatComputeExpectation(state, target_qubit, pauli_op, &result));
            return result;
        }, py::arg("pauli_op"), py::arg("target_qubit"), "Compute the expectation value of a single-qubit Pauli observable.")
        .def("_compute_z_product_expectation", [](hipDensityMatState* state, std::vector<int> z_qubit_indices) {
            double result;
            HIPDENSITYMAT_CHECK(hipDensityMatComputePauliZProductExpectation(state, z_qubit_indices.size(), z_qubit_indices.data(), &result));
            return result;
        }, py::arg("z_qubit_indices"), "Compute expectation of a Pauli Z product.")
        .def("apply_bit_flip_channel", [](hipDensityMatState* state, int target_qubit, double probability) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyBitFlipChannel(state, target_qubit, probability));
        }, py::arg("target_qubit"), py::arg("probability"), "Apply a bit-flip noise channel.")
        .def("apply_depolarizing_channel", [](hipDensityMatState* state, int target_qubit, double probability) {
            HIPDENSITYMAT_CHECK(hipDensityMatApplyDepolarizingChannel(state, target_qubit, probability));
        }, py::arg("target_qubit"), py::arg("probability"), "Apply a depolarizing noise channel.");

    py::enum_<hipDensityMatPauli_t>(m, "Pauli")
        .value("X", HIPDENSITYMAT_PAULI_X)
        .value("Y", HIPDENSITYMAT_PAULI_Y)
        .value("Z", HIPDENSITYMAT_PAULI_Z)
        .export_values();
}
