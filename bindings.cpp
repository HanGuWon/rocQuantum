#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "rocqCompiler/MLIRCompiler.h"
#include "rocqCompiler/QuantumBackend.h"
#include "rocquantum/QuantumSimulator.h"

namespace py = pybind11;

PYBIND11_MODULE(rocquantum_bind, m) {
    m.doc() = "pybind11 plugin for rocQuantum-1";

    py::class_<rocq::MLIRCompiler>(m, "MLIRCompiler")
        .def(py::init([](unsigned num_qubits, const std::string& backend_name) {
            auto backend = rocq::create_backend(backend_name);
            return std::make_unique<rocq::MLIRCompiler>(num_qubits, std::move(backend));
        }))
        .def("compile_and_execute", 
             [](rocq::MLIRCompiler &self, const std::string &mlir, py::dict args) {
                 // ... (implementation from previous step)
                 return self.compile_and_execute(mlir, {});
        })
        // Bind the new emit_qir method
        .def("emit_qir", &rocq::MLIRCompiler::emit_qir,
             "Compiles the MLIR string down to QIR (LLVM IR).");

    auto simulator = py::class_<rocquantum::QuantumSimulator>(m, "QuantumSimulator");

    simulator
        .def(py::init<unsigned>(), py::arg("num_qubits"))
        .def("reset", &rocquantum::QuantumSimulator::reset)
        .def("apply_gate",
             [](rocquantum::QuantumSimulator& self,
                const std::string& gate_name,
                const std::vector<unsigned>& targets,
                const std::vector<double>& params) {
                 self.apply_gate(gate_name, targets, params);
             },
             py::arg("gate_name"),
             py::arg("targets"),
             py::arg("params") = std::vector<double>{})
        .def("apply_matrix",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                const std::vector<unsigned>& targets) {
                 if (matrix.ndim() != 2 || matrix.shape(0) != 2 || matrix.shape(1) != 2) {
                     throw std::invalid_argument("apply_matrix expects a 2x2 complex matrix.");
                 }
                 std::vector<std::complex<double>> host(4);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 self.apply_matrix(host, targets);
             },
             py::arg("matrix"),
             py::arg("targets"))
        .def("get_statevector",
             [](const rocquantum::QuantumSimulator& self) {
                 auto state = self.get_statevector();
                 py::array_t<std::complex<double>> result(state.size());
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             })
        .def("measure", &rocquantum::QuantumSimulator::measure, py::arg("qubits"), py::arg("shots"))
        .def("num_qubits", &rocquantum::QuantumSimulator::num_qubits)
        // Legacy-style bindings
        .def("ApplyGate",
             [](rocquantum::QuantumSimulator& self, const std::string& gate_name, int target_qubit) {
                 self.ApplyGate(gate_name, target_qubit);
             },
             py::arg("gate_name"),
             py::arg("target_qubit"))
        .def("ApplyGate",
             [](rocquantum::QuantumSimulator& self, const std::string& gate_name, int control_qubit, int target_qubit) {
                 self.ApplyGate(gate_name, control_qubit, target_qubit);
             },
             py::arg("gate_name"),
             py::arg("control_qubit"),
             py::arg("target_qubit"))
        .def("ApplyGate",
             [](rocquantum::QuantumSimulator& self,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrix,
                int target_qubit) {
                 if (matrix.ndim() != 2 || matrix.shape(0) != 2 || matrix.shape(1) != 2) {
                     throw std::invalid_argument("ApplyGate expects a 2x2 complex matrix.");
                 }
                 std::vector<std::complex<double>> host(4);
                 std::memcpy(host.data(), matrix.data(), host.size() * sizeof(std::complex<double>));
                 self.ApplyGate(host, target_qubit);
             },
             py::arg("gate_matrix"),
             py::arg("target_qubit"))
        .def("Execute", &rocquantum::QuantumSimulator::Execute)
        .def("GetStateVector",
             [](const rocquantum::QuantumSimulator& self) {
                 auto state = self.GetStateVector();
                 py::array_t<std::complex<double>> result(state.size());
                 std::memcpy(result.mutable_data(), state.data(), state.size() * sizeof(std::complex<double>));
                 return result;
             });

    // Historical alias for integrations/tests still referencing QSim
    m.attr("QSim") = m.attr("QuantumSimulator");
}
