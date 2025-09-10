#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "rocqCompiler/MLIRCompiler.h"
#include "rocqCompiler/QuantumBackend.h"

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
}
