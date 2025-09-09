// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "rocquantum.h"

namespace py = pybind11;

// Helper function to convert a NumPy array to std::vector<hipComplex>
std::vector<hipComplex> numpy_to_hipComplex_vector(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 2 || arr.shape(0) != 2 || arr.shape(1) != 2) {
        throw std::runtime_error("Input must be a 2x2 NumPy array for a single-qubit gate.");
    }
    std::vector<hipComplex> vec(arr.size());
    auto r = arr.unchecked<2>();
    for (py::ssize_t i = 0; i < r.shape(0); i++) {
        for (py::ssize_t j = 0; j < r.shape(1); j++) {
            vec[i * r.shape(1) + j] = hipComplex(r(i, j).real(), r(i, j).imag());
        }
    }
    return vec;
}

PYBIND11_MODULE(rocquantum_bind, m) {
    m.doc() = "Python bindings for the rocQuantum C++ simulator library";

    py::class_<rocquantum::QSim>(m, "QSim")
        .def(py::init<int>(), py::arg("num_qubits"), "Simulator constructor")
        .def("ApplyGate",
             py::overload_cast<const std::string&, int>(&rocquantum::QSim::ApplyGate),
             "Applies a named single-qubit gate.",
             py::arg("gate_name"), py::arg("target_qubit"))
        .def("ApplyGate",
             py::overload_cast<const std::string&, int, int>(&rocquantum::QSim::ApplyGate),
             "Applies a named two-qubit gate.",
             py::arg("gate_name"), py::arg("control_qubit"), py::arg("target_qubit"))
        .def("ApplyGate",
             [](rocquantum::QSim &self, py::array_t<std::complex<double>> gate_matrix, int target_qubit) {
                 std::vector<hipComplex> matrix_vec = numpy_to_hipComplex_vector(gate_matrix);
                 self.ApplyGate(matrix_vec, target_qubit);
             },
             "Applies a custom gate from a NumPy array.",
             py::arg("gate_matrix"), py::arg("target_qubit"))
        .def("Execute", &rocquantum::QSim::Execute, "Executes the circuit")
        .def("GetStateVector",
             [](const rocquantum::QSim &qsim) {
                 std::vector<hipComplex> state_vec = qsim.GetStateVector();
                 // Create a Python list of complex numbers
                 py::list py_list;
                 for (const auto& c : state_vec) {
                     py_list.append(std::complex<double>(c.x, c.y));
                 }
                 // Convert the list to a NumPy array
                 return py::array_t<std::complex<double>>(py_list);
             },
             "Returns the final state vector as a NumPy array.");
}
