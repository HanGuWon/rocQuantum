#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rocquantum/include/rocquantum.h"

namespace py = pybind11;

// Helper function to convert numpy matrix to std::vector
std::vector<std::complex<double>> numpy_to_std_vector(py::array_t<std::complex<double>> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("NumPy array must be a 2D matrix");
    }
    std::complex<double> *ptr = static_cast<std::complex<double> *>(buf.ptr);
    return std::vector<std::complex<double>>(ptr, ptr + buf.size);
}

PYBIND11_MODULE(rocquantum_bind, m) {
    m.doc() = "Python bindings for the rocQuantum C++/HIP simulator";

    py::class_<rocquantum::QuantumSimulator>(m, "QuantumSimulator")
        .def(py::init<unsigned>(), py::arg("num_qubits"))
        .def("apply_gate", &rocquantum::QuantumSimulator::apply_gate, py::arg("gate_name"), py::arg("targets"), py::arg("params"))
        .def("apply_matrix", [](rocquantum::QuantumSimulator &self, py::array_t<std::complex<double>> matrix, const std::vector<unsigned>& targets) {
            self.apply_matrix(numpy_to_std_vector(matrix), targets);
        }, py::arg("matrix"), py::arg("targets"))
        .def("get_statevector", [](rocquantum::QuantumSimulator &self) {
            std::vector<std::complex<double>> sv = self.get_statevector();
            // Create a new numpy array and copy the data to avoid returning a pointer to a temporary vector.
            py::array_t<std::complex<double>> result(sv.size());
            py::buffer_info buf = result.request();
            std::complex<double> *ptr = static_cast<std::complex<double> *>(buf.ptr);
            std::copy(sv.begin(), sv.end(), ptr);
            return result;
        })
        .def("measure", &rocquantum::QuantumSimulator::measure, py::arg("qubits"), py::arg("shots"))
        .def("reset", &rocquantum::QuantumSimulator::reset, "Resets the simulator to the |0...0> state.");
}