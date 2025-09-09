#include "rocquantum/include/rocquantum.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

namespace rocquantum {

QuantumSimulator::QuantumSimulator(unsigned num_qubits) : num_qubits_(num_qubits) {
    std::cout << "[C++] QuantumSimulator created with " << num_qubits_ << " qubits." << std::endl;
    this->reset();
}

void QuantumSimulator::reset() {
    std::cout << "[C++] Simulator reset to |0...0> state." << std::endl;
    // Implementation for state reset would go here.
}

void QuantumSimulator::apply_gate(const std::string& gate_name, const std::vector<unsigned>& targets, const std::vector<double>& params) {
    std::cout << "[C++] Applying gate '" << gate_name << "' to qubit(s) ";
    for(const auto& t : targets) std::cout << t << " ";
    std::cout << std::endl;
}

void QuantumSimulator::apply_matrix(const std::vector<std::complex<double>>& matrix, const std::vector<unsigned>& targets) {
    std::cout << "[C++] Applying custom matrix to qubit(s) ";
    for(const auto& t : targets) std::cout << t << " ";
    std::cout << std::endl;
}

std::vector<std::complex<double>> QuantumSimulator::get_statevector() {
    std::cout << "[C++] Getting statevector." << std::endl;
    size_t state_vec_size = 1 << num_qubits_;
    std::vector<std::complex<double>> state_vector(state_vec_size, {0.0, 0.0});
    state_vector[0] = {1.0, 0.0};
    return state_vector;
}

std::vector<long long> QuantumSimulator::measure(const std::vector<unsigned>& qubits, int shots) {
    std::cout << "[C++] Measuring " << shots << " shots." << std::endl;
    return std::vector<long long>(shots, 0);
}

} // namespace rocquantum
