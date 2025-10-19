#pragma once

#include <complex>
#include <string>
#include <vector>

struct hipDoubleComplex;

namespace rocquantum {

class QuantumSimulator {
public:
    explicit QuantumSimulator(unsigned num_qubits);
    ~QuantumSimulator();

    void reset();
    void apply_gate(const std::string& gate_name,
                    const std::vector<unsigned>& targets,
                    const std::vector<double>& params = {});
    void apply_matrix(const std::vector<std::complex<double>>& matrix,
                      const std::vector<unsigned>& targets);
    std::vector<std::complex<double>> get_statevector() const;
    std::vector<long long> measure(const std::vector<unsigned>& qubits, int shots);
    unsigned num_qubits() const noexcept;

    // Legacy API retained for compatibility with previous QSim bindings.
    void ApplyGate(const std::string& gate_name, int target_qubit);
    void ApplyGate(const std::string& gate_name, int control_qubit, int target_qubit);
    void ApplyGate(const std::vector<std::complex<double>>& gate_matrix, int target_qubit);
    void Execute();
    std::vector<std::complex<double>> GetStateVector() const;

private:
    void ensure_valid_qubit(unsigned qubit) const;
    void synchronize() const;

    unsigned num_qubits_;
    size_t state_vec_size_;
    hipDoubleComplex* device_state_vector_;
};

using QSim = QuantumSimulator;

} // namespace rocquantum
