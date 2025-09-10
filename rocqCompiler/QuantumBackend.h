#ifndef QUANTUM_BACKEND_H
#define QUANTUM_BACKEND_H

#include <string>
#include <vector>
#include <complex>
#include <stdexcept>

namespace rocq {

// Abstract base class for all quantum execution backends.
class QuantumBackend {
public:
    virtual ~QuantumBackend() = default;

    // Initializes the backend's state for a given number of qubits.
    virtual void initialize(unsigned num_qubits) = 0;

    // Applies a simple, non-parameterized gate.
    virtual void apply_gate(const std::string& gate_name, const std::vector<unsigned>& targets) = 0;

    // Applies a gate with a single double parameter (e.g., rotations).
    virtual void apply_parametrized_gate(const std::string& gate_name, double parameter, const std::vector<unsigned>& targets) = 0;

    // Retrieves the final state vector from the backend.
    virtual std::vector<std::complex<double>> get_state_vector() = 0;

    // Releases any resources held by the backend.
    virtual void destroy() = 0;
};

// A simple factory to create backend instances from a string identifier.
std::unique_ptr<QuantumBackend> create_backend(const std::string& backend_name);

} // namespace rocq

#endif // QUANTUM_BACKEND_H
