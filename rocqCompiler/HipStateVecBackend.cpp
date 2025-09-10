#include "HipStateVecBackend.h"
#include <stdexcept>

namespace rocq {

HipStateVecBackend::HipStateVecBackend() : sim_handle(nullptr), num_qubits(0), is_initialized(false) {
    if (rocsvCreate(&sim_handle) != ROCQ_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create rocStateVec handle.");
    }
}

HipStateVecBackend::~HipStateVecBackend() {
    if (sim_handle) {
        destroy();
        rocsvDestroy(sim_handle);
    }
}

void HipStateVecBackend::initialize(unsigned n_qubits) {
    this->num_qubits = n_qubits;
    rocsvAllocateState(sim_handle, num_qubits, nullptr, 1);
    rocsvInitializeState(sim_handle, nullptr, num_qubits);
    is_initialized = true;
}

void HipStateVecBackend::apply_gate(const std::string& gate_name, const std::vector<unsigned>& targets) {
    if (!is_initialized) throw std::runtime_error("Backend not initialized.");
    if (targets.empty()) throw std::invalid_argument("Gate must have targets.");

    if (gate_name == "h") rocsvApplyH(sim_handle, nullptr, num_qubits, targets[0]);
    else if (gate_name == "x") rocsvApplyX(sim_handle, nullptr, num_qubits, targets[0]);
    else if (gate_name == "y") rocsvApplyY(sim_handle, nullptr, num_qubits, targets[0]);
    else if (gate_name == "cnot") {
        if (targets.size() < 2) throw std::invalid_argument("CNOT requires 2 targets.");
        rocsvApplyCNOT(sim_handle, nullptr, num_qubits, targets[0], targets[1]);
    }
    // ... other gates
    else throw std::runtime_error("Unknown gate: " + gate_name);
}

void HipStateVecBackend::apply_parametrized_gate(const std::string& gate_name, double parameter, const std::vector<unsigned>& targets) {
    // This would be implemented for RX, RY, RZ etc.
    throw std::runtime_error("Parametrized gates not implemented in this prototype.");
}

std::vector<std::complex<double>> HipStateVecBackend::get_state_vector() {
    if (!is_initialized) throw std::runtime_error("Backend not initialized.");
    
    size_t state_vec_size = 1ULL << num_qubits;
    std::vector<std::complex<double>> final_state(state_vec_size);
    
    rocsvGetStateVectorFull(sim_handle, nullptr, reinterpret_cast<rocComplex*>(final_state.data()));
    return final_state;
}

void HipStateVecBackend::destroy() {
    // In a real implementation, this would free the device state memory.
    // rocsvFreeState(sim_handle);
    is_initialized = false;
}

// --- Backend Factory Implementation ---
std::unique_ptr<QuantumBackend> create_backend(const std::string& backend_name) {
    if (backend_name == "hip_statevec") {
        return std::make_unique<HipStateVecBackend>();
    }
    throw std::invalid_argument("Unknown backend: " + backend_name);
}

} // namespace rocq
