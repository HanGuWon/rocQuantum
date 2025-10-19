#include "HipStateVecBackend.h"

#include <algorithm>
#include <complex>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline std::string to_lower(const std::string& name) {
    std::string out(name.size(), '\0');
    std::transform(name.begin(), name.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

inline std::complex<double> to_std_complex(const rocComplex& value) {
    return {static_cast<double>(value.x), static_cast<double>(value.y)};
}

inline void check_status(rocqStatus_t status, const char* context) {
    if (status != ROCQ_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("hipStateVec error during ") + context +
                                 " (status " + std::to_string(status) + ")");
    }
}

} // namespace

namespace rocq {

HipStateVecBackend::HipStateVecBackend()
    : sim_handle(nullptr),
      num_qubits(0),
      device_state(nullptr),
      batch_size(1),
      is_initialized(false) {
    if (rocsvCreate(&sim_handle) != ROCQ_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create hipStateVec handle.");
    }
}

HipStateVecBackend::~HipStateVecBackend() {
    if (sim_handle) {
        destroy();
        rocsvDestroy(sim_handle);
    }
}

void HipStateVecBackend::initialize(unsigned n_qubits) {
    if (n_qubits == 0) {
        throw std::invalid_argument("hipStateVec backend requires at least one qubit.");
    }

    num_qubits = n_qubits;
    batch_size = 1;

    rocComplex* buffer = nullptr;
    check_status(rocsvAllocateState(sim_handle, num_qubits, &buffer, batch_size), "state allocation");
    device_state = buffer;
    check_status(rocsvInitializeState(sim_handle, device_state, num_qubits), "state initialisation");
    is_initialized = true;
}

void HipStateVecBackend::apply_gate(const std::string& gate_name, const std::vector<unsigned>& targets) {
    if (!is_initialized) throw std::runtime_error("Backend not initialized.");
    if (targets.empty()) throw std::invalid_argument("Gate requires at least one target qubit.");

    const std::string lowered = to_lower(gate_name);

    if (lowered == "h") {
        check_status(rocsvApplyH(sim_handle, device_state, num_qubits, targets[0]), "apply H");
    } else if (lowered == "x") {
        check_status(rocsvApplyX(sim_handle, device_state, num_qubits, targets[0]), "apply X");
    } else if (lowered == "y") {
        check_status(rocsvApplyY(sim_handle, device_state, num_qubits, targets[0]), "apply Y");
    } else if (lowered == "z") {
        check_status(rocsvApplyZ(sim_handle, device_state, num_qubits, targets[0]), "apply Z");
    } else if (lowered == "s") {
        check_status(rocsvApplyS(sim_handle, device_state, num_qubits, targets[0]), "apply S");
    } else if (lowered == "sdg" || lowered == "sdag") {
        check_status(rocsvApplySdg(sim_handle, device_state, num_qubits, targets[0]), "apply Sdg");
    } else if (lowered == "t") {
        check_status(rocsvApplyT(sim_handle, device_state, num_qubits, targets[0]), "apply T");
    } else if (lowered == "cx" || lowered == "cnot") {
        if (targets.size() < 2) throw std::invalid_argument("CNOT requires two qubit indices.");
        check_status(rocsvApplyCNOT(sim_handle, device_state, num_qubits, targets[0], targets[1]), "apply CNOT");
    } else if (lowered == "cz") {
        if (targets.size() < 2) throw std::invalid_argument("CZ requires two qubit indices.");
        check_status(rocsvApplyCZ(sim_handle, device_state, num_qubits, targets[0], targets[1]), "apply CZ");
    } else if (lowered == "swap") {
        if (targets.size() < 2) throw std::invalid_argument("SWAP requires two qubit indices.");
        check_status(rocsvApplySWAP(sim_handle, device_state, num_qubits, targets[0], targets[1]), "apply SWAP");
    } else {
        throw std::runtime_error("Unknown gate: " + gate_name);
    }
}

void HipStateVecBackend::apply_parametrized_gate(const std::string& gate_name,
                                                 double parameter,
                                                 const std::vector<unsigned>& targets) {
    throw std::runtime_error("Parametrized gates are not yet implemented for hipStateVec.");
}

std::vector<std::complex<double>> HipStateVecBackend::get_state_vector() {
    if (!is_initialized) throw std::runtime_error("Backend not initialized.");

    const size_t state_vec_size = 1ULL << num_qubits;
    std::vector<rocComplex> raw(state_vec_size);
    check_status(rocsvGetStateVectorFull(sim_handle, device_state, raw.data()), "fetch state vector");

    std::vector<std::complex<double>> result(state_vec_size);
    std::transform(raw.begin(), raw.end(), result.begin(), to_std_complex);
    return result;
}

void HipStateVecBackend::destroy() {
    if (sim_handle && device_state) {
        rocsvFreeState(sim_handle);
        device_state = nullptr;
    }
    is_initialized = false;
    num_qubits = 0;
}

// --- Backend Factory Implementation ---
std::unique_ptr<QuantumBackend> create_backend(const std::string& backend_name) {
    if (backend_name == "hip_statevec") {
        return std::make_unique<HipStateVecBackend>();
    }
    throw std::invalid_argument("Unknown backend: " + backend_name);
}

} // namespace rocq
