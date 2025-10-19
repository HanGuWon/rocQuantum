#include "HipStateVecBackend.h"

#include <algorithm>
#include <complex>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

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

inline void validate_target_arity(const std::vector<unsigned>& targets,
                                  std::size_t min_targets,
                                  std::size_t max_targets,
                                  const std::string& gate_name) {
    if (targets.size() < min_targets) {
        throw std::invalid_argument("Gate '" + gate_name + "' expected at least " +
                                    std::to_string(min_targets) + " target qubits but received " +
                                    std::to_string(targets.size()) + ".");
    }
    if (max_targets != 0 && targets.size() > max_targets) {
        throw std::invalid_argument("Gate '" + gate_name + "' expected at most " +
                                    std::to_string(max_targets) + " target qubits but received " +
                                    std::to_string(targets.size()) + ".");
    }
}

} // namespace

namespace rocq {

namespace {

struct GateSpec {
    std::size_t min_targets;
    std::size_t max_targets; // 0 denotes unbounded.
    HipStateVecBackend::GateHandler handler;
};

struct ParamGateSpec {
    std::size_t min_targets;
    std::size_t max_targets;
    HipStateVecBackend::ParamGateHandler handler;
};

const std::unordered_map<std::string, GateSpec>& non_param_gate_table() {
    static const auto table = []() {
        std::unordered_map<std::string, GateSpec> map;
        auto emplace_alias = [&map](const std::string& alias, GateSpec spec) {
            map.emplace(alias, spec);
        };

        const GateSpec h_spec{1, 1, &HipStateVecBackend::apply_h_gate};
        emplace_alias("h", h_spec);

        const GateSpec x_spec{1, 1, &HipStateVecBackend::apply_x_gate};
        emplace_alias("x", x_spec);
        emplace_alias("paulix", x_spec);

        const GateSpec y_spec{1, 1, &HipStateVecBackend::apply_y_gate};
        emplace_alias("y", y_spec);
        emplace_alias("pauliy", y_spec);

        const GateSpec z_spec{1, 1, &HipStateVecBackend::apply_z_gate};
        emplace_alias("z", z_spec);
        emplace_alias("pauliz", z_spec);

        const GateSpec s_spec{1, 1, &HipStateVecBackend::apply_s_gate};
        emplace_alias("s", s_spec);

        const GateSpec sdg_spec{1, 1, &HipStateVecBackend::apply_sdg_gate};
        emplace_alias("sdg", sdg_spec);
        emplace_alias("sdag", sdg_spec);

        const GateSpec t_spec{1, 1, &HipStateVecBackend::apply_t_gate};
        emplace_alias("t", t_spec);

        const GateSpec cx_spec{2, 2, &HipStateVecBackend::apply_cnot_gate};
        emplace_alias("cx", cx_spec);
        emplace_alias("cnot", cx_spec);

        const GateSpec cz_spec{2, 2, &HipStateVecBackend::apply_cz_gate};
        emplace_alias("cz", cz_spec);

        const GateSpec swap_spec{2, 2, &HipStateVecBackend::apply_swap_gate};
        emplace_alias("swap", swap_spec);

        const GateSpec mcx_spec{2, 0, &HipStateVecBackend::apply_multi_controlled_x_gate};
        emplace_alias("mcx", mcx_spec);
        emplace_alias("ccx", mcx_spec);
        emplace_alias("toffoli", mcx_spec);

        const GateSpec cswap_spec{3, 3, &HipStateVecBackend::apply_cswap_gate};
        emplace_alias("cswap", cswap_spec);
        emplace_alias("fredkin", cswap_spec);

        return map;
    }();
    return table;
}

const std::unordered_map<std::string, ParamGateSpec>& param_gate_table() {
    static const auto table = []() {
        std::unordered_map<std::string, ParamGateSpec> map;
        auto emplace_alias = [&map](const std::string& alias, ParamGateSpec spec) {
            map.emplace(alias, spec);
        };

        const ParamGateSpec rx_spec{1, 1, &HipStateVecBackend::apply_rx_gate};
        emplace_alias("rx", rx_spec);

        const ParamGateSpec ry_spec{1, 1, &HipStateVecBackend::apply_ry_gate};
        emplace_alias("ry", ry_spec);

        const ParamGateSpec rz_spec{1, 1, &HipStateVecBackend::apply_rz_gate};
        emplace_alias("rz", rz_spec);

        const ParamGateSpec crx_spec{2, 2, &HipStateVecBackend::apply_crx_gate};
        emplace_alias("crx", crx_spec);

        const ParamGateSpec cry_spec{2, 2, &HipStateVecBackend::apply_cry_gate};
        emplace_alias("cry", cry_spec);

        const ParamGateSpec crz_spec{2, 2, &HipStateVecBackend::apply_crz_gate};
        emplace_alias("crz", crz_spec);

        return map;
    }();
    return table;
}

} // namespace

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

void HipStateVecBackend::ensure_initialized() const {
    if (!is_initialized) {
        throw std::runtime_error("Backend not initialized.");
    }
}

void HipStateVecBackend::apply_gate(const std::string& gate_name, const std::vector<unsigned>& targets) {
    ensure_initialized();

    const std::string lowered = to_lower(gate_name);
    const auto& table = non_param_gate_table();
    const auto it = table.find(lowered);
    if (it == table.end()) {
        throw std::runtime_error("Unknown gate: " + gate_name);
    }

    const auto& spec = it->second;
    validate_target_arity(targets, spec.min_targets, spec.max_targets, gate_name);

    (this->*spec.handler)(targets);
}

void HipStateVecBackend::apply_parametrized_gate(const std::string& gate_name,
                                                 double parameter,
                                                 const std::vector<unsigned>& targets) {
    ensure_initialized();

    const std::string lowered = to_lower(gate_name);
    const auto& table = param_gate_table();
    const auto it = table.find(lowered);
    if (it == table.end()) {
        throw std::runtime_error("Unknown parametrised gate: " + gate_name);
    }

    const auto& spec = it->second;
    validate_target_arity(targets, spec.min_targets, spec.max_targets, gate_name);

    (this->*spec.handler)(parameter, targets);
}

std::vector<std::complex<double>> HipStateVecBackend::get_state_vector() {
    ensure_initialized();

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

namespace rocq {

void HipStateVecBackend::apply_h_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyH(sim_handle, device_state, num_qubits, targets[0]), "apply H");
}

void HipStateVecBackend::apply_x_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyX(sim_handle, device_state, num_qubits, targets[0]), "apply X");
}

void HipStateVecBackend::apply_y_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyY(sim_handle, device_state, num_qubits, targets[0]), "apply Y");
}

void HipStateVecBackend::apply_z_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyZ(sim_handle, device_state, num_qubits, targets[0]), "apply Z");
}

void HipStateVecBackend::apply_s_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyS(sim_handle, device_state, num_qubits, targets[0]), "apply S");
}

void HipStateVecBackend::apply_sdg_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplySdg(sim_handle, device_state, num_qubits, targets[0]), "apply Sdg");
}

void HipStateVecBackend::apply_t_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyT(sim_handle, device_state, num_qubits, targets[0]), "apply T");
}

void HipStateVecBackend::apply_cnot_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyCNOT(sim_handle, device_state, num_qubits, targets[0], targets[1]), "apply CNOT");
}

void HipStateVecBackend::apply_cz_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyCZ(sim_handle, device_state, num_qubits, targets[0], targets[1]), "apply CZ");
}

void HipStateVecBackend::apply_swap_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplySWAP(sim_handle, device_state, num_qubits, targets[0], targets[1]), "apply SWAP");
}

void HipStateVecBackend::apply_multi_controlled_x_gate(const std::vector<unsigned>& targets) {
    if (targets.size() < 2) {
        throw std::invalid_argument("Multi-controlled X requires at least one control qubit and one target qubit.");
    }
    const unsigned target = targets.back();
    std::vector<unsigned> controls(targets.begin(), targets.end() - 1);
    check_status(rocsvApplyMultiControlledX(sim_handle,
                                            device_state,
                                            num_qubits,
                                            controls.data(),
                                            static_cast<unsigned>(controls.size()),
                                            target),
                 "apply multi-controlled X");
}

void HipStateVecBackend::apply_cswap_gate(const std::vector<unsigned>& targets) {
    check_status(rocsvApplyCSWAP(sim_handle,
                                 device_state,
                                 num_qubits,
                                 targets[0],
                                 targets[1],
                                 targets[2]),
                 "apply CSWAP");
}

void HipStateVecBackend::apply_rx_gate(double parameter, const std::vector<unsigned>& targets) {
    check_status(rocsvApplyRx(sim_handle, device_state, num_qubits, targets[0], parameter), "apply RX");
}

void HipStateVecBackend::apply_ry_gate(double parameter, const std::vector<unsigned>& targets) {
    check_status(rocsvApplyRy(sim_handle, device_state, num_qubits, targets[0], parameter), "apply RY");
}

void HipStateVecBackend::apply_rz_gate(double parameter, const std::vector<unsigned>& targets) {
    check_status(rocsvApplyRz(sim_handle, device_state, num_qubits, targets[0], parameter), "apply RZ");
}

void HipStateVecBackend::apply_crx_gate(double parameter, const std::vector<unsigned>& targets) {
    check_status(rocsvApplyCRX(sim_handle, device_state, num_qubits, targets[0], targets[1], parameter),
                 "apply CRX");
}

void HipStateVecBackend::apply_cry_gate(double parameter, const std::vector<unsigned>& targets) {
    check_status(rocsvApplyCRY(sim_handle, device_state, num_qubits, targets[0], targets[1], parameter),
                 "apply CRY");
}

void HipStateVecBackend::apply_crz_gate(double parameter, const std::vector<unsigned>& targets) {
    check_status(rocsvApplyCRZ(sim_handle, device_state, num_qubits, targets[0], targets[1], parameter),
                 "apply CRZ");
}

} // namespace rocq
