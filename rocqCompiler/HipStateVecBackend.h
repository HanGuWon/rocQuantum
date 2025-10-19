#ifndef HIP_STATE_VEC_BACKEND_H
#define HIP_STATE_VEC_BACKEND_H

#include "QuantumBackend.h"
#include "rocquantum/hipStateVec.h" // The concrete simulator API

#include <vector>
#include <string>

namespace rocq {

class HipStateVecBackend : public QuantumBackend {
public:
    HipStateVecBackend();
    virtual ~HipStateVecBackend() override;

    void initialize(unsigned num_qubits) override;
    void apply_gate(const std::string& gate_name, const std::vector<unsigned>& targets) override;
    void apply_parametrized_gate(const std::string& gate_name, double parameter, const std::vector<unsigned>& targets) override;
    std::vector<std::complex<double>> get_state_vector() override;
    void destroy() override;

private:
    rocsvHandle_t sim_handle;
    unsigned num_qubits;
    rocComplex* device_state;
    size_t batch_size;
    bool is_initialized;

    using GateHandler = void (HipStateVecBackend::*)(const std::vector<unsigned>&);
    using ParamGateHandler = void (HipStateVecBackend::*)(double, const std::vector<unsigned>&);

    void ensure_initialized() const;
    void apply_h_gate(const std::vector<unsigned>& targets);
    void apply_x_gate(const std::vector<unsigned>& targets);
    void apply_y_gate(const std::vector<unsigned>& targets);
    void apply_z_gate(const std::vector<unsigned>& targets);
    void apply_s_gate(const std::vector<unsigned>& targets);
    void apply_sdg_gate(const std::vector<unsigned>& targets);
    void apply_t_gate(const std::vector<unsigned>& targets);
    void apply_cnot_gate(const std::vector<unsigned>& targets);
    void apply_cz_gate(const std::vector<unsigned>& targets);
    void apply_swap_gate(const std::vector<unsigned>& targets);
    void apply_multi_controlled_x_gate(const std::vector<unsigned>& targets);
    void apply_cswap_gate(const std::vector<unsigned>& targets);

    void apply_rx_gate(double parameter, const std::vector<unsigned>& targets);
    void apply_ry_gate(double parameter, const std::vector<unsigned>& targets);
    void apply_rz_gate(double parameter, const std::vector<unsigned>& targets);
    void apply_crx_gate(double parameter, const std::vector<unsigned>& targets);
    void apply_cry_gate(double parameter, const std::vector<unsigned>& targets);
    void apply_crz_gate(double parameter, const std::vector<unsigned>& targets);
};

} // namespace rocq

#endif // HIP_STATE_VEC_BACKEND_H
