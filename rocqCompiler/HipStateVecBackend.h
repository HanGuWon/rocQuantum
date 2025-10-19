#ifndef HIP_STATE_VEC_BACKEND_H
#define HIP_STATE_VEC_BACKEND_H

#include "QuantumBackend.h"
#include "rocquantum/hipStateVec.h" // The concrete simulator API

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
};

} // namespace rocq

#endif // HIP_STATE_VEC_BACKEND_H
