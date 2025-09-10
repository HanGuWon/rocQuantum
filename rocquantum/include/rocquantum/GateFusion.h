#ifndef GATEFUSION_H
#define GATEFUSION_H

#include "hipStateVec.h"
#include <vector>
#include <string>
#include <variant>

namespace rocquantum {

// Represents a single gate operation for the fusion queue
struct GateOp {
    std::string name;
    std::vector<unsigned> targets;
    std::vector<unsigned> controls;
    std::vector<double> params;
};

class GateFusion {
public:
    GateFusion(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits);

    // Processes a queue of gates, applying fusion where possible
    rocqStatus_t processQueue(const std::vector<GateOp>& queue);

private:
    rocsvHandle_t handle_;
    rocComplex* d_state_;
    unsigned numQubits_;

    // A simple fusion strategy for single-qubit gates
    rocqStatus_t fuseAndApplySingleQubitGates(const std::vector<GateOp>& gate_chunk);
};

} // namespace rocquantum

#endif // GATEFUSION_H
