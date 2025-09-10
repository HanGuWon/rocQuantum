#include "rocquantum/GateFusion.h"
#include <map>
#include <iostream>

// For matrix multiplication on the host
#include <complex>

namespace rocquantum {

// Simple 2x2 complex matrix multiplication: C = A * B
void matmul_2x2(std::complex<double>& c00, std::complex<double>& c01, std::complex<double>& c10, std::complex<double>& c11,
                const std::complex<double>& a00, const std::complex<double>& a01, const std::complex<double>& a10, const std::complex<double>& a11,
                const std::complex<double>& b00, const std::complex<double>& b01, const std::complex<double>& b10, const std::complex<double>& b11) {
    c00 = a00 * b00 + a01 * b10;
    c01 = a00 * b01 + a01 * b11;
    c10 = a10 * b00 + a11 * b10;
    c11 = a10 * b01 + a11 * b11;
}

GateFusion::GateFusion(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits)
    : handle_(handle), d_state_(d_state), numQubits_(numQubits) {}

rocqStatus_t GateFusion::processQueue(const std::vector<GateOp>& queue) {
    // This is a very basic implementation. A real implementation would be a state machine.
    for (size_t i = 0; i < queue.size(); ) {
        const auto& op = queue[i];

        // Attempt to fuse a chunk of single-qubit gates
        if (op.controls.empty() && op.targets.size() == 1) {
            std::vector<GateOp> chunk;
            chunk.push_back(op);
            size_t j = i + 1;
            while (j < queue.size() && queue[j].controls.empty() && queue[j].targets.size() == 1) {
                chunk.push_back(queue[j]);
                j++;
            }
            fuseAndApplySingleQubitGates(chunk);
            i = j;
        } else { // Non-fusable gate, apply directly
            if (op.name == "CNOT") {
                rocsvApplyCNOT(handle_, d_state_, numQubits_, op.controls[0], op.targets[0]);
            }
            // Add other multi-qubit gates here...
            i++;
        }
    }
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t GateFusion::fuseAndApplySingleQubitGates(const std::vector<GateOp>& gate_chunk) {
    // Group gates by target qubit
    std::map<unsigned, std::vector<GateOp>> gates_by_qubit;
    for (const auto& op : gate_chunk) {
        gates_by_qubit[op.targets[0]].push_back(op);
    }

    for (auto const& [qubit, gates] : gates_by_qubit) {
        // Start with identity matrix
        std::complex<double> fused_matrix[4] = {{1,0}, {0,0}, {0,0}, {1,0}};

        for (const auto& op : gates) {
            std::complex<double> op_matrix[4];
            // Get matrix for the current gate
            if (op.name == "H") {
                double val = 1.0 / sqrt(2.0);
                op_matrix[0] = {val, 0}; op_matrix[1] = {val, 0};
                op_matrix[2] = {val, 0}; op_matrix[3] = {-val, 0};
            } else if (op.name == "X") {
                op_matrix[0] = {0,0}; op_matrix[1] = {1,0};
                op_matrix[2] = {1,0}; op_matrix[3] = {0,0};
            }
            // ... add all other single qubit gates (Z, S, T, RZ etc.)
            else {
                 // For simplicity, skip gates we don't have matrices for yet
                continue;
            }

            std::complex<double> temp_matrix[4];
            // Note: Gate order is important. New gate matrix multiplies from the left.
            matmul_2x2(temp_matrix[0], temp_matrix[1], temp_matrix[2], temp_matrix[3],
                       op_matrix[0], op_matrix[1], op_matrix[2], op_matrix[3],
                       fused_matrix[0], fused_matrix[1], fused_matrix[2], fused_matrix[3]);
            
            for(int i=0; i<4; ++i) fused_matrix[i] = temp_matrix[i];
        }

        // Apply the final fused matrix
        rocComplex* d_fused_matrix;
        hipMalloc(&d_fused_matrix, 4 * sizeof(rocComplex));
        
        // Convert std::complex<double> to rocComplex (float or double complex)
        rocComplex h_fused_matrix[4];
        for(int i=0; i<4; ++i) {
            h_fused_matrix[i] = make_hipFloatComplex(fused_matrix[i].real(), fused_matrix[i].imag());
        }

        hipMemcpy(d_fused_matrix, h_fused_matrix, 4 * sizeof(rocComplex), hipMemcpyHostToDevice);
        
        unsigned qubit_indices[] = {qubit};
        rocsvApplyMatrix(handle_, d_state_, numQubits_, qubit_indices, 1, d_fused_matrix, 2);

        hipFree(d_fused_matrix);
    }
    return ROCQ_STATUS_SUCCESS;
}

} // namespace rocquantum
