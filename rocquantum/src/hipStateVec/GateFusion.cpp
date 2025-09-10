#include "rocquantum/GateFusion.h"
#include <map>
#include <iostream>
#include <complex>
#include <numeric>
#include <algorithm>

namespace rocquantum {

using c128 = std::complex<double>;

// --- Matrix Math Helpers ---

// Helper for 4x4 matrix multiplication: C = A * B (row-major)
void matmul_4x4(c128* C, const c128* A, const c128* B) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i * 4 + j] = {0.0, 0.0};
            for (int k = 0; k < 4; ++k) {
                C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
            }
        }
    }
}

// Helper to create a 4x4 tensor product matrix: C = A ⊗ B
void tensor_product_2x2(c128* C, const c128* A, const c128* B) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) {
                    C[(i * 2 + k) * 4 + (j * 2 + l)] = A[i * 2 + j] * B[k * 2 + l];
                }
            }
        }
    }
}

// Helper to get the 2x2 matrix for a single-qubit gate
bool get_gate_matrix_2x2(const GateOp& op, c128* matrix) {
    if (op.name == "X") {
        matrix[0] = {0,0}; matrix[1] = {1,0};
        matrix[2] = {1,0}; matrix[3] = {0,0};
    } else if (op.name == "Y") {
        matrix[0] = {0,0}; matrix[1] = {0,-1};
        matrix[2] = {0,1}; matrix[3] = {0,0};
    } else if (op.name == "Z") {
        matrix[0] = {1,0}; matrix[1] = {0,0};
        matrix[2] = {0,0}; matrix[3] = {-1,0};
    } else if (op.name == "H") {
        double val = 1.0 / sqrt(2.0);
        matrix[0] = {val, 0}; matrix[1] = {val, 0};
        matrix[2] = {val, 0}; matrix[3] = {-val, 0};
    } else if (op.name == "S") {
        matrix[0] = {1,0}; matrix[1] = {0,0};
        matrix[2] = {0,0}; matrix[3] = {0,1};
    } else if (op.name == "T") {
        double val = 1.0 / sqrt(2.0);
        matrix[0] = {1,0}; matrix[1] = {0,0};
        matrix[2] = {0,0}; matrix[3] = {val,val};
    } else if (op.name == "RX") {
        double angle = op.params[0];
        double cos_t = cos(angle / 2.0);
        double sin_t = sin(angle / 2.0);
        matrix[0] = {cos_t, 0}; matrix[1] = {0, -sin_t};
        matrix[2] = {0, -sin_t}; matrix[3] = {cos_t, 0};
    } else if (op.name == "RY") {
        double angle = op.params[0];
        double cos_t = cos(angle / 2.0);
        double sin_t = sin(angle / 2.0);
        matrix[0] = {cos_t, 0}; matrix[1] = {-sin_t, 0};
        matrix[2] = {sin_t, 0}; matrix[3] = {cos_t, 0};
    } else if (op.name == "RZ") {
        double angle = op.params[0];
        double cos_t = cos(angle / 2.0);
        double sin_t = sin(angle / 2.0);
        matrix[0] = {cos_t, -sin_t}; matrix[1] = {0, 0};
        matrix[2] = {0, 0}; matrix[3] = {cos_t, sin_t};
    } else {
        return false; // Not a fusable single-qubit gate
    }
    return true;
}


GateFusion::GateFusion(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits)
    : handle_(handle), d_state_(d_state), numQubits_(numQubits) {}

rocqStatus_t GateFusion::processQueue(const std::vector<GateOp>& queue) {
    std::vector<bool> processed(queue.size(), false);

    for (size_t i = 0; i < queue.size(); ++i) {
        if (processed[i]) continue;

        // --- CNOT Fusion Pattern ---
        if (queue[i].name == "CNOT") {
            const auto& cnot_op = queue[i];
            unsigned ctrl = cnot_op.controls[0];
            unsigned targ = cnot_op.targets[0];

            c128 pre_matrix[16] = { {1,0},{0,0},{0,0},{0,0}, {0,0},{1,0},{0,0},{0,0}, {0,0},{0,0},{1,0},{0,0}, {0,0},{0,0},{0,0},{1,0} };
            c128 post_matrix[16] = { {1,0},{0,0},{0,0},{0,0}, {0,0},{1,0},{0,0},{0,0}, {0,0},{0,0},{1,0},{0,0}, {0,0},{0,0},{0,0},{1,0} };

            // Greedily fuse one gate before
            if (i > 0 && !processed[i-1] && queue[i-1].targets.size() == 1) {
                const auto& pre_op = queue[i-1];
                c128 pre_op_mat[4];
                if (get_gate_matrix_2x2(pre_op, pre_op_mat)) {
                    c128 id_mat[4] = {{1,0},{0,0},{0,0},{1,0}};
                    if (pre_op.targets[0] == ctrl) {
                        tensor_product_2x2(pre_matrix, pre_op_mat, id_mat);
                    } else if (pre_op.targets[0] == targ) {
                        tensor_product_2x2(pre_matrix, id_mat, pre_op_mat);
                    }
                    processed[i-1] = true;
                }
            }

            // Greedily fuse one gate after
            if (i < queue.size() - 1 && !processed[i+1] && queue[i+1].targets.size() == 1) {
                const auto& post_op = queue[i+1];
                c128 post_op_mat[4];
                if (get_gate_matrix_2x2(post_op, post_op_mat)) {
                    c128 id_mat[4] = {{1,0},{0,0},{0,0},{1,0}};
                    if (post_op.targets[0] == ctrl) {
                        tensor_product_2x2(post_matrix, post_op_mat, id_mat);
                    } else if (post_op.targets[0] == targ) {
                        tensor_product_2x2(post_matrix, id_mat, post_op_mat);
                    }
                    processed[i+1] = true;
                }
            }

            c128 cnot_matrix[16] = { {1,0},{0,0},{0,0},{0,0}, {0,0},{1,0},{0,0},{0,0}, {0,0},{0,0},{0,0},{1,0}, {0,0},{0,0},{1,0},{0,0} };
            
            c128 temp_matrix[16], final_matrix[16];
            matmul_4x4(temp_matrix, cnot_matrix, pre_matrix);
            matmul_4x4(final_matrix, post_matrix, temp_matrix);

            rocComplex* d_fused_matrix;
            hipMalloc(&d_fused_matrix, 16 * sizeof(rocComplex));
            hipMemcpy(d_fused_matrix, final_matrix, 16 * sizeof(rocComplex), hipMemcpyHostToDevice);
            
            unsigned qubit_indices[] = {std::min(ctrl, targ), std::max(ctrl, targ)};
            rocsvApplyMatrix(handle_, d_state_, numQubits_, qubit_indices, 2, d_fused_matrix, 4);

            hipFree(d_fused_matrix);
            processed[i] = true;
            continue;
        }
        
        // Fallback for non-fused gates
        // ...
    }
    return ROCQ_STATUS_SUCCESS;
}

} // namespace rocquantum
