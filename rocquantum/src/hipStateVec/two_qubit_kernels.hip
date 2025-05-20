#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h" // For rocComplex

// Convention: Qubit 0 is LSB.
// For a two-qubit gate on target qubits q1 (higher index) and q0 (lower index):
// Amplitudes are grouped in quartets (amp00, amp01, amp10, amp11)
// amp00 corresponds to state |...q1=0...q0=0...⟩
// amp01 corresponds to state |...q1=0...q0=1...⟩
// amp10 corresponds to state |...q1=1...q0=0...⟩
// amp11 corresponds to state |...q1=1...q0=1...⟩
//
// Stride for q0 is k0 = 1ULL << q0_idx
// Stride for q1 is k1 = 1ULL << q1_idx
//
// Each thread will handle one quartet of amplitudes.

__device__ inline void apply_2q_matrix(rocComplex& amp00, rocComplex& amp01, rocComplex& amp10, rocComplex& amp11, const rocComplex* matrix) {
    // Matrix is 4x4, column-major.
    // out[i] = sum_j M[i][j] * in[j]
    
    rocComplex in[4] = {amp00, amp01, amp10, amp11};
    rocComplex out[4];

    for (int i = 0; i < 4; ++i) {
        out[i] = make_hipFloatComplex(0.0f, 0.0f); // Initialize to zero
        for (int j = 0; j < 4; ++j) {
            // M[i][j] is matrix[i + j*4] for column-major
            rocComplex M_ij = matrix[i + j * 4];
            rocComplex val = in[j];
            // out[i] += M_ij * val;
            out[i].x += M_ij.x * val.x - M_ij.y * val.y;
            out[i].y += M_ij.x * val.y + M_ij.y * val.x;
        }
    }
    amp00 = out[0];
    amp01 = out[1];
    amp10 = out[2];
    amp11 = out[3];
}

// Generic kernel for applying an arbitrary 2-qubit unitary matrix
__global__ void apply_two_qubit_generic_matrix_kernel(rocComplex* state,
                                                      unsigned numQubits,
                                                      unsigned qubit0_idx, // Lower index qubit
                                                      unsigned qubit1_idx, // Higher index qubit
                                                      const rocComplex* matrix_gpu) { // Matrix already on GPU
    size_t N = 1ULL << numQubits;
    size_t k0 = 1ULL << qubit0_idx;
    size_t k1 = 1ULL << qubit1_idx;

    // Each thread processes one quartet of amplitudes
    // Grid Dim should be N / 4
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Complex indexing to pick out the 4 amplitudes
    // Imagine the state vector divided into blocks of size 2^(max(q0,q1)+1)
    // Within these blocks, further subdivide.
    // Let q_min = min(qubit0_idx, qubit1_idx)
    // Let q_max = max(qubit0_idx, qubit1_idx) (this is qubit1_idx if sorted)
    //
    // The indices are:
    // idx00: base
    // idx01: base + k0
    // idx10: base + k1
    // idx11: base + k0 + k1

    size_t outer_stride = 1ULL << (qubit1_idx + 1); // Stride for the outer loops (related to q1)
    size_t inner_stride = 1ULL << (qubit0_idx + 1); // Stride for the inner loops (related to q0)
    
    // 'tid' identifies the quartet.
    // The bits *not* involved in q0 or q1 determine the 'block' this quartet belongs to.
    // Example: N=3 qubits (0,1,2). Gate on q0, q2. (q0_idx=0, q1_idx=2)
    // k0=1, k1=4.
    // Quartets:
    // (000,001,100,101) for q1=0 -> (idx0,idx1,idx4,idx5)
    // (010,011,110,111) for q1=1 -> (idx2,idx3,idx6,idx7)

    // Simplified calculation for base_idx00:
    // Treat tid as an index into a conceptual array of N/4 quartets.
    // We need to map tid to the actual idx00 in the full state vector.
    // Bits of tid correspond to the state of qubits *other* than qubit0_idx and qubit1_idx.
    
    size_t current_offset = 0;
    // size_t bit_val_q0 = 0; // will be 0 or k0. Not used in this simplified version.
    // size_t bit_val_q1 = 0; // will be 0 or k1. Not used in this simplified version.
    
    size_t effective_tid = tid;
    for (unsigned i = 0; i < numQubits; ++i) {
        if (i == qubit0_idx || i == qubit1_idx) continue;
        if ((effective_tid & 1) != 0) {
            current_offset += (1ULL << i);
        }
        effective_tid >>= 1;
    }

    size_t idx00 = current_offset;
    size_t idx01 = idx00 + k0;
    size_t idx10 = idx00 + k1;
    size_t idx11 = idx00 + k0 + k1;

    if (idx11 < N) { // Ensure all indices are within bounds
        rocComplex amp00 = state[idx00];
        rocComplex amp01 = state[idx01];
        rocComplex amp10 = state[idx10];
        rocComplex amp11 = state[idx11];

        apply_2q_matrix(amp00, amp01, amp10, amp11, matrix_gpu);

        state[idx00] = amp00;
        state[idx01] = amp01;
        state[idx10] = amp10;
        state[idx11] = amp11;
    }
}

// CNOT Gate: controlQubit, targetQubit
// Flips target if control is 1.
// Assumes controlQubit_idx and targetQubit_idx are distinct.
// Matrix for CNOT(control, target) where control is higher bit for matrix:
// q1=control, q0=target:  (Indices: 00, 01, 10, 11)
// No change if control is 0: amp00, amp01 unchanged
// Flip target if control is 1: amp10 swapped with amp11
// |1 0 0 0|
// |0 1 0 0|
// |0 0 0 1|
// |0 0 1 0|
__global__ void apply_CNOT_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned controlQubit_idx,
                                  unsigned targetQubit_idx) {
    size_t N = 1ULL << numQubits;
    size_t k_target = 1ULL << targetQubit_idx;
    size_t k_control = 1ULL << controlQubit_idx;

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles a pair to potentially swap

    // Indexing here is for the |..control=1..target=0..> and |..control=1..target=1..> states
    // The 'control' bit must be 1.
    // The 'target' bit can be 0 or 1.
    // All other bits define 'tid_other_bits'.
    // Number of "other bits" is numQubits - 2.
    // tid ranges from 0 to (1ULL << (numQubits - 2)) - 1.
    
    size_t current_offset = 0;
    size_t effective_tid = tid; // tid now represents the combination of "other" qubit states
    for (unsigned i = 0; i < numQubits; ++i) {
        if (i == controlQubit_idx || i == targetQubit_idx) continue;
        if ((effective_tid & 1) != 0) { // If the LSB of effective_tid is 1
            current_offset += (1ULL << i); // Add this qubit's stride to the offset
        }
        effective_tid >>= 1; // Move to the next bit of the "other" qubits
    }

    // Base index where control bit is 1, and target bit is 0
    size_t idx_c1_t0 = current_offset + k_control; // control bit is 1, target bit is 0
    size_t idx_c1_t1 = idx_c1_t0 + k_target;      // control bit is 1, target bit is 1

    if (idx_c1_t1 < N) {
        rocComplex temp = state[idx_c1_t0];
        state[idx_c1_t0] = state[idx_c1_t1];
        state[idx_c1_t1] = temp;
    }
}


// CZ Gate: controlQubit, targetQubit
// Flips phase of |11> state.
// Matrix: diag(1, 1, 1, -1) assuming q1, q0 order
__global__ void apply_CZ_kernel(rocComplex* state,
                                unsigned numQubits,
                                unsigned qubit0_idx, // Irrelevant which is control/target due to symmetry
                                unsigned qubit1_idx) {
    size_t N = 1ULL << numQubits;
    size_t k0 = 1ULL << qubit0_idx;
    size_t k1 = 1ULL << qubit1_idx;

    // tid ranges from 0 to (1ULL << (numQubits - 2)) - 1.
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 

    size_t current_offset = 0;
    size_t effective_tid = tid;
    for (unsigned i = 0; i < numQubits; ++i) {
        if (i == qubit0_idx || i == qubit1_idx) continue;
        if ((effective_tid & 1) != 0) {
            current_offset += (1ULL << i);
        }
        effective_tid >>= 1;
    }
    
    // Index for the |..1..1..> state component for this group
    // This is where both qubit0_idx and qubit1_idx are 1, plus the offset from other qubits.
    size_t idx11 = current_offset + k0 + k1;

    if (idx11 < N) {
        state[idx11].x = -state[idx11].x;
        state[idx11].y = -state[idx11].y;
    }
}


// SWAP Gate: qubit0_idx, qubit1_idx
// Swaps states |01> and |10>
// Matrix: (q1, q0 ordering)
// |1 0 0 0|
// |0 0 1 0|  swaps amp01 with amp10
// |0 1 0 0|
// |0 0 0 1|
__global__ void apply_SWAP_kernel(rocComplex* state,
                                  unsigned numQubits,
                                  unsigned qubit0_idx, // lower index qubit
                                  unsigned qubit1_idx) { // higher index qubit
    size_t N = 1ULL << numQubits;
    size_t k0 = 1ULL << qubit0_idx; 
    size_t k1 = 1ULL << qubit1_idx; 

    // tid ranges from 0 to (1ULL << (numQubits - 2)) - 1.
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 

    size_t current_offset = 0;
    size_t effective_tid = tid;

    for (unsigned i = 0; i < numQubits; ++i) {
        if (i == qubit0_idx || i == qubit1_idx) continue;
        if ((effective_tid & 1) != 0) {
            current_offset += (1ULL << i);
        }
        effective_tid >>= 1;
    }

    // Index for |...q1=0...q0=1...> (other bits are 'current_offset')
    size_t idx_q1_0_q0_1 = current_offset + k0;
    // Index for |...q1=1...q0=0...> (other bits are 'current_offset')
    size_t idx_q1_1_q0_0 = current_offset + k1;

    if (idx_q1_0_q0_1 < N && idx_q1_1_q0_0 < N) { // Ensure both are valid
        rocComplex temp = state[idx_q1_0_q0_1];
        state[idx_q1_0_q0_1] = state[idx_q1_1_q0_0];
        state[idx_q1_1_q0_0] = temp;
    }
}
