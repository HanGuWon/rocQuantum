#include <hip/hip_runtime.h>
#include "rocquantum/hipStateVec.h" // For rocComplex, real_t definition

// Constant memory for the 2x2 single-qubit matrix used by apply_single_qubit_generic_matrix_kernel
// This matrix is copied here by the host before launching the kernel.
// Only one such matrix can be "active" in constant memory at a time for this specific kernel.
__constant__ rocComplex const_single_q_matrix[4];

// Convention: Qubit 0 is the LSB.
// For a single qubit gate on qubit 'targetQubit':
// The state vector is |..., q_targetQubit, ...⟩
// Amplitudes are grouped in pairs (alpha, beta) where alpha corresponds to |...0...⟩
// and beta corresponds to |...1...⟩ at the targetQubit position.
// The stride between alpha and beta is 2^targetQubit.
// The size of the block of pairs being processed is 2^(targetQubit+1).
// Each thread will handle one pair of amplitudes (alpha, beta).

__device__ inline void apply_1q_matrix(rocComplex& amp0, rocComplex& amp1, const rocComplex* matrix) {
    rocComplex M00 = matrix[0]; // matrix is column-major: M00, M10, M01, M11
    rocComplex M10 = matrix[1];
    rocComplex M01 = matrix[2];
    rocComplex M11 = matrix[3];

    rocComplex temp_amp0 = amp0;
    rocComplex temp_amp1 = amp1;

    amp0.x = M00.x * temp_amp0.x - M00.y * temp_amp0.y + M01.x * temp_amp1.x - M01.y * temp_amp1.y;
    amp0.y = M00.x * temp_amp0.y + M00.y * temp_amp0.x + M01.x * temp_amp1.y + M01.y * temp_amp1.x;

    amp1.x = M10.x * temp_amp0.x - M10.y * temp_amp0.y + M11.x * temp_amp1.x - M11.y * temp_amp1.y;
    amp1.y = M10.x * temp_amp0.y + M10.y * temp_amp0.x + M11.x * temp_amp1.y + M11.y * temp_amp1.x;
}

// Generic kernel for applying an arbitrary 1-qubit unitary matrix
// Matrix is read from __constant__ memory symbol const_single_q_matrix
__global__ void apply_single_qubit_generic_matrix_kernel(rocComplex* state,
                                                         unsigned numQubits,
                                                         unsigned targetQubit) { // Matrix parameter removed
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit; // Stride between paired amplitudes

    // Each thread processes one pair of amplitudes
    // Grid Dim should be N / 2
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate base index for the pair, ensuring we don't go out of bounds
    // The pattern of indices repeats every 2*k elements.
    // tid gives the "pair number"
    // base_idx0 = (tid / k) * (2 * k) + (tid % k)
    // base_idx1 = base_idx0 + k
    // Example for targetQubit=1 (k=2), N=8:
    // tid=0: (0/2)*(4) + (0%2) = 0. idx0=0, idx1=2
    // tid=1: (1/2)*(4) + (1%2) = 1. idx0=1, idx1=3
    // tid=2: (2/2)*(4) + (2%2) = 4. idx0=4, idx1=6
    // tid=3: (3/2)*(4) + (3%2) = 5. idx0=5, idx1=7
    
    size_t outer_loop_factor = 1ULL << (targetQubit + 1); // 2k
    size_t inner_loop_factor = k;

    size_t group_idx = tid / inner_loop_factor; // Which group of 2k elements
    size_t in_group_idx = tid % inner_loop_factor; // Offset within the first k elements of the group

    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) { // Ensure both indices are within bounds
        rocComplex amp0 = state[idx0];
        rocComplex amp1 = state[idx1];
        // apply_1q_matrix reads from const_single_q_matrix directly
        apply_1q_matrix(amp0, amp1, const_single_q_matrix);
        state[idx0] = amp0;
        state[idx1] = amp1;
    }
}


// --- Specific Gate Kernels (can call the generic one or be specialized) ---

// Pauli-X Gate
__global__ void apply_X_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit) {
    // X = [[0, 1], [1, 0]]
    // M00=0, M10=1, M01=1, M11=0
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex temp = state[idx0];
        state[idx0] = state[idx1];
        state[idx1] = temp;
    }
}

// Pauli-Y Gate
__global__ void apply_Y_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit) {
    // Y = [[0, -i], [i, 0]] -> M00=0, M10={0,1}, M01={0,-1}, M11=0
    // amp0_new = -i * amp1_old
    // amp1_new =  i * amp0_old
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex amp0_old = state[idx0];
        rocComplex amp1_old = state[idx1];

        // amp0_new = -i * amp1_old = {amp1_old.y, -amp1_old.x}
#ifdef ROCQ_PRECISION_DOUBLE
        state[idx0].x = amp1_old.y;
        state[idx0].y = -amp1_old.x;
        // amp1_new = i * amp0_old = {-amp0_old.y, amp0_old.x}
        state[idx1].x = -amp0_old.y;
        state[idx1].y = amp0_old.x;
#else
        state[idx0].x = amp1_old.y;
        state[idx0].y = -amp1_old.x;
        state[idx1].x = -amp0_old.y;
        state[idx1].y = amp0_old.x;
#endif
    }
}

// Pauli-Z Gate
__global__ void apply_Z_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit) {
    // Z = [[1, 0], [0, -1]] -> M00=1, M10=0, M01=0, M11=-1
    // amp0_new = amp0_old
    // amp1_new = -amp1_old
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx; // This is the 0-component, not affected
    size_t idx1 = idx0 + inner_loop_factor; // This is the 1-component, negated

    if (idx1 < N) { // Only need to check idx1 as idx0 will also be valid
        state[idx1].x = -state[idx1].x;
        state[idx1].y = -state[idx1].y;
    }
}

// Hadamard Gate
__global__ void apply_H_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit) {
    // H = 1/sqrt(2) * [[1, 1], [1, -1]]
    // amp0_new = (amp0_old + amp1_old) / sqrt(2)
    // amp1_new = (amp0_old - amp1_old) / sqrt(2)
#ifdef ROCQ_PRECISION_DOUBLE
    const real_t inv_sqrt2 = 1.0 / sqrt(2.0);
#else
    const real_t inv_sqrt2 = 1.0f / sqrtf(2.0f);
#endif
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex amp0_old = state[idx0];
        rocComplex amp1_old = state[idx1];

        state[idx0].x = (amp0_old.x + amp1_old.x) * inv_sqrt2;
        state[idx0].y = (amp0_old.y + amp1_old.y) * inv_sqrt2;

        state[idx1].x = (amp0_old.x - amp1_old.x) * inv_sqrt2;
        state[idx1].y = (amp0_old.y - amp1_old.y) * inv_sqrt2;
    }
}

// S Gate (Phase Gate)
__global__ void apply_S_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit) {
    // S = [[1, 0], [0, i]]
    // amp0_new = amp0_old
    // amp1_new = i * amp1_old = {-amp1_old.y, amp1_old.x}
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex amp1_old = state[idx1];
#ifdef ROCQ_PRECISION_DOUBLE
        state[idx1].x = -amp1_old.y;
        state[idx1].y = amp1_old.x;
#else
        state[idx1].x = -amp1_old.y;
        state[idx1].y = amp1_old.x;
#endif
    }
}

// S Dagger Gate
__global__ void apply_Sdg_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit) {
    // Sdg = [[1, 0], [0, -i]]
    // amp0_new = amp0_old
    // amp1_new = -i * amp1_old = {amp1_old.y, -amp1_old.x}
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex amp1_old = state[idx1];
#ifdef ROCQ_PRECISION_DOUBLE
        state[idx1].x = amp1_old.y;
        state[idx1].y = -amp1_old.x;
#else
        state[idx1].x = amp1_old.y;
        state[idx1].y = -amp1_old.x;
#endif
    }
}


// T Gate
__global__ void apply_T_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit) {
    // T = [[1, 0], [0, exp(i*pi/4)]]
    // exp(i*pi/4) = cos(pi/4) + i*sin(pi/4) = 1/sqrt(2) + i/sqrt(2)
#ifdef ROCQ_PRECISION_DOUBLE
    const real_t val = 1.0 / sqrt(2.0);
    rocComplex phase_factor = {val, val};
#else
    const real_t val = 1.0f / sqrtf(2.0f);
    rocComplex phase_factor = {val, val};
#endif
    
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex amp1_old = state[idx1];
        state[idx1].x = phase_factor.x * amp1_old.x - phase_factor.y * amp1_old.y;
        state[idx1].y = phase_factor.x * amp1_old.y + phase_factor.y * amp1_old.x;
    }
}

// Rx(theta) Gate
__global__ void apply_Rx_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, real_t theta) {
    // Rx(theta) = [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]
    // M00 = cos(t/2), M01 = {0, -sin(t/2)}, M10 = {0, -sin(t/2)}, M11 = cos(t/2)
#ifdef ROCQ_PRECISION_DOUBLE
    real_t cos_half_theta = cos(theta / 2.0);
    real_t sin_half_theta = sin(theta / 2.0);
    rocComplex M00 = {cos_half_theta, 0.0};
    rocComplex M01 = {0.0, -sin_half_theta};
    rocComplex M10 = {0.0, -sin_half_theta};
    rocComplex M11 = {cos_half_theta, 0.0};
#else
    real_t cos_half_theta = cosf(theta / 2.0f);
    real_t sin_half_theta = sinf(theta / 2.0f);
    rocComplex M00 = {cos_half_theta, 0.0f};
    rocComplex M01 = {0.0f, -sin_half_theta};
    rocComplex M10 = {0.0f, -sin_half_theta};
    rocComplex M11 = {cos_half_theta, 0.0f};
#endif
    
    // This can be directly passed to the generic kernel as well
    // For now, let's use the structure of other specialized kernels
    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex amp0_old = state[idx0];
        rocComplex amp1_old = state[idx1];

        state[idx0].x = M00.x * amp0_old.x - M00.y * amp0_old.y + M01.x * amp1_old.x - M01.y * amp1_old.y;
        state[idx0].y = M00.x * amp0_old.y + M00.y * amp0_old.x + M01.x * amp1_old.y + M01.y * amp1_old.x;

        state[idx1].x = M10.x * amp0_old.x - M10.y * amp0_old.y + M11.x * amp1_old.x - M11.y * amp1_old.y;
        state[idx1].y = M10.x * amp0_old.y + M10.y * amp0_old.x + M11.x * amp1_old.y + M11.y * amp1_old.x;
    }
}

// Ry(theta) Gate
__global__ void apply_Ry_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, real_t theta) {
    // Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    // M00 = cos(t/2), M01 = -sin(t/2), M10 = sin(t/2), M11 = cos(t/2)
#ifdef ROCQ_PRECISION_DOUBLE
    real_t cos_half_theta = cos(theta / 2.0);
    real_t sin_half_theta = sin(theta / 2.0);
    rocComplex M00 = {cos_half_theta, 0.0};
    rocComplex M01 = {-sin_half_theta, 0.0};
    rocComplex M10 = {sin_half_theta, 0.0};
    rocComplex M11 = {cos_half_theta, 0.0};
#else
    real_t cos_half_theta = cosf(theta / 2.0f);
    real_t sin_half_theta = sinf(theta / 2.0f);
    rocComplex M00 = {cos_half_theta, 0.0f};
    rocComplex M01 = {-sin_half_theta, 0.0f};
    rocComplex M10 = {sin_half_theta, 0.0f};
    rocComplex M11 = {cos_half_theta, 0.0f};
#endif

    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx1 < N) {
        rocComplex amp0_old = state[idx0];
        rocComplex amp1_old = state[idx1];

        state[idx0].x = M00.x * amp0_old.x + M01.x * amp1_old.x; // Simplified due to real matrix
        state[idx0].y = M00.x * amp0_old.y + M01.x * amp1_old.y;

        state[idx1].x = M10.x * amp0_old.x + M11.x * amp1_old.x;
        state[idx1].y = M10.x * amp0_old.y + M11.x * amp1_old.y;
    }
}

// Rz(theta) Gate (Phase shift)
__global__ void apply_Rz_kernel(rocComplex* state, unsigned numQubits, unsigned targetQubit, real_t theta) {
    // Rz(theta) = [[exp(-it/2), 0], [0, exp(it/2)]]
    // amp0_new = exp(-it/2) * amp0_old
    // amp1_new = exp(it/2) * amp1_old
#ifdef ROCQ_PRECISION_DOUBLE
    real_t cos_half_theta = cos(theta / 2.0);
    real_t sin_half_theta = sin(theta / 2.0);
    rocComplex phase_neg = {cos_half_theta, -sin_half_theta}; // e^(-it/2)
    rocComplex phase_pos = {cos_half_theta, sin_half_theta};  // e^(it/2)
#else
    real_t cos_half_theta = cosf(theta / 2.0f);
    real_t sin_half_theta = sinf(theta / 2.0f);
    rocComplex phase_neg = {cos_half_theta, -sin_half_theta}; // e^(-it/2)
    rocComplex phase_pos = {cos_half_theta, sin_half_theta};  // e^(it/2)
#endif

    size_t N = 1ULL << numQubits;
    size_t k = 1ULL << targetQubit;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t outer_loop_factor = 1ULL << (targetQubit + 1);
    size_t inner_loop_factor = k;
    size_t group_idx = tid / inner_loop_factor;
    size_t in_group_idx = tid % inner_loop_factor;
    size_t idx0 = group_idx * outer_loop_factor + in_group_idx;
    size_t idx1 = idx0 + inner_loop_factor;

    if (idx0 < N) { // Apply to amp0
        rocComplex amp0_old = state[idx0];
        state[idx0].x = phase_neg.x * amp0_old.x - phase_neg.y * amp0_old.y;
        state[idx0].y = phase_neg.x * amp0_old.y + phase_neg.y * amp0_old.x;
    }
    if (idx1 < N) { // Apply to amp1
        rocComplex amp1_old = state[idx1];
        state[idx1].x = phase_pos.x * amp1_old.x - phase_pos.y * amp1_old.y;
        state[idx1].y = phase_pos.x * amp1_old.y + phase_pos.y * amp1_old.x;
    }
}
