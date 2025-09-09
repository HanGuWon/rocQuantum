#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

__global__ void apply_single_qubit_gate(
    hipDoubleComplex* state_vector, 
    unsigned int num_qubits, 
    unsigned int target_qubit, 
    const hipDoubleComplex* unitary) 
{
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = 1 << target_qubit;
    unsigned int low_mask = stride - 1;
    unsigned int high_mask = ~low_mask;
    unsigned int idx0 = (thread_id & low_mask) | ((thread_id & high_mask) << 1);
    unsigned int idx1 = idx0 | stride;

    hipDoubleComplex amp0 = state_vector[idx0];
    hipDoubleComplex amp1 = state_vector[idx1];

    hipDoubleComplex u00 = unitary[0], u01 = unitary[1], u10 = unitary[2], u11 = unitary[3];

    state_vector[idx0] = hipCadd(hipCmul(u00, amp0), hipCmul(u01, amp1));
    state_vector[idx1] = hipCadd(hipCmul(u10, amp0), hipCmul(u11, amp1));
}

__global__ void apply_cnot_gate(
    hipDoubleComplex* state_vector,
    unsigned int num_qubits,
    unsigned int control_qubit,
    unsigned int target_qubit)
{
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int control_stride = 1 << control_qubit;
    unsigned int target_stride = 1 << target_qubit;

    // Create a mask that has all bits set except control and target
    unsigned int mask = ~ (control_stride | target_stride);

    // Map thread ID to the part of the index that is not control or target
    unsigned int base_idx_masked = (thread_id & (target_stride - 1)) | ((thread_id & ~(target_stride - 1)) << 1);
    base_idx_masked = (base_idx_masked & (control_stride - 1)) | ((base_idx_masked & ~(control_stride - 1)) << 1);
    
    // We only care about states where the control bit is 1
    unsigned int idx0 = base_idx_masked | control_stride;
    unsigned int idx1 = idx0 | target_stride;

    // Swap the amplitudes
    hipDoubleComplex temp = state_vector[idx0];
    state_vector[idx0] = state_vector[idx1];
    state_vector[idx1] = temp;
}


__global__ void calculate_probabilities(
    const hipDoubleComplex* state_vector,
    double* probabilities,
    size_t state_vec_size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < state_vec_size) {
        probabilities[idx] = hipCreal(state_vector[idx]) * hipCreal(state_vector[idx]) + 
                             hipCimag(state_vector[idx]) * hipCimag(state_vector[idx]);
    }
}