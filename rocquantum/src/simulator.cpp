#include "rocquantum/include/rocquantum.h"
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <map>
#include <random>

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

// Forward declarations
__global__ void apply_single_qubit_gate(hipDoubleComplex* sv, unsigned int nq, unsigned int t, const hipDoubleComplex* u);
__global__ void apply_cnot_gate(hipDoubleComplex* sv, unsigned int nq, unsigned int c, unsigned int t);
__global__ void calculate_probabilities(const hipDoubleComplex* sv, double* probs, size_t size);

#define HIP_CHECK(error) \
    if (error != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(error)); \
    }

namespace rocquantum {

// Helper to generate rotation matrices
static std::vector<std::complex<double>> get_rotation_matrix(const std::string& gate_name, double angle) {
    if (gate_name == "RX") {
        return {{cos(angle/2), 0}, {0, -sin(angle/2)}, {0, -sin(angle/2)}, {cos(angle/2), 0}};
    }
    if (gate_name == "RY") {
        return {{cos(angle/2), 0}, {-sin(angle/2), 0}, {sin(angle/2), 0}, {cos(angle/2), 0}};
    }
    if (gate_name == "RZ") {
        return {{cos(-angle/2), -sin(-angle/2)}, {0, 0}, {0, 0}, {cos(angle/2), -sin(angle/2)}};
    }
    throw std::runtime_error("Invalid rotation gate name.");
}

static const std::map<std::string, std::vector<std::complex<double>>> predefined_gates = {
    {"H", {{1/sqrt(2),0}, {1/sqrt(2),0}, {1/sqrt(2),0}, {-1/sqrt(2),0}}},
    {"Hadamard", {{1/sqrt(2),0}, {1/sqrt(2),0}, {1/sqrt(2),0}, {-1/sqrt(2),0}}},
    {"PauliX", {{0,0}, {1,0}, {1,0}, {0,0}}},
    {"PauliY", {{0,0}, {0,-1}, {0,1}, {0,0}}},
    {"PauliZ", {{1,0}, {0,0}, {0,0}, {-1,0}}},
    {"Identity", {{1,0}, {0,0}, {0,0}, {1,0}}}
};

QuantumSimulator::QuantumSimulator(unsigned num_qubits) : num_qubits_(num_qubits), device_state_vector_(nullptr) {
    state_vec_size_ = 1 << num_qubits_;
    size_t mem_size = state_vec_size_ * sizeof(hipDoubleComplex);
    HIP_CHECK(hipMalloc(&device_state_vector_, mem_size));
    this->reset();
}

QuantumSimulator::~QuantumSimulator() {
    if (device_state_vector_) {
        hipFree(device_state_vector_);
    }
}

void QuantumSimulator::reset() {
    size_t mem_size = state_vec_size_ * sizeof(hipDoubleComplex);
    HIP_CHECK(hipMemset(device_state_vector_, 0, mem_size));
    hipDoubleComplex val = {1.0, 0.0};
    HIP_CHECK(hipMemcpy(device_state_vector_, &val, sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
}

void QuantumSimulator::apply_gate(const std::string& gate_name, const std::vector<unsigned>& targets, const std::vector<double>& params) {
    if (gate_name == "CNOT" || gate_name == "cx") {
        if (targets.size() != 2) throw std::runtime_error("CNOT requires 2 target qubits.");
        unsigned int control = targets[0], target = targets[1];
        
        unsigned int num_threads = 1 << (num_qubits_ - 2);
        unsigned int threads_per_block = 256;
        unsigned int blocks = (num_threads + threads_per_block - 1) / threads_per_block;
        
        hipLaunchKernelGGL(apply_cnot_gate, dim3(blocks), dim3(threads_per_block), 0, 0, 
                           device_state_vector_, num_qubits_, control, target);
        HIP_CHECK(hipDeviceSynchronize());
        return;
    }

    if (gate_name == "RX" || gate_name == "RY" || gate_name == "RZ") {
        if (params.empty()) throw std::runtime_error("Rotation gate requires an angle parameter.");
        auto matrix = get_rotation_matrix(gate_name, params[0]);
        this->apply_matrix(matrix, targets);
        return;
    }

    if (predefined_gates.count(gate_name)) {
        this->apply_matrix(predefined_gates.at(gate_name), targets);
        return;
    }
    
    throw std::runtime_error("Gate '" + gate_name + "' is not supported.");
}

void QuantumSimulator::apply_matrix(const std::vector<std::complex<double>>& matrix, const std::vector<unsigned>& targets) {
    if (targets.size() != 1) throw std::runtime_error("Only single-qubit matrices are supported.");
    if (matrix.size() != 4) throw std::runtime_error("Matrix must have 4 elements.");
    
    hipDoubleComplex* device_unitary;
    HIP_CHECK(hipMalloc(&device_unitary, 4 * sizeof(hipDoubleComplex)));
    HIP_CHECK(hipMemcpy(device_unitary, matrix.data(), 4 * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));

    unsigned int num_threads = 1 << (num_qubits_ - 1);
    unsigned int threads_per_block = 256;
    unsigned int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(apply_single_qubit_gate, dim3(blocks), dim3(threads_per_block), 0, 0, 
                       device_state_vector_, num_qubits_, targets[0], device_unitary);
    
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(device_unitary));
}

std::vector<std::complex<double>> QuantumSimulator::get_statevector() {
    std::vector<std::complex<double>> host_state_vector(state_vec_size_);
    size_t mem_size = state_vec_size_ * sizeof(hipDoubleComplex);
    HIP_CHECK(hipMemcpy(host_state_vector.data(), device_state_vector_, mem_size, hipMemcpyDeviceToHost));
    return host_state_vector;
}

std::vector<long long> QuantumSimulator::measure(const std::vector<unsigned>& qubits, int shots) {
    // 1. Calculate probabilities on GPU
    double* device_probabilities;
    size_t probs_mem_size = state_vec_size_ * sizeof(double);
    HIP_CHECK(hipMalloc(&device_probabilities, probs_mem_size));

    unsigned int threads_per_block = 256;
    unsigned int blocks = (state_vec_size_ + threads_per_block - 1) / threads_per_block;
    hipLaunchKernelGGL(calculate_probabilities, dim3(blocks), dim3(threads_per_block), 0, 0,
                       device_state_vector_, device_probabilities, state_vec_size_);
    HIP_CHECK(hipDeviceSynchronize());

    // 2. Copy probabilities to host
    std::vector<double> host_probabilities(state_vec_size_);
    HIP_CHECK(hipMemcpy(host_probabilities.data(), device_probabilities, probs_mem_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(device_probabilities));

    // 3. Perform weighted sampling on the host
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<> dist(host_probabilities.begin(), host_probabilities.end());
    
    std::vector<long long> results;
    results.reserve(shots);
    for (int i = 0; i < shots; ++i) {
        results.push_back(dist(gen));
    }
    
    return results;
}

} // namespace rocquantum
