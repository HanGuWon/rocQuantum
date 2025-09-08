#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iomanip>

#include <hip/hip_runtime.h>

// Include the header for the function we are testing
#include "rocquantum/PermutationKernels.h"
// Include complex types
#include "rocquantum/hipStateVec.h"

// Error checking utility
#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + " in file " + __FILE__ + " at line " + std::to_string(__LINE__)); \
    } \
} while(0)

// =================================================================
// 1. TEST FILE HEADER & SETUP
// =================================================================

/**
 * @brief Initializes a host tensor with sequential complex values.
 */
template<typename T>
void init_tensor(std::vector<T>& tensor) {
    using real_t = typename T::value_type;
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = T{static_cast<real_t>(i), static_cast<real_t>(i)};
    }
}

/**
 * @brief Compares two host tensors for near-exact equality.
 */
template<typename T>
bool compare_tensors(const std::vector<T>& tensor_a, const std::vector<T>& tensor_b, const std::string& test_name) {
    using real_t = typename T::value_type;
    if (tensor_a.size() != tensor_b.size()) {
        std::cerr << "\n[FAIL] " << test_name << " - Tensor sizes mismatch!" << std::endl;
        return false;
    }
    const real_t epsilon = 1e-6;
    for (size_t i = 0; i < tensor_a.size(); ++i) {
        if (std::abs(tensor_a[i].x - tensor_b[i].x) > epsilon || std::abs(tensor_a[i].y - tensor_b[i].y) > epsilon) {
            std::cerr << "\n[FAIL] " << test_name << " - Mismatch at index " << i << ".\n"
                      << "       Expected: (" << tensor_a[i].x << ", " << tensor_a[i].y << ")\n"
                      << "       Got:      (" << tensor_b[i].x << ", " << tensor_b[i].y << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// =================================================================
// 2. CPU REFERENCE IMPLEMENTATION
// =================================================================

/**
 * @brief Performs an N-dimensional tensor permutation on the CPU.
 */
template<typename T>
void permute_tensor_cpu(
    std::vector<T>& output_tensor,
    const std::vector<T>& input_tensor,
    const std::vector<long long>& input_dims,
    const std::vector<int>& permutation_map
) {
    int rank = input_dims.size();
    std::vector<long long> output_dims(rank);
    for(int i = 0; i < rank; ++i) {
        output_dims[i] = input_dims[permutation_map[i]];
    }

    std::vector<long long> input_strides(rank);
    input_strides[0] = 1;
    for (int i = 1; i < rank; ++i) {
        input_strides[i] = input_strides[i - 1] * input_dims[i - 1];
    }

    std::vector<long long> output_strides(rank);
    output_strides[0] = 1;
    for (int i = 1; i < rank; ++i) {
        output_strides[i] = output_strides[i - 1] * output_dims[i - 1];
    }

    long long total_elements = input_tensor.size();
    for (long long output_linear_idx = 0; output_linear_idx < total_elements; ++output_linear_idx) {
        long long remaining_idx = output_linear_idx;
        long long source_linear_idx = 0;
        std::vector<long long> coords(rank);

        // De-linearize output index to get multi-dimensional coordinates
        for (int i = rank - 1; i >= 0; --i) {
            coords[i] = remaining_idx / output_strides[i];
            remaining_idx %= output_strides[i];
        }

        // Re-linearize using permuted coordinates and input strides
        for (int i = 0; i < rank; ++i) {
            int source_dim = permutation_map[i];
            source_linear_idx += coords[i] * input_strides[source_dim];
        }
        output_tensor[output_linear_idx] = input_tensor[source_linear_idx];
    }
}

// =================================================================
// 3. CORE TEST FUNCTION
// =================================================================

/**
 * @brief Runs a single permutation test case for a given type, dimensions, and map.
 */
template<typename T>
bool run_test_case(const std::string& test_name, const std::vector<long long>& dims, const std::vector<int>& p_map) {
    std::cout << "Running: " << std::left << std::setw(50) << test_name << "... " << std::flush;

    // --- 1. Setup ---
    long long total_elements = 1;
    for(auto d : dims) total_elements *= d;

    std::vector<T> h_input(total_elements);
    std::vector<T> h_output_cpu(total_elements);
    std::vector<T> h_output_gpu(total_elements);

    init_tensor(h_input);

    T *d_input = nullptr, *d_output = nullptr;
    try {
        HIP_CHECK(hipMalloc(&d_input, total_elements * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, total_elements * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, h_input.data(), total_elements * sizeof(T), hipMemcpyHostToDevice));
    } catch (const std::runtime_error& e) {
        std::cerr << "\n[FAIL] " << test_name << " - HIP setup failed: " << e.what() << std::endl;
        if(d_input) hipFree(d_input);
        if(d_output) hipFree(d_output);
        return false;
    }

    // --- 2. Execute ---
    // CPU ground truth
    permute_tensor_cpu(h_output_cpu, h_input, dims, p_map);

    // GPU execution
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    launch_permute_tensor<T>(d_output, d_input, dims, p_map, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    // --- 3. Verify ---
    HIP_CHECK(hipMemcpy(h_output_gpu.data(), d_output, total_elements * sizeof(T), hipMemcpyDeviceToHost));
    bool success = compare_tensors(h_output_cpu, h_output_gpu, test_name);

    // --- 4. Cleanup ---
    hipFree(d_input);
    hipFree(d_output);
    hipStreamDestroy(stream);

    if (success) {
        std::cout << "[PASS]" << std::endl;
    }
    return success;
}

// =================================================================
// 4. main FUNCTION (TEST ORCHESTRATOR)
// =================================================================

int main() {
    int tests_passed = 0;
    int tests_total = 0;

    std::cout << "=================================================" << std::endl;
    std::cout << "  Running Unit Tests for permute_tensor_kernel   " << std::endl;
    std::cout << "=================================================" << std::endl;

    // --- Test Cases ---

    // Rank-2 (Matrix Transpose)
    if (run_test_case<rocComplex>("Rank-2 Transpose (rocComplex)", {128, 256}, {1, 0})) tests_passed++; tests_total++;
    if (run_test_case<rocDoubleComplex>("Rank-2 Transpose (rocDoubleComplex)", {128, 256}, {1, 0})) tests_passed++; tests_total++;

    // Rank-3 (Asymmetric)
    if (run_test_case<rocComplex>("Rank-3 Asymmetric (rocComplex)", {11, 13, 17}, {2, 0, 1})) tests_passed++; tests_total++;
    if (run_test_case<rocDoubleComplex>("Rank-3 Asymmetric (rocDoubleComplex)", {11, 13, 17}, {1, 2, 0})) tests_passed++; tests_total++;

    // Rank-4 (Symmetric)
    if (run_test_case<rocComplex>("Rank-4 Symmetric (rocComplex)", {8, 8, 8, 8}, {3, 0, 1, 2})) tests_passed++; tests_total++;
    if (run_test_case<rocDoubleComplex>("Rank-4 Symmetric (rocDoubleComplex)", {8, 8, 8, 8}, {1, 3, 0, 2})) tests_passed++; tests_total++;

    // Rank-3 (Identity)
    if (run_test_case<rocComplex>("Rank-3 Identity (rocComplex)", {10, 20, 30}, {0, 1, 2})) tests_passed++; tests_total++;
    if (run_test_case<rocDoubleComplex>("Rank-3 Identity (rocDoubleComplex)", {10, 20, 30}, {0, 1, 2})) tests_passed++; tests_total++;
    
    // Rank-5
    if (run_test_case<rocComplex>("Rank-5 (rocComplex)", {2, 3, 4, 5, 6}, {4, 3, 2, 1, 0})) tests_passed++; tests_total++;


    std::cout << "=================================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << " / " << tests_total << " passed." << std::endl;
    std::cout << "=================================================" << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}
