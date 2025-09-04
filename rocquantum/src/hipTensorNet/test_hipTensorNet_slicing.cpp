#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "rocquantum/hipTensorNet.h"
#include "rocquantum/rocTensorUtil.h"
#include "rocquantum/rocWorkspaceManager.h"
#include "rocquantum/hipStateVec.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// --- Test Infrastructure (Copied from test_hipTensorNet_rocTensorUtil.cpp for consistency) ---

#define HIP_ASSERT(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("HIP Error"); \
    } \
} while (0)

#define ROCBLAS_ASSERT(x) do { \
    rocblas_status err = x; \
    if (err != rocblas_status_success) { \
        std::cerr << "rocBLAS Error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("rocBLAS Error"); \
    } \
} while (0)

#define ASSERT_EQ(val1, val2, test_name) do { \
    if ((val1) != (val2)) { \
        std::cerr << "Assertion failed in " << (test_name) << ": " << (val1) << " != " << (val2) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while (0)

#define ASSERT_TRUE(cond, test_name) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed in " << (test_name) << ": condition is false at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while (0)

const float COMPLEX_EPSILON = 1e-5f;

bool compare_rocComplex(rocComplex c1, rocComplex c2, float epsilon = COMPLEX_EPSILON) {
    return (std::abs(c1.x - c2.x) < epsilon) && (std::abs(c1.y - c2.y) < epsilon);
}

#define ASSERT_COMPLEX_VEC_EQ(vec1_h, vec2_d, size, test_name) do { \
    std::vector<rocComplex> vec2_h(size); \
    HIP_ASSERT(hipMemcpy(vec2_h.data(), vec2_d, size * sizeof(rocComplex), hipMemcpyDeviceToHost)); \
    for (size_t i = 0; i < size; ++i) { \
        if (!compare_rocComplex(vec1_h[i], vec2_h[i])) { \
            std::cerr << "Assertion failed in " << (test_name) << " at index " << i << ": (" \
                      << vec1_h[i].x << "," << vec1_h[i].y << ") != (" \
                      << vec2_h[i].x << "," << vec2_h[i].y << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } \
} while(0)

typedef bool (*TestFunc)();
std::vector<std::pair<std::string, TestFunc>> tests;
#define ADD_TEST(name) tests.push_back({#name, name})

void RUN_ALL_TESTS() {
    int passed_count = 0;
    int failed_count = 0;
    std::cout << "Running " << tests.size() << " tests from test_hipTensorNet_slicing..." << std::endl;
    for (const auto& test_pair : tests) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Running test: " << test_pair.first << "..." << std::endl;
        bool result = false;
        try {
            result = test_pair.second();
        } catch (const std::runtime_error& e) {
            std::cerr << "Test " << test_pair.first << " threw an exception: " << e.what() << std::endl;
            result = false;
        }
        if (result) {
            std::cout << "Test " << test_pair.first << ": PASSED" << std::endl;
            passed_count++;
        } else {
            std::cout << "Test " << test_pair.first << ": FAILED" << std::endl;
            failed_count++;
        }
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "All slicing tests completed." << std::endl;
    std::cout << "Passed: " << passed_count << ", Failed: " << failed_count << std::endl;
}

rocblas_handle blas_handle = nullptr;
hipStream_t test_stream = 0;

void setup_global_test_resources() {
    HIP_ASSERT(hipStreamCreate(&test_stream));
    ROCBLAS_ASSERT(rocblas_create_handle(&blas_handle));
    ROCBLAS_ASSERT(rocblas_set_stream(blas_handle, test_stream));
}

void teardown_global_test_resources() {
    if (blas_handle) ROCBLAS_ASSERT(rocblas_destroy_handle(blas_handle));
    if (test_stream) HIP_ASSERT(hipStreamDestroy(test_stream));
}

// --- Helper Functions ---

rocquantum::util::rocTensor create_gpu_tensor(const std::vector<long long>& dims, const std::vector<std::string>& labels, const std::vector<rocComplex>& h_data) {
    rocquantum::util::rocTensor tensor;
    tensor.dims_ = dims;
    tensor.labels_ = labels;
    rocquantum::util::rocTensorAllocate(&tensor);
    HIP_ASSERT(hipMemcpy(tensor.data_, h_data.data(), h_data.size() * sizeof(rocComplex), hipMemcpyHostToDevice));
    return tensor;
}

// --- Slicing Test ---

bool test_slicing_execution_correctness() {
    // Network: A(a,b) * B(b,c) * C(c,d) -> Result(a,d)
    // Dimensions: a=2, b=8, c=10, d=3.
    // A(2,8), B(8,10), C(10,3)
    // Optimal path is A*(B*C).
    // Intermediate B*C is a tensor with shape (b,d) -> (8,3), size 24 elements.
    // Size in bytes = 24 * sizeof(rocComplex) = 24 * 8 = 192 bytes.
    // We will set memory limit to force slicing on this intermediate.

    // 1. Prepare Tensors
    std::vector<rocComplex> h_A_data(2 * 8);
    for(size_t i = 0; i < h_A_data.size(); ++i) h_A_data[i] = {(float)i, 0.1f * (float)i};
    
    std::vector<rocComplex> h_B_data(8 * 10);
    for(size_t i = 0; i < h_B_data.size(); ++i) h_B_data[i] = {(float)i * 0.5f, 0.2f};

    std::vector<rocComplex> h_C_data(10 * 3);
    for(size_t i = 0; i < h_C_data.size(); ++i) h_C_data[i] = {0.8f, (float)i * 0.1f};

    rocquantum::util::rocTensor tensorA = create_gpu_tensor({2, 8}, {"a", "b"}, h_A_data);
    rocquantum::util::rocTensor tensorB = create_gpu_tensor({8, 10}, {"b", "c"}, h_B_data);
    rocquantum::util::rocTensor tensorC = create_gpu_tensor({10, 3}, {"c", "d"}, h_C_data);

    // 2. Run contraction WITHOUT slicing to get the ground truth result
    std::cout << "  Computing ground truth result (no slicing)..." << std::endl;
    rocquantum::util::rocTensor result_no_slicing;
    {
        rocquantum::TensorNetwork<rocComplex> tn_no_slice;
        tn_no_slice.add_tensor(tensorA);
        tn_no_slice.add_tensor(tensorB);
        tn_no_slice.add_tensor(tensorC);
        
        hipTensorNetContractionOptimizerConfig_t config = {};
        config.memory_limit_bytes = 1024 * 1024; // 1 MB, large enough to prevent slicing

        rocqStatus_t status = tn_no_slice.contract(&config, &result_no_slicing, blas_handle, test_stream);
        ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "slicing_test.ground_truth_contraction_status");
        HIP_ASSERT(hipStreamSynchronize(test_stream));
    }
    
    // 3. Run contraction WITH slicing
    std::cout << "  Computing result with slicing forced..." << std::endl;
    rocquantum::util::rocTensor result_with_slicing;
    {
        rocquantum::TensorNetwork<rocComplex> tn_slice;
        tn_slice.add_tensor(tensorA);
        tn_slice.add_tensor(tensorB);
        tn_slice.add_tensor(tensorC);

        hipTensorNetContractionOptimizerConfig_t config = {};
        // Force slicing on the B*C contraction.
        // Required memory for B(8,10) + C(10,3) + BC(8,3) is roughly
        // (80 + 30 + 24) * 8 bytes = 134 * 8 = 1072 bytes.
        // Setting limit to 1KB (1024 bytes) should trigger slicing.
        config.memory_limit_bytes = 1024; 

        rocqStatus_t status = tn_slice.contract(&config, &result_with_slicing, blas_handle, test_stream);
        ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "slicing_test.sliced_contraction_status");
        HIP_ASSERT(hipStreamSynchronize(test_stream));
    }

    // 4. Compare results
    std::cout << "  Comparing results..." << std::endl;
    ASSERT_EQ(result_no_slicing.get_element_count(), result_with_slicing.get_element_count(), "slicing_test.result_element_count");
    
    size_t num_elements = result_no_slicing.get_element_count();
    std::vector<rocComplex> h_result_no_slicing(num_elements);
    HIP_ASSERT(hipMemcpy(h_result_no_slicing.data(), result_no_slicing.data_, num_elements * sizeof(rocComplex), hipMemcpyDeviceToHost));

    ASSERT_COMPLEX_VEC_EQ(h_result_no_slicing, result_with_slicing.data_, num_elements, "slicing_test.result_data_matches");

    // 5. Cleanup
    rocquantum::util::rocTensorFree(&tensorA);
    rocquantum::util::rocTensorFree(&tensorB);
    rocquantum::util::rocTensorFree(&tensorC);
    rocquantum::util::rocTensorFree(&result_no_slicing);
    rocquantum::util::rocTensorFree(&result_with_slicing);

    return true;
}


int main() {
    setup_global_test_resources();
    ADD_TEST(test_slicing_execution_correctness);
    RUN_ALL_TESTS();
    teardown_global_test_resources();
    return 0;
}
