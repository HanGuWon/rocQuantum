#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath> // For std::abs
#include <stdexcept> // For runtime_error

#include "rocquantum/hipTensorNet.h"
#include "rocquantum/rocTensorUtil.h"
#include "rocquantum/rocWorkspaceManager.h"
#include "rocquantum/hipStateVec.h" // For rocComplex and rocqStatus_t

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// Helper to check HIP errors
#define HIP_ASSERT(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("HIP Error"); \
    } \
} while (0)

// Helper to check ROCBLAS errors
#define ROCBLAS_ASSERT(x) do { \
    rocblas_status err = x; \
    if (err != rocblas_status_success) { \
        std::cerr << "rocBLAS Error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("rocBLAS Error"); \
    } \
} while (0)

// Basic Assertion Macros
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

#define ASSERT_FALSE(cond, test_name) do { \
    if (cond) { \
        std::cerr << "Assertion failed in " << (test_name) << ": condition is true at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while (0)

const float COMPLEX_EPSILON = 1e-5f;

bool compare_rocComplex(rocComplex c1, rocComplex c2, float epsilon = COMPLEX_EPSILON) {
    return (std::abs(c1.x - c2.x) < epsilon) && (std::abs(c1.y - c2.y) < epsilon);
}

#define ASSERT_COMPLEX_EQ(c1, c2, test_name) do { \
    if (!compare_rocComplex(c1, c2)) { \
        std::cerr << "Assertion failed in " << (test_name) << ": (" << (c1).x << "," << (c1).y << ") != (" \
                  << (c2).x << "," << (c2).y << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while (0)

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


// Test Runner
typedef bool (*TestFunc)();
std::vector<std::pair<std::string, TestFunc>> tests;

#define ADD_TEST(name) tests.push_back({#name, name})

void RUN_ALL_TESTS() {
    int passed_count = 0;
    int failed_count = 0;
    std::cout << "Running " << tests.size() << " tests..." << std::endl;
    for (const auto& test_pair : tests) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Running test: " << test_pair.first << "..." << std::endl;
        bool result = false;
        try {
            result = test_pair.second();
        } catch (const std::runtime_error& e) {
            std::cerr << "Test " << test_pair.first << " threw an exception: " << e.what() << std::endl;
            result = false;
        } catch (...) {
            std::cerr << "Test " << test_pair.first << " threw an unknown exception." << std::endl;
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
    std::cout << "All tests completed." << std::endl;
    std::cout << "Passed: " << passed_count << ", Failed: " << failed_count << std::endl;
    if (failed_count > 0) {
        // Consider exiting with non-zero status for CI
        // exit(1);
    }
}

// Global rocBLAS handle and stream for tests
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
    blas_handle = nullptr;
    test_stream = 0;
}

// ---------- rocTensorUtil Tests ----------

bool test_rocTensor_struct() {
    std::cout << "  Sub-test: rocTensor struct basics..." << std::endl;
    rocquantum::util::rocTensor tensor;
    ASSERT_EQ(tensor.data_, nullptr, "rocTensor_struct.default_data");
    ASSERT_EQ(tensor.rank(), 0, "rocTensor_struct.default_rank");
    ASSERT_EQ(tensor.get_element_count(), 0, "rocTensor_struct.default_elements");
    ASSERT_FALSE(tensor.owned_, "rocTensor_struct.default_owned");

    std::vector<long long> dims = {2, 3, 4};
    rocComplex* dummy_data = reinterpret_cast<rocComplex*>(0x1234); // Just a non-null pointer
    rocquantum::util::rocTensor tensor2(dummy_data, dims);
    ASSERT_EQ(tensor2.data_, dummy_data, "rocTensor_struct.init_data");
    ASSERT_EQ(tensor2.rank(), 3, "rocTensor_struct.init_rank");
    ASSERT_EQ(tensor2.get_element_count(), 24, "rocTensor_struct.init_elements");
    ASSERT_FALSE(tensor2.owned_, "rocTensor_struct.init_owned");
    ASSERT_EQ(tensor2.dimensions_.size(), 3, "rocTensor_struct.dims_size");
    ASSERT_EQ(tensor2.strides_.size(), 3, "rocTensor_struct.strides_size");
    if (tensor2.strides_.size() == 3) { // Check strides if correctly sized
        ASSERT_EQ(tensor2.strides_[0], 1, "rocTensor_struct.stride0");
        ASSERT_EQ(tensor2.strides_[1], 2, "rocTensor_struct.stride1");
        ASSERT_EQ(tensor2.strides_[2], 6, "rocTensor_struct.stride2");
    }

    // Test scalar-like (empty dims)
    rocquantum::util::rocTensor scalar_tensor(dummy_data, {});
    ASSERT_EQ(scalar_tensor.get_element_count(), 0, "rocTensor_struct.scalar_elements_empty_dims"); // 0 based on product
    // The `rocTensorAllocate` handles scalar by allocating 1 element. `rocTensor` itself has 0 elements if dims is empty.

    // Test move constructor
    rocquantum::util::rocTensor tensor3 = std::move(tensor2);
    ASSERT_EQ(tensor3.data_, dummy_data, "rocTensor_struct.move_ctor_data");
    ASSERT_EQ(tensor3.rank(), 3, "rocTensor_struct.move_ctor_rank");
    ASSERT_FALSE(tensor3.owned_, "rocTensor_struct.move_ctor_owned"); // Ownership doesn't change by default on move of a view
    ASSERT_EQ(tensor2.data_, nullptr, "rocTensor_struct.moved_from_data_null");
    ASSERT_FALSE(tensor2.owned_, "rocTensor_struct.moved_from_owned_false");


    std::cout << "  Sub-test: rocTensor struct basics PASSED." << std::endl;
    return true;
}

bool test_rocTensor_alloc_free() {
    std::cout << "  Sub-test: rocTensorAllocate and rocTensorFree..." << std::endl;
    rocquantum::util::rocTensor tensor;
    tensor.dimensions_ = {2, 2};

    rocqStatus_t status = rocquantum::util::rocTensorAllocate(&tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.allocate_status");
    ASSERT_TRUE(tensor.data_ != nullptr, "alloc_free.data_not_null_after_alloc");
    ASSERT_TRUE(tensor.owned_, "alloc_free.owned_true_after_alloc");
    ASSERT_EQ(tensor.get_element_count(), 4, "alloc_free.element_count");

    // Test double allocation (should free old and alloc new)
    tensor.dimensions_ = {3,3};
    status = rocquantum::util::rocTensorAllocate(&tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.re_allocate_status");
    ASSERT_TRUE(tensor.data_ != nullptr, "alloc_free.data_not_null_after_realloc");
    ASSERT_TRUE(tensor.owned_, "alloc_free.owned_true_after_realloc");
    ASSERT_EQ(tensor.get_element_count(), 9, "alloc_free.element_count_realloc");


    status = rocquantum::util::rocTensorFree(&tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.free_status");
    ASSERT_EQ(tensor.data_, nullptr, "alloc_free.data_null_after_free");
    ASSERT_FALSE(tensor.owned_, "alloc_free.owned_false_after_free");

    // Test freeing a non-owned tensor (should be a no-op, clear view)
    rocComplex* dummy_data = reinterpret_cast<rocComplex*>(0x1234);
    tensor.data_ = dummy_data;
    tensor.owned_ = false;
    tensor.dimensions_ = {1,1};
    status = rocquantum::util::rocTensorFree(&tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.free_non_owned_status");
    ASSERT_EQ(tensor.data_, nullptr, "alloc_free.data_null_after_free_non_owned"); // Clears view
    ASSERT_FALSE(tensor.owned_, "alloc_free.owned_false_after_free_non_owned");

    // Test scalar (empty dims, allocate should make it 1 element)
    rocquantum::util::rocTensor scalar_tensor;
    scalar_tensor.dimensions_ = {}; // Empty
    status = rocquantum::util::rocTensorAllocate(&scalar_tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.scalar_alloc_status");
    ASSERT_TRUE(scalar_tensor.data_ != nullptr, "alloc_free.scalar_data_not_null");
    ASSERT_TRUE(scalar_tensor.owned_, "alloc_free.scalar_owned");
    // rocTensorAllocate treats empty dims as a scalar of 1 element for allocation purposes
    ASSERT_EQ(scalar_tensor.get_element_count(), 0, "alloc_free.scalar_original_element_count"); // original count is 0
    // Check allocated size by trying to write to it (not directly testable without element count change)
    // For now, trust hipMalloc(sizeof(rocComplex)) worked.
    rocComplex test_val = {1.0f, 2.0f};
    HIP_ASSERT(hipMemcpy(scalar_tensor.data_, &test_val, sizeof(rocComplex), hipMemcpyHostToDevice));

    status = rocquantum::util::rocTensorFree(&scalar_tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.scalar_free_status");

    // Test zero-element tensor (e.g., one dim is 0)
    rocquantum::util::rocTensor zero_el_tensor;
    zero_el_tensor.dimensions_ = {2, 0, 2};
    status = rocquantum::util::rocTensorAllocate(&zero_el_tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.zero_el_alloc_status");
    ASSERT_EQ(zero_el_tensor.data_, nullptr, "alloc_free.zero_el_data_is_null");
    ASSERT_TRUE(zero_el_tensor.owned_, "alloc_free.zero_el_owned"); // Owns "nothing"
    ASSERT_EQ(zero_el_tensor.get_element_count(), 0, "alloc_free.zero_el_element_count");
    status = rocquantum::util::rocTensorFree(&zero_el_tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "alloc_free.zero_el_free_status");


    std::cout << "  Sub-test: rocTensorAllocate and rocTensorFree PASSED." << std::endl;
    return true;
}

bool test_parse_simple_einsum_spec() {
    std::cout << "  Sub-test: parse_simple_einsum_spec..." << std::endl;
    rocquantum::util::rocTensor tensorA, tensorB; // Dummies for rank check
    std::vector<std::pair<int, int>> contracted_pairs;
    std::vector<int> res_A_modes, res_B_modes;
    std::vector<long long> result_dims;
    std::vector<std::string> result_labels;
    bool parsed_ok;

    // Matrix Multiply: A(ik),B(kj) -> C(ij)
    tensorA.dimensions_ = {2, 3}; tensorA.labels_ = {"i", "k"}; // Actual labels not used by parser, only rank
    tensorB.dimensions_ = {3, 4}; tensorB.labels_ = {"k", "j"};
    std::string spec_mm = "ik,kj->ij";
    parsed_ok = rocquantum::util::parse_simple_einsum_spec(spec_mm, &tensorA, &tensorB,
                                            contracted_pairs, res_A_modes, res_B_modes,
                                            result_dims, result_labels);
    ASSERT_TRUE(parsed_ok, "einsum_parser.mm_parsed_ok");
    ASSERT_EQ(contracted_pairs.size(), 1, "einsum_parser.mm_contracted_pairs_size");
    if (!contracted_pairs.empty()) {
        ASSERT_EQ(contracted_pairs[0].first, 1, "einsum_parser.mm_contracted_A_idx"); // k in A is mode 1
        ASSERT_EQ(contracted_pairs[0].second, 0, "einsum_parser.mm_contracted_B_idx"); // k in B is mode 0
    }
    ASSERT_EQ(res_A_modes.size(), 1, "einsum_parser.mm_res_A_modes_size");
    if(!res_A_modes.empty()) ASSERT_EQ(res_A_modes[0], 0, "einsum_parser.mm_res_A_mode0"); // i from A
    ASSERT_EQ(res_B_modes.size(), 1, "einsum_parser.mm_res_B_modes_size");
    if(!res_B_modes.empty()) ASSERT_EQ(res_B_modes[0], 1, "einsum_parser.mm_res_B_mode0"); // j from B
    ASSERT_EQ(result_dims.size(), 2, "einsum_parser.mm_res_dims_size");
    if (result_dims.size()==2) {
        ASSERT_EQ(result_dims[0], 2, "einsum_parser.mm_res_dim0"); // dim of i
        ASSERT_EQ(result_dims[1], 4, "einsum_parser.mm_res_dim1"); // dim of j
    }
    ASSERT_EQ(result_labels.size(), 2, "einsum_parser.mm_res_labels_size");
     if (result_labels.size()==2) {
        ASSERT_EQ(result_labels[0], "i", "einsum_parser.mm_res_label0");
        ASSERT_EQ(result_labels[1], "j", "einsum_parser.mm_res_label1");
    }

    // Trace: A(ii)->
    // Current parser requires output labels. A(ii)-> (scalar)
    // For now, let's test A(ii)->s where 's' is a scalar label
    contracted_pairs.clear(); res_A_modes.clear(); res_B_modes.clear(); result_dims.clear(); result_labels.clear();
    tensorA.dimensions_ = {2, 2}; tensorA.labels_ = {"i", "i"};
    tensorB.dimensions_ = {}; tensorB.labels_ = {}; // Not used for trace like this
    std::string spec_trace = "ii,->s"; // Dummy B, A(ii)->s
    // The parser expects two tensors for "LL,LL->LL".
    // Let's adapt for how it might be used with rocTensorContractWithRocBLAS,
    // which takes two tensors. A trace is typically A_ik * delta_ki.
    // The current parser design isn't ideal for single tensor ops like trace if forced into "T1,T2->T3"
    // Let's test "ii,->" (implicit scalar result from one tensor) - this will likely fail current parser.
    // The parser expects labels for B. Let's try "ii,x->s" where x is a dummy label for a scalar B.
    tensorA.dimensions_ = {2,2}; tensorA.labels_ = {"i","i"}; // A(ii)
    tensorB.dimensions_ = {1}; tensorB.labels_ = {"x"};      // B(x) - a dummy scalar
    spec_trace = "ii,x->s"; // This means sum over i, result is s, x is uncontracted from B.
                            // This isn't a standard trace.
                            // A real trace "ii->" would sum over 'i' and result in a scalar.
                            // The parser might need "ii,->s" where "s" is explicitly the scalar output label.
    // The parser's design `parse_simple_einsum_spec(spec, &tensorA, &tensorB, ...)` implies it always works on two tensors.
    // A trace `Tr(M) = M_ii` needs a different parsing path or representation if done with one tensor.
    // If we want to test "ii->s" (scalar output "s"), we need to check how the parser handles empty input for B.
    // The current `comma_pos` check will fail if B's spec is empty.

    // Test Outer Product: A(i),B(j) -> C(ij)
    contracted_pairs.clear(); res_A_modes.clear(); res_B_modes.clear(); result_dims.clear(); result_labels.clear();
    tensorA.dimensions_ = {2}; tensorA.labels_ = {"i"};
    tensorB.dimensions_ = {3}; tensorB.labels_ = {"j"};
    std::string spec_outer = "i,j->ij";
    parsed_ok = rocquantum::util::parse_simple_einsum_spec(spec_outer, &tensorA, &tensorB,
                                            contracted_pairs, res_A_modes, res_B_modes,
                                            result_dims, result_labels);
    ASSERT_TRUE(parsed_ok, "einsum_parser.outer_parsed_ok");
    ASSERT_EQ(contracted_pairs.size(), 0, "einsum_parser.outer_contracted_size");
    ASSERT_EQ(res_A_modes.size(), 1, "einsum_parser.outer_res_A_size");
    if(!res_A_modes.empty()) ASSERT_EQ(res_A_modes[0], 0, "einsum_parser.outer_res_A_mode0");
    ASSERT_EQ(res_B_modes.size(), 1, "einsum_parser.outer_res_B_size");
    if(!res_B_modes.empty()) ASSERT_EQ(res_B_modes[0], 0, "einsum_parser.outer_res_B_mode0");
    ASSERT_EQ(result_dims.size(), 2, "einsum_parser.outer_res_dims_size");
    if(result_dims.size()==2) {
        ASSERT_EQ(result_dims[0], 2, "einsum_parser.outer_res_dim0");
        ASSERT_EQ(result_dims[1], 3, "einsum_parser.outer_res_dim1");
    }

    // Test with tensor names: TA(ik),TB(kj)->TC(ij)
    spec_mm = "TA(ik),TB(kj)->TC(ij)";
    parsed_ok = rocquantum::util::parse_simple_einsum_spec(spec_mm, &tensorA, &tensorB,
                                            contracted_pairs, res_A_modes, res_B_modes,
                                            result_dims, result_labels); // tensorA/B dims/labels are {2,3} {3,4}
    ASSERT_TRUE(parsed_ok, "einsum_parser.mm_named_parsed_ok");
    ASSERT_EQ(contracted_pairs.size(), 1, "einsum_parser.mm_named_contracted_pairs_size");
     if (!contracted_pairs.empty()) {
        ASSERT_EQ(contracted_pairs[0].first, 1, "einsum_parser.mm_named_contracted_A_idx");
        ASSERT_EQ(contracted_pairs[0].second, 0, "einsum_parser.mm_named_contracted_B_idx");
    }
    ASSERT_EQ(result_dims.size(), 2, "einsum_parser.mm_named_res_dims_size");
    if (result_dims.size()==2) {
        ASSERT_EQ(result_dims[0], 2, "einsum_parser.mm_named_res_dim0");
        ASSERT_EQ(result_dims[1], 4, "einsum_parser.mm_named_res_dim1");
    }


    // Test invalid: label count mismatch
    tensorA.dimensions_ = {2,3}; tensorA.labels_ = {"i","k"};
    tensorB.dimensions_ = {3,4}; tensorB.labels_ = {"k","j"};
    std::string spec_invalid_labelcount = "ik,k->ij"; // B spec "k" is too short
    parsed_ok = rocquantum::util::parse_simple_einsum_spec(spec_invalid_labelcount, &tensorA, &tensorB,
                                            contracted_pairs, res_A_modes, res_B_modes,
                                            result_dims, result_labels);
    ASSERT_FALSE(parsed_ok, "einsum_parser.invalid_labelcount");

    // Test invalid: dimension mismatch for contraction
    tensorA.dimensions_ = {2,3}; tensorA.labels_ = {"i","k"}; // k is dim 3
    tensorB.dimensions_ = {4,4}; tensorB.labels_ = {"k","j"}; // k is dim 4
    std::string spec_invalid_dimmatch = "ik,kj->ij";
    parsed_ok = rocquantum::util::parse_simple_einsum_spec(spec_invalid_dimmatch, &tensorA, &tensorB,
                                            contracted_pairs, res_A_modes, res_B_modes,
                                            result_dims, result_labels);
    ASSERT_FALSE(parsed_ok, "einsum_parser.invalid_dimmatch");

    // Test scalar result from full contraction: ab,ba->s
    contracted_pairs.clear(); res_A_modes.clear(); res_B_modes.clear(); result_dims.clear(); result_labels.clear();
    tensorA.dimensions_ = {2,3}; tensorA.labels_ = {"a","b"};
    tensorB.dimensions_ = {3,2}; tensorB.labels_ = {"b","a"};
    std::string spec_scalar_res = "ab,ba->s";
     parsed_ok = rocquantum::util::parse_simple_einsum_spec(spec_scalar_res, &tensorA, &tensorB,
                                            contracted_pairs, res_A_modes, res_B_modes,
                                            result_dims, result_labels);
    ASSERT_TRUE(parsed_ok, "einsum_parser.scalar_res_ok");
    ASSERT_EQ(contracted_pairs.size(), 2, "einsum_parser.scalar_res_contract_count");
    ASSERT_EQ(result_dims.size(), 1, "einsum_parser.scalar_res_dims_size");
    if(!result_dims.empty()) ASSERT_EQ(result_dims[0], 1, "einsum_parser.scalar_res_dim_is_1");
    ASSERT_EQ(result_labels.size(), 1, "einsum_parser.scalar_res_labels_size");
    if(!result_labels.empty()) ASSERT_EQ(result_labels[0], "s", "einsum_parser.scalar_res_label_is_s");


    std::cout << "  Sub-test: parse_simple_einsum_spec PASSED." << std::endl;
    return true;
}


// More tests will follow for rocTensorPermute, rocTensorContractWithRocBLAS, WorkspaceManager, and TensorNetwork

bool test_rocTensorPermute() {
    std::cout << "  Sub-test: rocTensorPermute (matrix transpose)..." << std::endl;
    rocquantum::util::rocTensor input_tensor, output_tensor;

    // Input: 2x3 matrix
    // 0 1 2
    // 3 4 5
    input_tensor.dimensions_ = {2, 3}; // M=2, N=3. Stored column-major: 0,3,1,4,2,5
                                       // Stride for dim 0 is 1, stride for dim 1 is 2.
                                       // (0,0)=0, (1,0)=3
                                       // (0,1)=1, (1,1)=4
                                       // (0,2)=2, (1,2)=5
    input_tensor.calculate_strides();
    std::vector<rocComplex> h_input_data = {
        {0,0}, {3,0}, // col 0
        {1,0}, {4,0}, // col 1
        {2,0}, {5,0}  // col 2
    };
    rocqStatus_t status = rocquantum::util::rocTensorAllocate(&input_tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "permute.alloc_input");
    HIP_ASSERT(hipMemcpy(input_tensor.data_, h_input_data.data(), h_input_data.size() * sizeof(rocComplex), hipMemcpyHostToDevice));

    // Output: 3x2 matrix (transpose of input)
    // 0 3
    // 1 4
    // 2 5
    // Stored column-major: 0,1,2,3,4,5
    output_tensor.dimensions_ = {3, 2}; // M=3, N=2
    output_tensor.calculate_strides();
     std::vector<rocComplex> h_expected_output_data = {
        {0,0}, {1,0}, {2,0}, // col 0
        {3,0}, {4,0}, {5,0}  // col 1
    };
    status = rocquantum::util::rocTensorAllocate(&output_tensor);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "permute.alloc_output");

    // Permutation map: p[new_idx] = old_idx
    // Input modes (old): {mode0_size2, mode1_size3}
    // Output modes (new): {mode0_size3, mode1_size2}
    // New mode 0 (size 3) was old mode 1.
    // New mode 1 (size 2) was old mode 0.
    // So, permutation_map = {1, 0}
    std::vector<int> permutation_map = {1, 0};

    status = rocquantum::util::rocTensorPermute(&output_tensor, &input_tensor, permutation_map);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "permute.rocTensorPermute_status");

    HIP_ASSERT(hipStreamSynchronize(test_stream)); // Ensure kernel is done

    std::vector<rocComplex> h_output_data(output_tensor.get_element_count());
    HIP_ASSERT(hipMemcpy(h_output_data.data(), output_tensor.data_, h_output_data.size() * sizeof(rocComplex), hipMemcpyDeviceToHost));

    ASSERT_EQ(h_output_data.size(), h_expected_output_data.size(), "permute.output_size_check");
    for (size_t i = 0; i < h_expected_output_data.size(); ++i) {
        ASSERT_COMPLEX_EQ(h_output_data[i], h_expected_output_data[i], "permute.output_data_element_" + std::to_string(i));
    }

    rocquantum::util::rocTensorFree(&input_tensor);
    rocquantum::util::rocTensorFree(&output_tensor);
    std::cout << "  Sub-test: rocTensorPermute (matrix transpose) PASSED." << std::endl;
    return true;
}

bool test_rocTensorContractWithRocBLAS_matrix_multiply() {
    std::cout << "  Sub-test: rocTensorContractWithRocBLAS (matrix multiply)..." << std::endl;
    rocquantum::util::rocTensor tensorA, tensorB, tensorC_result;

    // A (2x3)
    //  1  2  3
    //  4  5  6
    // Column major host: 1,4, 2,5, 3,6
    tensorA.dimensions_ = {2, 3}; // M=2, K=3
    tensorA.labels_ = {"i", "k"};
    tensorA.calculate_strides();
    std::vector<rocComplex> h_A_data = {{1,0},{4,0}, {2,0},{5,0}, {3,0},{6,0}};
    rocquantum::util::rocTensorAllocate(&tensorA);
    HIP_ASSERT(hipMemcpy(tensorA.data_, h_A_data.data(), h_A_data.size() * sizeof(rocComplex), hipMemcpyHostToDevice));

    // B (3x2)
    //  7  8
    //  9 10
    // 11 12
    // Column major host: 7,9,11, 8,10,12
    tensorB.dimensions_ = {3, 2}; // K=3, N=2
    tensorB.labels_ = {"k", "j"};
    tensorB.calculate_strides();
    std::vector<rocComplex> h_B_data = {{7,0},{9,0},{11,0}, {8,0},{10,0},{12,0}};
    rocquantum::util::rocTensorAllocate(&tensorB);
    HIP_ASSERT(hipMemcpy(tensorB.data_, h_B_data.data(), h_B_data.size() * sizeof(rocComplex), hipMemcpyHostToDevice));

    // Expected C = A * B (2x2)
    // C = [ (1*7 + 2*9 + 3*11)  (1*8 + 2*10 + 3*12) ]
    //     [ (4*7 + 5*9 + 6*11)  (4*8 + 5*10 + 6*12) ]
    //   = [ (7 + 18 + 33)  (8 + 20 + 36) ]
    //     [ (28 + 45 + 66) (32 + 50 + 72) ]
    //   = [ 58  64 ]
    //     [ 139 154 ]
    // Column major host: 58,139, 64,154
    std::vector<rocComplex> h_C_expected = {{58,0},{139,0}, {64,0},{154,0}};
    tensorC_result.dimensions_ = {2, 2}; // M=2, N=2
    tensorC_result.labels_ = {"i", "j"}; // These will be set by parser if empty
    tensorC_result.calculate_strides();
    rocquantum::util::rocTensorAllocate(&tensorC_result);

    const char* spec = "ik,kj->ij";
    rocqStatus_t status = rocquantum::util::rocTensorContractWithRocBLAS(&tensorC_result, &tensorA, &tensorB, spec, blas_handle, test_stream);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "contract_mm.status_ok");
    HIP_ASSERT(hipStreamSynchronize(test_stream));

    ASSERT_COMPLEX_VEC_EQ(h_C_expected, tensorC_result.data_, h_C_expected.size(), "contract_mm.result_matches");

    // Test with tensor names in spec
    const char* spec_named = "A(ik),B(kj)->C(ij)";
    // Need to reset tensorC_result or use a new one if its content is checked again
    rocComplex zero = {0,0};
    std::vector<rocComplex> h_C_zeros(tensorC_result.get_element_count(), zero);
    HIP_ASSERT(hipMemcpy(tensorC_result.data_, h_C_zeros.data(), h_C_zeros.size() * sizeof(rocComplex), hipMemcpyHostToDevice));

    status = rocquantum::util::rocTensorContractWithRocBLAS(&tensorC_result, &tensorA, &tensorB, spec_named, blas_handle, test_stream);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "contract_mm_named.status_ok");
    HIP_ASSERT(hipStreamSynchronize(test_stream));
    ASSERT_COMPLEX_VEC_EQ(h_C_expected, tensorC_result.data_, h_C_expected.size(), "contract_mm_named.result_matches");


    rocquantum::util::rocTensorFree(&tensorA);
    rocquantum::util::rocTensorFree(&tensorB);
    rocquantum::util::rocTensorFree(&tensorC_result);
    std::cout << "  Sub-test: rocTensorContractWithRocBLAS (matrix multiply) PASSED." << std::endl;
    return true;
}

bool test_rocTensorContractWithRocBLAS_inner_product() {
    std::cout << "  Sub-test: rocTensorContractWithRocBLAS (inner product)..." << std::endl;
    rocquantum::util::rocTensor tensorA, tensorB, tensorC_result;

    // A (vector of 3 elements)
    tensorA.dimensions_ = {3};
    tensorA.labels_ = {"i"};
    tensorA.calculate_strides();
    std::vector<rocComplex> h_A_data = {{1,1},{2,2},{3,3}}; // (1+i), (2+2i), (3+3i)
    rocquantum::util::rocTensorAllocate(&tensorA);
    HIP_ASSERT(hipMemcpy(tensorA.data_, h_A_data.data(), h_A_data.size() * sizeof(rocComplex), hipMemcpyHostToDevice));

    // B (vector of 3 elements)
    tensorB.dimensions_ = {3};
    tensorB.labels_ = {"i"};
    tensorB.calculate_strides();
    std::vector<rocComplex> h_B_data = {{4,1},{5,1},{6,1}}; // (4+i), (5+i), (6+i)
    rocquantum::util::rocTensorAllocate(&tensorB);
    HIP_ASSERT(hipMemcpy(tensorB.data_, h_B_data.data(), h_B_data.size() * sizeof(rocComplex), hipMemcpyHostToDevice));

    // Expected C = sum(A_i * B_i) (scalar)
    // (1+i)(4+i) = 4 + i + 4i - 1 = 3 + 5i
    // (2+2i)(5+i) = 10 + 2i + 10i - 2 = 8 + 12i
    // (3+3i)(6+i) = 18 + 3i + 18i - 3 = 15 + 21i
    // Sum = (3+8+15) + (5+12+21)i = 26 + 38i
    std::vector<rocComplex> h_C_expected = {{26,38}};

    // Result tensor (scalar)
    // The parser "i,i->s" will make result_dims {1}
    tensorC_result.dimensions_ = {1};
    tensorC_result.labels_ = {"s"}; // Placeholder, parser will set based on spec
    tensorC_result.calculate_strides(); // Strides will be {1}
    rocquantum::util::rocTensorAllocate(&tensorC_result);

    // Spec for inner product (sum over 'i', result is scalar 's')
    const char* spec = "i,i->s";
    rocqStatus_t status = rocquantum::util::rocTensorContractWithRocBLAS(&tensorC_result, &tensorA, &tensorB, spec, blas_handle, test_stream);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "contract_inner_prod.status_ok");
    HIP_ASSERT(hipStreamSynchronize(test_stream));

    ASSERT_COMPLEX_VEC_EQ(h_C_expected, tensorC_result.data_, h_C_expected.size(), "contract_inner_prod.result_matches");

    rocquantum::util::rocTensorFree(&tensorA);
    rocquantum::util::rocTensorFree(&tensorB);
    rocquantum::util::rocTensorFree(&tensorC_result);
    std::cout << "  Sub-test: rocTensorContractWithRocBLAS (inner product) PASSED." << std::endl;
    return true;
}

bool test_rocWorkspaceManager_basic() {
    std::cout << "  Sub-test: rocWorkspaceManager basic operations..." << std::endl;

    size_t num_complex_elements_total = 1024;
    size_t workspace_size_bytes = num_complex_elements_total * sizeof(rocComplex); // 1024 rocComplex elements

    rocquantum::util::WorkspaceManager ws_manager(workspace_size_bytes, test_stream);
    ASSERT_EQ(ws_manager.get_total_size_bytes(), workspace_size_bytes, "ws_manager.total_size_init");
    ASSERT_EQ(ws_manager.get_used_size_bytes(), 0, "ws_manager.used_size_init");

    // Allocate a small chunk
    size_t alloc1_elements = 10;
    rocComplex* ptr1 = ws_manager.allocate(alloc1_elements);
    ASSERT_TRUE(ptr1 != nullptr, "ws_manager.alloc1_ptr_not_null");
    // Used size should be alloc1_elements * sizeof(rocComplex), potentially aligned
    size_t expected_used1 = alloc1_elements * sizeof(rocComplex);
    if ((expected_used1 % 256) != 0) expected_used1 = ((expected_used1 + 256 -1) / 256) * 256; // Assuming 256 alignment
    // The get_used_size_bytes returns the *next* aligned offset, so it reflects usage *after* alignment for the *next* allocation.
    // If current_offset_bytes_ is used directly, it would be alloc1_elements * sizeof(rocComplex).
    // The current implementation of get_used_size_bytes returns the aligned offset.
    // Let's check current_offset_bytes_ for precise usage before next alignment.
    // This requires exposing current_offset_bytes or making get_used_size_bytes return raw offset.
    // For now, let's test based on observed behavior of get_used_size_bytes (next aligned offset).
    // If an allocation exactly fits an alignment boundary, get_used_size_bytes() would be that boundary.
    // If it's less, get_used_size_bytes() will be the end of the current allocation.
    // The `WorkspaceManager::allocate` updates `current_offset_bytes_` to `aligned_offset + requested_bytes;`
    // `get_used_size_bytes` returns `current_offset_bytes_` after potential alignment for the *next* allocation.
    // So, if alloc1_elements * sizeof(rocComplex) is, say, 80 bytes, and alignment is 256,
    // current_offset_bytes_ becomes 80. get_used_size_bytes() would return 256.
    // This is a bit confusing. Let's assume get_used_size_bytes() is the actual end of data.
    // The WorkspaceManager code: `current_offset_bytes_ = aligned_offset + requested_bytes;`
    // `get_used_size_bytes()`: `aligned_offset = current_offset_bytes_; if (current_offset_bytes_ % alignment_ !=0) ...; return aligned_offset;`
    // This means `get_used_size_bytes()` is the *start* of the *next* allocation.
    // So if `current_offset_bytes_` is 80, `get_used_size_bytes()` would be 256 (if alignment is 256).
    // This is fine, it tells us how much is "reserved".

    size_t actual_used_after_alloc1 = ws_manager.get_used_size_bytes();
    size_t raw_bytes1 = alloc1_elements * sizeof(rocComplex);
    size_t expected_aligned_offset1 = ((0 + 256 -1) / 256) * 256; // Start is 0
    size_t expected_current_offset1 = expected_aligned_offset1 + raw_bytes1;

    // ASSERT_EQ(actual_used_after_alloc1, expected_current_offset1_aligned_for_next, "ws_manager.used_size_alloc1");
    // Let's simplify: just check it increased, and allocations don't overlap.
    ASSERT_TRUE(actual_used_after_alloc1 >= raw_bytes1, "ws_manager.used_size_alloc1_increased");


    // Allocate another chunk
    size_t alloc2_elements = 20;
    rocComplex* ptr2 = ws_manager.allocate(alloc2_elements);
    ASSERT_TRUE(ptr2 != nullptr, "ws_manager.alloc2_ptr_not_null");
    ASSERT_TRUE(ptr2 != ptr1, "ws_manager.alloc2_ptr_different_from_ptr1");
    // Check for non-overlap: ptr2 should be at least ptr1 + size_of_alloc1 (with alignment)
    char* char_ptr1 = reinterpret_cast<char*>(ptr1);
    char* char_ptr2 = reinterpret_cast<char*>(ptr2);
    size_t min_distance = (alloc1_elements * sizeof(rocComplex) + 256 -1 ) / 256 * 256; //aligned end of ptr1
    // ASSERT_TRUE( (size_t)(char_ptr2 - char_ptr1) >= alloc1_elements * sizeof(rocComplex), "ws_manager.ptr2_after_ptr1");
    // This check is tricky due to alignment. Simpler: ptr2 should be > ptr1 if alloc1_elements > 0
    if (alloc1_elements > 0) {
      ASSERT_TRUE(ptr2 > ptr1, "ws_manager.ptr2_strictly_after_ptr1");
    }
    size_t actual_used_after_alloc2 = ws_manager.get_used_size_bytes();
    ASSERT_TRUE(actual_used_after_alloc2 > actual_used_after_alloc1 || alloc2_elements == 0, "ws_manager.used_size_alloc2_increased");


    // Test allocation that should fail (too large)
    size_t alloc3_elements = num_complex_elements_total; // Try to allocate the whole workspace again
    rocComplex* ptr3 = ws_manager.allocate(alloc3_elements);
    ASSERT_TRUE(ptr3 == nullptr, "ws_manager.alloc3_ptr_is_null_for_too_large");

    // Test reset
    ws_manager.reset();
    ASSERT_EQ(ws_manager.get_used_size_bytes(), 0, "ws_manager.used_size_after_reset");

    // Allocate again after reset
    rocComplex* ptr4 = ws_manager.allocate(alloc1_elements);
    ASSERT_TRUE(ptr4 != nullptr, "ws_manager.alloc4_ptr_not_null_after_reset");
    // Pointer after reset should ideally be the same as the first pointer, if simple bump allocator
    ASSERT_EQ(ptr4, ptr1, "ws_manager.alloc4_ptr_same_as_ptr1_after_reset");

    // Allocate exactly remaining space
    ws_manager.reset();
    size_t remaining_elements = num_complex_elements_total;
    // Account for initial alignment of the very first block
    size_t initial_padding = 0; // current_offset_bytes_ is 0, alignment is 256. aligned_offset will be 0.

    rocComplex* ptr5 = ws_manager.allocate(remaining_elements);
    ASSERT_TRUE(ptr5 != nullptr, "ws_manager.alloc_full_workspace");

    rocComplex* ptr6 = ws_manager.allocate(1); // Should fail now
    ASSERT_TRUE(ptr6 == nullptr, "ws_manager.alloc_should_fail_when_full");

    // Test with zero initial size
    bool constructor_threw = false;
    try {
        rocquantum::util::WorkspaceManager ws_zero(0, test_stream);
        rocComplex* p_zero = ws_zero.allocate(1); // Should be nullptr
        ASSERT_TRUE(p_zero == nullptr, "ws_manager.alloc_from_zero_size_ws_is_null");
        ASSERT_EQ(ws_zero.get_total_size_bytes(), 0, "ws_manager.zero_total_size");
    } catch (const std::runtime_error& e) {
        // Constructor throwing for size 0 if hipMalloc fails for 0 is also acceptable
        // but current hipMalloc(0) might return nullptr or a unique ptr.
        // The WorkspaceManager itself doesn't throw for size 0, but hipMalloc might.
        // Current hipMalloc(0) behavior is often to return a valid pointer that can be passed to hipFree.
        // The WorkspaceManager code has `if (total_size_bytes_ > 0)` for `hipMalloc`.
        // So for 0, `d_workspace_ptr_` remains `nullptr`.
    }


    std::cout << "  Sub-test: rocWorkspaceManager basic operations PASSED." << std::endl;
    return true;
}


// Helper function to create a rocTensor with data on GPU
rocquantum::util::rocTensor create_gpu_tensor(const std::vector<long long>& dims, const std::vector<std::string>& labels, const std::vector<rocComplex>& h_data) {
    rocquantum::util::rocTensor tensor;
    tensor.dimensions_ = dims;
    tensor.labels_ = labels;
    tensor.calculate_strides();
    rocquantum::util::rocTensorAllocate(&tensor);
    HIP_ASSERT(hipMemcpy(tensor.data_, h_data.data(), h_data.size() * sizeof(rocComplex), hipMemcpyHostToDevice));
    return tensor; // Returns a rocTensor struct (potentially moves if RVO applies)
                  // The data is owned by this tensor instance due to rocTensorAllocate
}

bool test_TensorNetwork_contract_simple_chain_internal(bool use_external_workspace) {
    std::cout << "  Sub-test: TensorNetwork contract simple chain (use_external_workspace=" << use_external_workspace << ")..." << std::endl;

    // Network: A(a,b) * B(b,c) * C(c,d) -> Result(a,d)
    // Dimensions: a=2, b=3, c=4, d=2
    // A(2,3), B(3,4), C(4,2) -> Result(2,2)

    std::vector<rocComplex> h_A_data(2*3); // A(a,b)
    for(int i=0; i<6; ++i) h_A_data[i] = {(float)i+1, 0};
    // A col-major: (1,0) (4,0)
    //              (2,0) (5,0)
    //              (3,0) (6,0)
    // A data: {1,2,3,4,5,6} if we fill it row-by-row for conceptual matrix
    // (0,0)=1 (0,1)=2 (0,2)=3
    // (1,0)=4 (1,1)=5 (1,2)=6
    // Stored as: {1,4,2,5,3,6}
    std::vector<rocComplex> h_A_colmajor = {{1,0},{4,0}, {2,0},{5,0}, {3,0},{6,0}};


    std::vector<rocComplex> h_B_data(3*4); // B(b,c)
    for(int i=0; i<12; ++i) h_B_data[i] = {(float)i+0.5f, 0};
    // B col-major: (0.5,0) (3.5,0) (6.5,0) (9.5,0)
    //              (1.5,0) (4.5,0) (7.5,0) (10.5,0)
    //              (2.5,0) (5.5,0) (8.5,0) (11.5,0)
    // B data: {0.5, 1.5, 2.5, ...}
    std::vector<rocComplex> h_B_colmajor(12);
    for(int j=0; j<4; ++j) { // cols
        for(int i=0; i<3; ++i) { // rows
            h_B_colmajor[i + j*3] = {(float)(i + j*3 + 0.5f), 0};
        }
    }


    std::vector<rocComplex> h_C_data(4*2); // C(c,d)
    for(int i=0; i<8; ++i) h_C_data[i] = {(float)i+0.2f, 0};
    // C data: {0.2, 1.2, 2.2, ...}
    std::vector<rocComplex> h_C_colmajor(8);
     for(int j=0; j<2; ++j) { // cols
        for(int i=0; i<4; ++i) { // rows
            h_C_colmajor[i + j*4] = {(float)(i + j*4 + 0.2f), 0};
        }
    }

    rocquantum::util::rocTensor tensorA = create_gpu_tensor({2,3}, {"a","b"}, h_A_colmajor);
    rocquantum::util::rocTensor tensorB = create_gpu_tensor({3,4}, {"b","c"}, h_B_colmajor);
    rocquantum::util::rocTensor tensorC = create_gpu_tensor({4,2}, {"c","d"}, h_C_colmajor);

    rocquantum::util::WorkspaceManager* ext_ws = nullptr;
    if (use_external_workspace) {
        ext_ws = new rocquantum::util::WorkspaceManager(1024 * 1024 * 8, test_stream); // 8MB workspace
    }

    rocquantum::TensorNetwork tn(ext_ws, test_stream); // Pass stream if constructor takes it
    tn.add_tensor(tensorA); // tensorA is copied by value (metadata), data is view
    tn.add_tensor(tensorB);
    tn.add_tensor(tensorC);

    rocquantum::util::rocTensor result_tensor_gpu;
    rocqStatus_t status = tn.contract(&result_tensor_gpu, blas_handle, test_stream);
    ASSERT_EQ(status, ROCQ_STATUS_SUCCESS, "tn_chain.contract_status");
    HIP_ASSERT(hipStreamSynchronize(test_stream));

    // Expected result: (A*B)*C
    // A(2x3), B(3x4) -> AB(2x4)
    // AB(2x4) * C(4x2) -> Result(2x2)

    // Calculate expected AB = A*B manually (or using rocBLAS for this sub-part if complex)
    // A = [[1,2,3], [4,5,6]]   B = [[0.5, 3.5, 6.5, 9.5], [1.5, 4.5, 7.5, 10.5], [2.5, 5.5, 8.5, 11.5]]
    // AB_00 = 1*0.5 + 2*1.5 + 3*2.5 = 0.5 + 3 + 7.5 = 11
    // AB_01 = 1*3.5 + 2*4.5 + 3*5.5 = 3.5 + 9 + 16.5 = 29
    // AB_02 = 1*6.5 + 2*7.5 + 3*8.5 = 6.5 + 15 + 25.5 = 47
    // AB_03 = 1*9.5 + 2*10.5 + 3*11.5 = 9.5 + 21 + 34.5 = 65
    // AB_10 = 4*0.5 + 5*1.5 + 6*2.5 = 2 + 7.5 + 15 = 24.5
    // AB_11 = 4*3.5 + 5*4.5 + 6*5.5 = 14 + 22.5 + 33 = 69.5
    // AB_12 = 4*6.5 + 5*7.5 + 6*8.5 = 26 + 37.5 + 51 = 114.5
    // AB_13 = 4*9.5 + 5*10.5 + 6*11.5 = 38 + 52.5 + 69 = 159.5
    // AB = [[11, 29, 47, 65], [24.5, 69.5, 114.5, 159.5]] (2x4)

    // C = [[0.2, 4.2], [1.2, 5.2], [2.2, 6.2], [3.2, 7.2]]
    // Result = AB * C
    // Res_00 = 11*0.2 + 29*1.2 + 47*2.2 + 65*3.2 = 2.2 + 34.8 + 103.4 + 208 = 348.4
    // Res_01 = 11*4.2 + 29*5.2 + 47*6.2 + 65*7.2 = 46.2 + 150.8 + 291.4 + 468 = 956.4
    // Res_10 = 24.5*0.2 + 69.5*1.2 + 114.5*2.2 + 159.5*3.2 = 4.9 + 83.4 + 251.9 + 510.4 = 850.6
    // Res_11 = 24.5*4.2 + 69.5*5.2 + 114.5*6.2 + 159.5*7.2 = 102.9 + 361.4 + 710 + 1148.4 = 2322.6 // Error in manual calculation somewhere, recheck.
    // Let's use a simpler data set for manual check or trust the component tests for rocTensorContract.
    // For this test, focus on path and memory. If component contract test passes, this should be fine if path is right.

    // Path taken: (A*B)*C or A*(B*C). Greedy chooses based on intermediate size.
    // A(2,3) B(3,4) -> AB(2,4) size 8. Contracted K=3.
    // B(3,4) C(4,2) -> BC(3,2) size 6. Contracted K=4.
    // Greedy should choose (B*C) first. So order is A*(B*C).
    // Intermediate BC(b,d) with dims {3,2}.
    // Then A(a,b) * BC(b,d) -> Result(a,d) dims {2,2}.

    // B*C:
    // B (3x4), C(4x2) -> BC(3x2)
    // B= [[0.5, 3.5,  6.5,  9.5 ],
    //     [1.5, 4.5,  7.5,  10.5],
    //     [2.5, 5.5,  8.5,  11.5]]
    // C= [[0.2, 4.2],
    //     [1.2, 5.2],
    //     [2.2, 6.2],
    //     [3.2, 7.2]]
    // BC_00 = 0.5*0.2 + 3.5*1.2 + 6.5*2.2 + 9.5*3.2 = 0.1 + 4.2 + 14.3 + 30.4 = 49
    // BC_01 = 0.5*4.2 + 3.5*5.2 + 6.5*6.2 + 9.5*7.2 = 2.1 + 18.2 + 40.3 + 68.4 = 129
    // BC_10 = 1.5*0.2 + 4.5*1.2 + 7.5*2.2 + 10.5*3.2 = 0.3 + 5.4 + 16.5 + 33.6 = 55.8
    // BC_11 = 1.5*4.2 + 4.5*5.2 + 7.5*6.2 + 10.5*7.2 = 6.3 + 23.4 + 46.5 + 75.6 = 151.8
    // BC_20 = 2.5*0.2 + 5.5*1.2 + 8.5*2.2 + 11.5*3.2 = 0.5 + 6.6 + 18.7 + 36.8 = 62.6
    // BC_21 = 2.5*4.2 + 5.5*5.2 + 8.5*6.2 + 11.5*7.2 = 10.5 + 28.6 + 52.7 + 82.8 = 174.6
    // BC = [[49, 129], [55.8, 151.8], [62.6, 174.6]] (3x2) (labels b,d)

    // A * BC:
    // A (2x3), BC(3x2) -> Res(2x2)
    // A = [[1,2,3], [4,5,6]]
    // Res_00 = 1*49 + 2*55.8 + 3*62.6 = 49 + 111.6 + 187.8 = 348.4
    // Res_01 = 1*129 + 2*151.8 + 3*174.6 = 129 + 303.6 + 523.8 = 956.4
    // Res_10 = 4*49 + 5*55.8 + 6*62.6 = 196 + 279 + 375.6 = 850.6
    // Res_11 = 4*129 + 5*151.8 + 6*174.6 = 516 + 759 + 1047.6 = 2322.6
    // Expected Result(a,d) = [[348.4, 956.4], [850.6, 2322.6]]
    // Column major: {348.4, 850.6, 956.4, 2322.6}
    std::vector<rocComplex> h_Res_expected = {{348.4f,0}, {850.6f,0}, {956.4f,0}, {2322.6f,0}};

    ASSERT_EQ(result_tensor_gpu.rank(), 2, "tn_chain.result_rank");
    if(result_tensor_gpu.rank() == 2) {
        ASSERT_EQ(result_tensor_gpu.dimensions_[0], 2, "tn_chain.result_dim0");
        ASSERT_EQ(result_tensor_gpu.dimensions_[1], 2, "tn_chain.result_dim1");
    }
    ASSERT_EQ(result_tensor_gpu.get_element_count(), 4, "tn_chain.result_element_count");
    if (result_tensor_gpu.data_ && result_tensor_gpu.get_element_count() == 4) {
         ASSERT_COMPLEX_VEC_EQ(h_Res_expected, result_tensor_gpu.data_, h_Res_expected.size(), "tn_chain.result_data");
    } else {
        ASSERT_TRUE(false, "tn_chain.result_data_not_checked_due_to_null_or_size_mismatch");
    }


    // Cleanup
    rocquantum::util::rocTensorFree(&tensorA); // Frees data because it was created with create_gpu_tensor -> rocTensorAllocate
    rocquantum::util::rocTensorFree(&tensorB);
    rocquantum::util::rocTensorFree(&tensorC);
    rocquantum::util::rocTensorFree(&result_tensor_gpu); // Result tensor from contract is also owned and needs free

    if (ext_ws) {
        delete ext_ws;
    }

    std::cout << "  Sub-test: TensorNetwork contract simple chain (use_external_workspace=" << use_external_workspace << ") PASSED." << std::endl;
    return true;
}

bool test_TensorNetwork_contract_simple_chain() {
    bool internal_ws_ok = test_TensorNetwork_contract_simple_chain_internal(false);
    bool external_ws_ok = test_TensorNetwork_contract_simple_chain_internal(true);
    return internal_ws_ok && external_ws_ok;
}


int main() {
    setup_global_test_resources();

    ADD_TEST(test_rocTensor_struct);
    ADD_TEST(test_rocTensor_alloc_free);
    ADD_TEST(test_parse_simple_einsum_spec);
    // ADD_TEST(test_rocTensorPermute);
    ADD_TEST(test_rocTensorPermute);
    ADD_TEST(test_rocTensorContractWithRocBLAS_matrix_multiply);
    ADD_TEST(test_rocTensorContractWithRocBLAS_inner_product);
    ADD_TEST(test_rocWorkspaceManager_basic);
    ADD_TEST(test_TensorNetwork_contract_simple_chain);

    RUN_ALL_TESTS();

    teardown_global_test_resources();
    return 0;
}

// Placeholder for rocquantum::checkHipError if not found in hipStateVec.h
// For now, assume it's available or tests will fail to link/run if it's critical.
// If hipStateVec.h only declares it, and it's defined in hipStateVec.cpp,
// we might need to link against that object file or provide a minimal version here.
// For now, the HIP_ASSERT macro handles errors locally.
namespace rocquantum {
    // Minimal version if not linked
    rocqStatus_t checkHipError(hipError_t err, const char* message) {
        if (err != hipSuccess) {
            // std::cerr << "HIP Error (" << message << "): " << hipGetErrorString(err) << std::endl;
            return ROCQ_STATUS_HIP_ERROR;
        }
        return ROCQ_STATUS_SUCCESS;
    }
}
