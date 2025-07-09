#include "rocquantum/rocTensorUtil.h"
#include "rocquantum/hipStateVec.h" // For rocqStatus_t, rocComplex, checkHipError (from common header)
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h> // For rocblas_handle, rocblas_cgemm etc.
#include <vector>
#include <string>
#include <numeric>      // For std::accumulate, std::iota
#include <stdexcept>    // For error reporting
#include <algorithm>    // For std::sort, std::find, std::set_difference etc.
#include <map>          // For mapping mode indices
#include <set>          // For finding unique labels
#include <sstream>      // For parsing (though simple parsing is used here)


// Forward declare the kernel (it's in rocTensorUtil_kernels.hip, but this .cpp file compiles separately)
__global__ void permute_tensor_kernel(
    rocComplex* output_data,
    const rocComplex* input_data,
    const long long* d_input_dims,
    const long long* d_input_strides,
    const long long* d_output_dims,
    const long long* d_output_strides,
    const int* d_permutation_map, // p[new_mode_idx] = old_mode_idx
    int num_modes,
    long long total_elements);


namespace rocquantum {
namespace util {

rocqStatus_t rocTensorPermute(
    rocTensor* output_tensor,
    const rocTensor* input_tensor,
    const std::vector<int>& host_permutation_map
) {
    if (!output_tensor || !input_tensor) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (input_tensor->rank() != host_permutation_map.size() || output_tensor->rank() != host_permutation_map.size()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (input_tensor->rank() == 0) {
        if (input_tensor->get_element_count() == 1 && output_tensor->get_element_count() == 1) {
            if (output_tensor->data_ && input_tensor->data_) {
                 hipError_t err = hipMemcpy(output_tensor->data_, input_tensor->data_, sizeof(rocComplex), hipMemcpyDeviceToDevice);
                 // Assuming checkHipError is available globally or in this namespace from hipStateVec.h
                 // If not, it needs to be defined or included properly. For now, assume it is.
                  return rocquantum::checkHipError(err, "rocTensorPermute hipMemcpy scalar");
            }
            return ROCQ_STATUS_SUCCESS;
        }
        return ROCQ_STATUS_INVALID_VALUE;
    }

    long long total_elements = input_tensor->get_element_count();
    if (total_elements != output_tensor->get_element_count()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (total_elements == 0) {
        return ROCQ_STATUS_SUCCESS;
    }

    if (!input_tensor->data_ || !output_tensor->data_) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if(input_tensor->dimensions_.size() != input_tensor->strides_.size() && input_tensor->rank() > 0){
            // This implies strides were not calculated for a non-scalar tensor.
            // The rocTensor constructor should handle this, or it's an invalid setup.
             return ROCQ_STATUS_INVALID_VALUE;
    }
    if (output_tensor->strides_.empty() && output_tensor->rank() > 0) {
        output_tensor->calculate_strides();
    }


    int num_modes = static_cast<int>(input_tensor->rank());

    long long* d_input_dims = nullptr;
    long long* d_input_strides = nullptr;
    long long* d_output_dims = nullptr;
    long long* d_output_strides = nullptr;
    int* d_permutation_map_gpu = nullptr;

    rocqStatus_t status = ROCQ_STATUS_SUCCESS;
    hipError_t hip_err;

    hip_err = hipMalloc(&d_input_dims, num_modes * sizeof(long long));
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_ALLOCATION_FAILED; goto perm_cleanup; }
    hip_err = hipMemcpy(d_input_dims, input_tensor->dimensions_.data(), num_modes * sizeof(long long), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; goto perm_cleanup; }

    hip_err = hipMalloc(&d_input_strides, num_modes * sizeof(long long));
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_ALLOCATION_FAILED; goto perm_cleanup; }
    hip_err = hipMemcpy(d_input_strides, input_tensor->strides_.data(), num_modes * sizeof(long long), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; goto perm_cleanup; }

    hip_err = hipMalloc(&d_output_dims, num_modes * sizeof(long long));
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_ALLOCATION_FAILED; goto perm_cleanup; }
    hip_err = hipMemcpy(d_output_dims, output_tensor->dimensions_.data(), num_modes * sizeof(long long), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; goto perm_cleanup; }

    hip_err = hipMalloc(&d_output_strides, num_modes * sizeof(long long));
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_ALLOCATION_FAILED; goto perm_cleanup; }
    hip_err = hipMemcpy(d_output_strides, output_tensor->strides_.data(), num_modes * sizeof(long long), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; goto perm_cleanup; }

    hip_err = hipMalloc(&d_permutation_map_gpu, num_modes * sizeof(int));
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_ALLOCATION_FAILED; goto perm_cleanup; }
    hip_err = hipMemcpy(d_permutation_map_gpu, host_permutation_map.data(), num_modes * sizeof(int), hipMemcpyHostToDevice);
    if (hip_err != hipSuccess) { status = ROCQ_STATUS_HIP_ERROR; goto perm_cleanup; }

    unsigned int threads_per_block = 256;
    unsigned int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (total_elements > 0 && num_blocks == 0) num_blocks = 1;
    else if (total_elements == 0) num_blocks = 0;

    if (num_blocks > 0) {
        hipLaunchKernelGGL(permute_tensor_kernel,
                           dim3(num_blocks), dim3(threads_per_block), 0, 0,
                           output_tensor->data_, input_tensor->data_,
                           d_input_dims, d_input_strides,
                           d_output_dims, d_output_strides,
                           d_permutation_map_gpu, num_modes, total_elements);

        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) { status = rocquantum::checkHipError(hip_err, "permute_tensor_kernel launch"); goto perm_cleanup;}
        hip_err = hipStreamSynchronize(0);
        if (hip_err != hipSuccess) { status = rocquantum::checkHipError(hip_err, "rocTensorPermute hipStreamSynchronize");}
    }

perm_cleanup:
    if (d_input_dims) hipFree(d_input_dims);
    if (d_input_strides) hipFree(d_input_strides);
    if (d_output_dims) hipFree(d_output_dims);
    if (d_output_strides) hipFree(d_output_strides);
    if (d_permutation_map_gpu) hipFree(d_permutation_map_gpu);
    return status;
}

// Internal helper function for core contraction logic
rocqStatus_t rocTensorContractPair_internal(
    rocTensor* result_tensor,
    const rocTensor* tensorA,
    const rocTensor* tensorB,
    const std::vector<std::pair<int, int>>& contracted_mode_pairs_A_B,
    const std::vector<int>& result_A_modes_initial_order,
    const std::vector<int>& result_B_modes_initial_order,
    rocblas_handle blas_handle,
    hipStream_t stream) {

    if (!result_tensor || !tensorA || !tensorB || !blas_handle || !result_tensor->data_) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!tensorA->data_ || !tensorB->data_) return ROCQ_STATUS_INVALID_VALUE;

    rocqStatus_t status_check = ROCQ_STATUS_SUCCESS; // Using rocqStatus_t for internal checks too
    rocblas_status blas_status_err = rocblas_set_stream(blas_handle, stream);
    if (blas_status_err != rocblas_status_success) return ROCQ_STATUS_FAILURE;


    long long M = 1, N = 1, K = 1;
    std::vector<int> permA_map_new_idx_is_old_idx;
    std::vector<int> permB_map_new_idx_is_old_idx;

    std::vector<long long> dimsA_permuted_vec;
    std::vector<long long> dimsB_permuted_vec;

    std::vector<bool> is_mode_A_contracted(tensorA->rank(), false);
    std::vector<bool> is_mode_B_contracted(tensorB->rank(), false);
    std::vector<int> contracted_modes_A_orig_indices;
    std::vector<int> contracted_modes_B_orig_indices;


    for(const auto& p : contracted_mode_pairs_A_B) {
        is_mode_A_contracted[p.first] = true;
        contracted_modes_A_orig_indices.push_back(p.first);
        is_mode_B_contracted[p.second] = true;
        contracted_modes_B_orig_indices.push_back(p.second);
        K *= tensorA->dimensions_[p.first];
    }

    // Order for permuted A: [uncontracted_A_modes (M part), contracted_A_modes (K part)]
    // result_A_modes_initial_order contains original indices of uncontracted A modes
    for(int mode_idx : result_A_modes_initial_order) {
        permA_map_new_idx_is_old_idx.push_back(mode_idx);
        dimsA_permuted_vec.push_back(tensorA->dimensions_[mode_idx]);
        M *= tensorA->dimensions_[mode_idx];
    }
    for(int mode_idx : contracted_modes_A_orig_indices) {
        permA_map_new_idx_is_old_idx.push_back(mode_idx);
        dimsA_permuted_vec.push_back(tensorA->dimensions_[mode_idx]);
    }

    // Order for permuted B: [contracted_B_modes (K part), uncontracted_B_modes (N part)]
    for(int mode_idx : contracted_modes_B_orig_indices) {
        permB_map_new_idx_is_old_idx.push_back(mode_idx);
        dimsB_permuted_vec.push_back(tensorB->dimensions_[mode_idx]);
    }
    for(int mode_idx : result_B_modes_initial_order) {
        permB_map_new_idx_is_old_idx.push_back(mode_idx);
        dimsB_permuted_vec.push_back(tensorB->dimensions_[mode_idx]);
        N *= tensorB->dimensions_[mode_idx];
    }

    if (M == 0 || N == 0 || K == 0) {
        if (M * N != result_tensor->get_element_count() && result_tensor->get_element_count() !=0 ) return ROCQ_STATUS_INVALID_VALUE;
        if (result_tensor->data_ && result_tensor->get_element_count() > 0) {
            hipMemsetAsync(result_tensor->data_, 0, result_tensor->get_element_count() * sizeof(rocComplex), stream);
            hipStreamSynchronize(stream);
            return ROCQ_STATUS_SUCCESS;
        } else if (result_tensor->get_element_count() == 0) {
             return ROCQ_STATUS_SUCCESS; // Contracting to a 0-element tensor
        }
    }

    rocTensor permutedA_tensor, permutedB_tensor;
    permutedA_tensor.dimensions_ = dimsA_permuted_vec;
    permutedA_tensor.calculate_strides(); // Important
    permutedB_tensor.dimensions_ = dimsB_permuted_vec;
    permutedB_tensor.calculate_strides(); // Important

    status_check = rocTensorAllocate(&permutedA_tensor);
    if (status_check != ROCQ_STATUS_SUCCESS) return status_check;
    status_check = rocTensorAllocate(&permutedB_tensor);
    if (status_check != ROCQ_STATUS_SUCCESS) { rocTensorFree(&permutedA_tensor); return status_check; }

    status_check = rocTensorPermute(&permutedA_tensor, tensorA, permA_map_new_idx_is_old_idx);
    if (status_check != ROCQ_STATUS_SUCCESS) { rocTensorFree(&permutedA_tensor); rocTensorFree(&permutedB_tensor); return status_check; }

    status_check = rocTensorPermute(&permutedB_tensor, tensorB, permB_map_new_idx_is_old_idx);
    if (status_check != ROCQ_STATUS_SUCCESS) { rocTensorFree(&permutedA_tensor); rocTensorFree(&permutedB_tensor); return status_check; }

    hipError_t hip_sync_err = hipStreamSynchronize(stream);
    if(hip_sync_err != hipSuccess) {
        rocTensorFree(&permutedA_tensor); rocTensorFree(&permutedB_tensor);
        return rocquantum::checkHipError(hip_sync_err, "ContractPair sync after permute");
    }

    int gemm_M = static_cast<int>(M);
    int gemm_N = static_cast<int>(N);
    int gemm_K = static_cast<int>(K);

    const rocComplex alpha = {1.0f, 0.0f};
    const rocComplex beta  = {0.0f, 0.0f};

    blas_status_err = rocblas_cgemm(
        blas_handle, ROCBLAS_OPERATION_NONE, ROCBLAS_OPERATION_NONE,
        gemm_M, gemm_N, gemm_K,
        &alpha,
        permutedA_tensor.data_, gemm_M,
        permutedB_tensor.data_, gemm_K,
        &beta,
        result_tensor->data_, gemm_M
    );

    hip_sync_err = hipStreamSynchronize(stream);

    rocTensorFree(&permutedA_tensor);
    rocTensorFree(&permutedB_tensor);

    if (blas_status_err != rocblas_status_success) return ROCQ_STATUS_FAILURE;
    if(hip_sync_err != hipSuccess) return rocquantum::checkHipError(hip_sync_err, "ContractPair sync after GEMM");

    return ROCQ_STATUS_SUCCESS;
}

// --- Helper for parsing simple Einsum-like strings ---
// Example: "ab,bc->ac"
// Populates contracted_pairs, and orders for uncontracted modes from A and B
// Returns false on parsing error.
bool parse_simple_einsum_spec(
    const std::string& spec,
    const rocTensor* tensorA, const rocTensor* tensorB, // Used to map labels to mode indices
    std::vector<std::pair<int, int>>& contracted_pairs_A_B,
    std::vector<int>& result_A_modes_order, // Original mode indices of A that are uncontracted
    std::vector<int>& result_B_modes_order, // Original mode indices of B that are uncontracted
    std::vector<long long>& result_dims,    // Dimensions for the result tensor
    std::vector<std::string>& result_labels // Labels for the result tensor
) {
    contracted_pairs_A_B.clear();
    result_A_modes_order.clear();
    result_B_modes_order.clear();
    result_dims.clear();
    result_labels.clear();

    size_t arrow_pos = spec.find("->");
    if (arrow_pos == std::string::npos) return false; // Invalid format

    std::string inputs_str = spec.substr(0, arrow_pos);
    std::string output_str = spec.substr(arrow_pos + 2);

    size_t comma_pos = inputs_str.find(',');
    if (comma_pos == std::string::npos) return false; // Must have two input tensors

    std::string tensorA_spec_str = inputs_str.substr(0, comma_pos);
    std::string tensorB_spec_str = inputs_str.substr(comma_pos + 1);

    // Extract labels for A, B, and Result
    // Helper to trim whitespace from a string
    auto trim_string = [](std::string& s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    };

    // Helper to split a string by a delimiter
    auto split_string = [&](const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            trim_string(token);
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        return tokens;
    };

    bool implicit_output = false;
    size_t arrow_pos = spec.find("->");
    std::string inputs_str;
    std::string output_spec_str;

    if (arrow_pos == std::string::npos) {
        implicit_output = true;
        inputs_str = spec;
    } else {
        inputs_str = spec.substr(0, arrow_pos);
        output_spec_str = spec.substr(arrow_pos + 2);
        trim_string(output_spec_str);
    }
    trim_string(inputs_str);

    size_t comma_pos = inputs_str.find(',');
    if (comma_pos == std::string::npos) return false; // Must have two input tensors

    std::string tensorA_spec_str = inputs_str.substr(0, comma_pos);
    std::string tensorB_spec_str = inputs_str.substr(comma_pos + 1);
    trim_string(tensorA_spec_str);
    trim_string(tensorB_spec_str);

    // Extract labels for A, B, and Result
    // Supports "A(label1,label2)" or just "label1,label2" or "label1 label2" (space as secondary delimiter)
    auto extract_labels_from_spec = [&](const std::string& tensor_spec, size_t expected_rank) {
        std::string actual_labels_str = tensor_spec;
        size_t p_open = tensor_spec.find('(');
        size_t p_close = tensor_spec.find(')');
        if (p_open != std::string::npos && p_close != std::string::npos && p_close > p_open) {
            actual_labels_str = tensor_spec.substr(p_open + 1, p_close - p_open - 1);
        }

        std::vector<std::string> labels_vec = split_string(actual_labels_str, ',');
        if (labels_vec.empty() && !actual_labels_str.empty()) { // Try space delimiter if comma yields nothing but string is not empty
             labels_vec = split_string(actual_labels_str, ' ');
        }
        if (labels_vec.size() != expected_rank) return std::vector<std::string>(); // Mismatch
        return labels_vec;
    };

    std::vector<std::string> labelsA_vec = extract_labels_from_spec(tensorA_spec_str, tensorA->rank());
    std::vector<std::string> labelsB_vec = extract_labels_from_spec(tensorB_spec_str, tensorB->rank());
    std::vector<std::string> labelsRes_vec;

    if (labelsA_vec.empty() || labelsB_vec.empty()) return false; // Label count mismatch or parsing error

    if (!implicit_output) {
        labelsRes_vec = extract_labels_from_spec(output_spec_str, 0); // Pass 0 as expected rank, will be checked later
        if (labelsRes_vec.empty() && !output_spec_str.empty() && output_spec_str != "scalar") return false; // If output specified but not parsed
    }

    std::map<std::string, int> map_A_label_to_idx;
    std::map<std::string, int> map_B_label_to_idx;
    for(size_t i=0; i<labelsA_vec.size(); ++i) map_A_label_to_idx[labelsA_vec[i]] = i;
    for(size_t i=0; i<labelsB_vec.size(); ++i) map_B_label_to_idx[labelsB_vec[i]] = i;

    std::set<std::string> labels_A_set(labelsA_vec.begin(), labelsA_vec.end());
    std::set<std::string> labels_B_set(labelsB_vec.begin(), labelsB_vec.end());
    std::set<std::string> labels_Res_set(labelsRes_vec.begin(), labelsRes_vec.end());

    // Find contracted indices
    for (const std::string& label_str : labelsA_vec) {
        if (labels_B_set.count(label_str)) { // Potential contraction
            bool is_in_result = implicit_output ? false : labels_Res_set.count(label_str);
            if (!is_in_result) { // Contracted if in A and B, but not in result (or if result is implicit)
                int modeA_idx = map_A_label_to_idx[label_str];
                int modeB_idx = map_B_label_to_idx[label_str];
                if (tensorA->dimensions_[modeA_idx] != tensorB->dimensions_[modeB_idx]) return false; // Dim mismatch
                contracted_pairs_A_B.push_back({modeA_idx, modeB_idx});
            }
        }
    }

    if (implicit_output) {
        // Deduce result labels: uncontracted A, then uncontracted B
        std::set<std::string> contracted_label_set;
        for(const auto& p : contracted_pairs_A_B) {
            contracted_label_set.insert(labelsA_vec[p.first]);
        }
        for(const std::string& label_A : labelsA_vec) {
            if (!contracted_label_set.count(label_A)) {
                labelsRes_vec.push_back(label_A);
            }
        }
        for(const std::string& label_B : labelsB_vec) {
            if (!contracted_label_set.count(label_B) && !labels_A_set.count(label_B)) { // Ensure not already added from A if A had same uncontracted label
                labelsRes_vec.push_back(label_B);
            }
        }
        labels_Res_set.insert(labelsRes_vec.begin(), labelsRes_vec.end()); // Update Res_set for logic below
    }


    // Determine result mode order and dimensions based on labelsRes_vec
    for (const std::string& res_label_str : labelsRes_vec) {
        if (labels_A_set.count(res_label_str) && !labels_B_set.count(res_label_str)) { // From A only
            bool is_contracted = false; // Check if this label from A was actually part of a contracted pair
            for(const auto& p : contracted_pairs_A_B) if(labelsA_vec[p.first] == res_label_str) is_contracted = true;
            if(is_contracted) return false; // Result label cannot be a contracted one from A

            int modeA_idx = map_A_label_to_idx[res_label_str];
            result_A_modes_order.push_back(modeA_idx);
            result_dims.push_back(tensorA->dimensions_[modeA_idx]);
            result_labels.push_back(res_label_str);
        } else if (labels_B_set.count(res_label_str) && !labels_A_set.count(res_label_str)) { // From B only
            bool is_contracted = false;
            for(const auto& p : contracted_pairs_A_B) if(labelsB_vec[p.second] == res_label_str) is_contracted = true;
            if(is_contracted) return false; // Result label cannot be a contracted one from B

            int modeB_idx = map_B_label_to_idx[res_label_str];
            result_B_modes_order.push_back(modeB_idx);
            result_dims.push_back(tensorB->dimensions_[modeB_idx]);
            result_labels.push_back(res_label_str);
        } else if (labels_A_set.count(res_label_str) && labels_B_set.count(res_label_str)) {
             // Label appears in both A and B AND in Result - this means it was NOT contracted.
             // This case is complex for simple einsum. Usually, if a label appears in A, B, and Result,
             // it implies a diagonal/batch operation, not a standard GEMM-based contraction.
             // For this parser, disallow this unless it's the *only* way it's uncontracted.
             // Simpler: if it's common to A and B, it MUST be contracted (not in result for implicit, or explicitly not in result for explicit)
             bool is_actually_contracted = false;
             for(const auto& p : contracted_pairs_A_B) {
                 if (labelsA_vec[p.first] == res_label_str && labelsB_vec[p.second] == res_label_str) {
                     is_actually_contracted = true;
                     break;
                 }
             }
             if (is_actually_contracted) return false; // A result label cannot be one that was contracted.

            // If it's in A, B, and Res, and NOT contracted, it's an "outer product" like mode.
            // This parser might not support this robustly. Standard einsum usually means such labels are summed out if not in output.
            // For now, let's assume if it's in A, B, and Res, it's an error for this simplified parser.
            return false;
        } else if (res_label_str == "scalar" && labelsRes_vec.size()==1) {
            // Handled by the empty result_dims check later
        }
         else {
            return false; // Result label not found in uncontracted parts of inputs
        }
    }

    // Final check for output spec consistency
    if (!implicit_output && labelsRes_vec.size() != result_dims.size()) {
        if (!(labelsRes_vec.empty() && output_spec_str == "scalar" && result_dims.size() == 1 && result_dims[0] == 1) &&
            !(labelsRes_vec.size()==1 && labelsRes_vec[0]=="scalar" && result_dims.size() == 1 && result_dims[0] == 1) &&
            !(labelsRes_vec.empty() && result_dims.empty() && contracted_pairs_A_B.size() == tensorA->rank() && contracted_pairs_A_B.size() == tensorB->rank()) // full scalar contraction
        )
        return false; // Mismatch between specified output labels and derived output structure
    }


    if (result_dims.empty() && (!contracted_pairs_A_B.empty() || (tensorA->rank()==0 && tensorB->rank()==0) ) ) { // Scalar result
        result_dims.push_back(1);
        if (result_labels.empty()) result_labels.push_back("scalar");
    }

    return true;
}


rocqStatus_t rocTensorContractWithRocBLAS(
    rocTensor* result_tensor,
    const rocTensor* tensorA,
    const rocTensor* tensorB,
    const char* contraction_indices_spec_char,
    rocblas_handle blas_handle,
    hipStream_t stream) {

    if (!result_tensor || !tensorA || !tensorB || !blas_handle || !contraction_indices_spec_char) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    // result_tensor->data_ must be pre-allocated by the caller (e.g. TensorNetwork::contract)
    // after determining the result shape from parsing the spec.
    if (!tensorA->data_ || !tensorB->data_ ) { // result_tensor->data_ checked in internal
        return ROCQ_STATUS_INVALID_VALUE;
    }

    std::string spec(contraction_indices_spec_char);
    std::vector<std::pair<int, int>> contracted_pairs;
    std::vector<int> res_A_modes, res_B_modes;
    std::vector<long long> expected_res_dims;
    std::vector<std::string> expected_res_labels;

    bool parsed_ok = parse_simple_einsum_spec(spec, tensorA, tensorB,
                                            contracted_pairs, res_A_modes, res_B_modes,
                                            expected_res_dims, expected_res_labels);

    if (!parsed_ok) {
        return ROCQ_STATUS_INVALID_VALUE; // Failed to parse spec
    }

    // Caller must have set result_tensor dimensions and allocated memory.
    // Verify consistency:
    if (result_tensor->dimensions_ != expected_res_dims) {
        // Optional: could re-allocate result_tensor here if allowed by API contract,
        // but safer to require caller to pre-allocate correctly.
        return ROCQ_STATUS_INVALID_VALUE; // Result tensor dimensions mismatch
    }
    if (!result_tensor->data_ && result_tensor->get_element_count() > 0) {
        return ROCQ_STATUS_INVALID_VALUE; // Result tensor not allocated
    }
    // Set labels if not already set by caller, or verify
    if(result_tensor->labels_.empty() && !expected_res_labels.empty()) {
        result_tensor->labels_ = expected_res_labels;
    }
    result_tensor->calculate_strides();


    return rocTensorContractPair_internal(result_tensor, tensorA, tensorB,
                                          contracted_pairs, res_A_modes, res_B_modes,
                                          blas_handle, stream);
}


} // namespace util
} // namespace rocquantum
