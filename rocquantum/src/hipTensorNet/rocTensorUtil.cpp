#include "rocquantum/rocTensorUtil.h"
#include "rocquantum/hipStateVec.h" // For rocqStatus_t, rocComplex, checkHipError (from common header)
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h> // For rocblas_handle, rocblas_cgemm etc.
#include <vector>
#include <numeric>      // For std::accumulate, std::iota
#include <stdexcept>    // For error reporting
#include <algorithm>    // For std::sort, std::find, std::set_difference etc.
#include <map>          // For mapping mode indices

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

// External checkHipError from hipStateVec.cpp, or define one locally if this is a standalone util.
// For now, assume it's accessible as rocquantum::util::checkHipError or similar if needed.
// Or, more simply, just use the one from hipStateVec.h if it's made generally available.
// Let's assume checkHipError is available.

rocqStatus_t rocTensorPermute(
    rocTensor* output_tensor,
    const rocTensor* input_tensor,
    const std::vector<int>& host_permutation_map // p[new_mode_idx] = old_mode_idx
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
                 // Ensure current device is set if using default stream, or pass stream from handle
                 // hipStream_t stream = 0; // default stream
                 // For safety, if this util is used with specific streams, they should be passed.
                 hipError_t err = hipMemcpy(output_tensor->data_, input_tensor->data_, sizeof(rocComplex), hipMemcpyDeviceToDevice);
                 return checkHipError(err, "rocTensorPermute hipMemcpy scalar");
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
     // Ensure strides are calculated
    if (input_tensor->strides_.empty() && input_tensor->rank() > 0) {
        // This is problematic if input_tensor is const.
        // The design should ensure tensors always have strides when used.
        // For now, let's assume they are pre-calculated or rocTensor constructor did it.
        // const_cast<rocTensor*>(input_tensor)->calculate_strides(); // Risky if not intended
        if(input_tensor->dimensions_.size() != input_tensor->strides_.size()){
             return ROCQ_STATUS_INVALID_VALUE; // Strides must be present
        }
    }
    if (output_tensor->strides_.empty() && output_tensor->rank() > 0) {
        output_tensor->calculate_strides(); // Output tensor can be modified
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
                           dim3(num_blocks), dim3(threads_per_block), 0, 0, // Default stream
                           output_tensor->data_, input_tensor->data_,
                           d_input_dims, d_input_strides,
                           d_output_dims, d_output_strides,
                           d_permutation_map_gpu, num_modes, total_elements);

        hip_err = hipGetLastError();
        if (hip_err != hipSuccess) { status = checkHipError(hip_err, "permute_tensor_kernel launch"); goto perm_cleanup;}
        hip_err = hipStreamSynchronize(0);
        if (hip_err != hipSuccess) { status = checkHipError(hip_err, "rocTensorPermute hipStreamSynchronize");}
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
    const std::vector<std::pair<int, int>>& contracted_mode_pairs_A_B, // (modeA_idx, modeB_idx)
    const std::vector<int>& result_A_modes_initial_order, // Uncontracted mode indices from A, in their original order
    const std::vector<int>& result_B_modes_initial_order, // Uncontracted mode indices from B, in their original order
    rocblas_handle blas_handle,
    hipStream_t stream) {

    if (!result_tensor || !tensorA || !tensorB || !blas_handle || !result_tensor->data_) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!tensorA->data_ || !tensorB->data_) return ROCQ_STATUS_INVALID_VALUE;

    rocqStatus_t status = rocblas_set_stream(blas_handle, stream);
    if (status != rocblas_status_success) return ROCQ_STATUS_FAILURE;

    // 1. Prepare metadata for permuted tensors and GEMM
    std::vector<int> permA_map, permB_map;
    std::vector<long long> dimsA_permuted, dimsB_permuted;

    long long M = 1, N = 1, K = 1;
    std::vector<int> mode_map_A_to_K; // Contracted modes of A
    std::vector<int> mode_map_A_to_M; // Uncontracted modes of A (row index of matrix A)

    std::vector<int> mode_map_B_to_K; // Contracted modes of B
    std::vector<int> mode_map_B_to_N; // Uncontracted modes of B (col index of matrix B)

    std::vector<bool> is_mode_A_contracted(tensorA->rank(), false);
    std::vector<bool> is_mode_B_contracted(tensorB->rank(), false);

    for(const auto& p : contracted_mode_pairs_A_B) {
        mode_map_A_to_K.push_back(p.first);
        is_mode_A_contracted[p.first] = true;
        mode_map_B_to_K.push_back(p.second);
        is_mode_B_contracted[p.second] = true;
        K *= tensorA->dimensions_[p.first]; // Assuming dimensions match, checked by caller (TensorNetwork)
    }

    for(int mode_idx : result_A_modes_initial_order) { // These are original indices from tensorA
        if (!is_mode_A_contracted[mode_idx]) {
            mode_map_A_to_M.push_back(mode_idx);
            M *= tensorA->dimensions_[mode_idx];
        }
    }
    for(int mode_idx : result_B_modes_initial_order) { // These are original indices from tensorB
         if (!is_mode_B_contracted[mode_idx]) {
            mode_map_B_to_N.push_back(mode_idx);
            N *= tensorB->dimensions_[mode_idx];
        }
    }

    // If M, N, or K is 0 (e.g. from a zero dimension), contraction result is effectively zero or invalid.
    // For simplicity, assume valid non-zero dimensions for M,N,K here.
    // A robust implementation would handle zero dimensions.
    if (M==0 || N==0 || K==0) {
        if (result_tensor->get_element_count() > 0) { // If result is expected to be non-empty
             // This implies an issue, or result should be zero-filled.
             // For now, let's assume M,N,K > 0 for a valid contraction leading to non-empty result.
             // If result_tensor has 0 elements, it might be fine.
            if (M*N != result_tensor->get_element_count()) return ROCQ_STATUS_INVALID_VALUE; // Shape mismatch
            if (result_tensor->data_) { // Fill with zero if expected result is non-empty but M,N,K implies zero work
                hipMemsetAsync(result_tensor->data_, 0, result_tensor->get_element_count() * sizeof(rocComplex), stream);
                return ROCQ_STATUS_SUCCESS; // Or indicate a trivial contraction
            }
        } else if (M*N == 0 && result_tensor->get_element_count() == 0) {
            return ROCQ_STATUS_SUCCESS; // Contracting to a 0-element tensor
        }
    }


    // Create permutation maps: new_order[new_idx] = old_idx
    // For A: uncontracted modes (M part), then contracted modes (K part)
    permA_map.reserve(tensorA->rank());
    for(int old_idx : mode_map_A_to_M) permA_map.push_back(old_idx);
    for(int old_idx : mode_map_A_to_K) permA_map.push_back(old_idx);

    // For B: contracted modes (K part), then uncontracted modes (N part)
    permB_map.reserve(tensorB->rank());
    for(int old_idx : mode_map_B_to_K) permB_map.push_back(old_idx);
    for(int old_idx : mode_map_B_to_N) permB_map.push_back(old_idx);

    // Inverse permutation maps for rocTensorPermute (p[new_idx] = old_idx)
    std::vector<int> inv_permA_map(num_modesA);
    std::vector<int> inv_permB_map(num_modesB);

    for(size_t i=0; i < permA_map.size(); ++i) {
        dimsA_permuted.push_back(tensorA->dimensions_[permA_map[i]]);
        inv_permA_map[i] = permA_map[i]; // This is not inverse, this is p[new]=old if permA_map is p[new]=old
                                        // rocTensorPermute expects p[new_idx] = old_idx.
                                        // The permA_map IS p[new_idx]=old_idx.
    }
     for(size_t i=0; i < permB_map.size(); ++i) {
        dimsB_permuted.push_back(tensorB->dimensions_[permB_map[i]]);
        // inv_permB_map[i] = permB_map[i]; // Same here
    }


    // 2. Allocate and Permute Tensors
    rocTensor permutedA_tensor, permutedB_tensor;
    permutedA_tensor.dimensions_ = dimsA_permuted;
    permutedB_tensor.dimensions_ = dimsB_permuted;

    status = rocTensorAllocate(&permutedA_tensor);
    if (status != ROCQ_STATUS_SUCCESS) return status;
    status = rocTensorAllocate(&permutedB_tensor);
    if (status != ROCQ_STATUS_SUCCESS) { rocTensorFree(&permutedA_tensor); return status; }

    // rocTensorPermute expects map p[new_idx] = old_idx.
    // Our permA_map and permB_map are already in this format.
    status = rocTensorPermute(&permutedA_tensor, tensorA, permA_map);
    if (status != ROCQ_STATUS_SUCCESS) { rocTensorFree(&permutedA_tensor); rocTensorFree(&permutedB_tensor); return status; }

    status = rocTensorPermute(&permutedB_tensor, tensorB, permB_map);
    if (status != ROCQ_STATUS_SUCCESS) { rocTensorFree(&permutedA_tensor); rocTensorFree(&permutedB_tensor); return status; }

    // Synchronize after permutations if rocTensorPermute doesn't sync internally
    // Assuming rocTensorPermute syncs its stream (default stream 0 for now).
    // If it used the passed 'stream', then sync that stream here.
    hipError_t hip_sync_err = hipStreamSynchronize(stream); // Sync the main computation stream
    if(hip_sync_err != hipSuccess) {
        rocTensorFree(&permutedA_tensor); rocTensorFree(&permutedB_tensor);
        return checkHipError(hip_sync_err, "ContractPair sync after permute");
    }


    // 3. Call rocblas_cgemm
    // A_mat is M x K, B_mat is K x N, C_mat is M x N
    // For column major: lda=M, ldb=K, ldc=M
    rocblas_operation opA = ROCBLAS_OPERATION_NONE;
    rocblas_operation opB = ROCBLAS_OPERATION_NONE;

    // M, N, K must be int for rocBLAS
    int gemm_M = static_cast<int>(M);
    int gemm_N = static_cast<int>(N);
    int gemm_K = static_cast<int>(K);

    const rocComplex alpha = {1.0f, 0.0f};
    const rocComplex beta  = {0.0f, 0.0f};

    rocblas_status blas_status = rocblas_cgemm(
        blas_handle, opA, opB,
        gemm_M, gemm_N, gemm_K,
        &alpha,
        permutedA_tensor.data_, gemm_M, // lda = M (number of rows of matrix A if opA is N)
        permutedB_tensor.data_, gemm_K, // ldb = K (number of rows of matrix B if opB is N)
        &beta,
        result_tensor->data_, gemm_M    // ldc = M (number of rows of matrix C)
    );

    hip_sync_err = hipStreamSynchronize(stream); // Sync after GEMM

    rocTensorFree(&permutedA_tensor);
    rocTensorFree(&permutedB_tensor);

    if (blas_status != rocblas_status_success) {
        // Consider mapping rocblas_status to rocqStatus_t more granularly
        return ROCQ_STATUS_FAILURE;
    }
    if(hip_sync_err != hipSuccess) {
        return checkHipError(hip_sync_err, "ContractPair sync after GEMM");
    }

    // Reshape of result_tensor (dimensions, strides, labels) should have been done by the caller (TensorNetwork::contract)
    // before allocating its memory. rocTensorContractPair_internal just fills the data.
    return ROCQ_STATUS_SUCCESS;
}


rocqStatus_t rocTensorContractWithRocBLAS(
    rocTensor* result_tensor,
    const rocTensor* tensorA,
    const rocTensor* tensorB,
    const char* contraction_indices_spec, // e.g., "abc,cd->abd" or list of pairs like "{{1,0},{2,1}}"
    rocblas_handle blas_handle,
    hipStream_t stream) {

    if (!result_tensor || !tensorA || !tensorB || !blas_handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    if (!result_tensor->data_ || !tensorA->data_ || !tensorB->data_) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    // TODO: Implement parsing of contraction_indices_spec.
    // This is a significant task. For now, this C-API is a stub for that part.
    // It should parse the spec into:
    // - std::vector<std::pair<int, int>> contracted_mode_pairs_A_B
    // - std::vector<int> result_A_modes_order (uncontracted modes of A in final result order)
    // - std::vector<int> result_B_modes_order (uncontracted modes of B in final result order)
    // And then call rocTensorContractPair_internal.

    // Example of how it might be called if spec was already parsed:
    // std::vector<std::pair<int, int>> example_contract_pairs = {{0,0}}; // Contract mode 0 of A with mode 0 of B
    // std::vector<int> example_A_order; for(int i=1; i<tensorA->rank(); ++i) example_A_order.push_back(i);
    // std::vector<int> example_B_order; for(int i=1; i<tensorB->rank(); ++i) example_B_order.push_back(i);
    // return rocTensorContractPair_internal(result_tensor, tensorA, tensorB,
    //                                       example_contract_pairs, example_A_order, example_B_order,
    //                                       blas_handle, stream);

    return ROCQ_STATUS_NOT_IMPLEMENTED; // Parsing of spec and calling internal helper not done.
}


} // namespace util
} // namespace rocquantum
