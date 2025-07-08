#include "rocquantum/hipTensorNet.h"
#include "rocquantum/hipStateVec.h" // For rocqStatus_t
#include <vector>
#include <new> // For std::nothrow
#include <algorithm> // For std::sort, std::min, std::max
#include <iostream>  // For debugging (temporary)
#include <map>

// Define the opaque struct that rocTensorNetworkHandle_t points to
struct rocTnStruct {
    rocquantum::TensorNetwork network;
};

extern "C" {

rocqStatus_t rocTensorNetworkCreate(rocTensorNetworkHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = new(std::nothrow) rocTnStruct();
    if (!tn_struct) {
        *handle = nullptr;
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    *handle = tn_struct;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkDestroy(rocTensorNetworkHandle_t handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = static_cast<rocTnStruct*>(handle);
    // If TensorNetwork owns any tensors in active_tensors_during_contraction_
    // (e.g. intermediates), they should be freed here.
    for(auto& tensor : tn_struct->network.active_tensors_during_contraction_) {
        if (tensor.owned_ && tensor.data_) {
            rocquantum::util::rocTensorFree(&tensor);
        }
    }
    tn_struct->network.active_tensors_during_contraction_.clear();
    // initial_tensors_ are assumed to be views not owned by TensorNetwork itself.
    delete tn_struct;
    return ROCQ_STATUS_SUCCESS;
}

rocqStatus_t rocTensorNetworkAddTensor(rocTensorNetworkHandle_t handle, const rocquantum::util::rocTensor* tensor) {
    if (!handle || !tensor) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = static_cast<rocTnStruct*>(handle);
    if (!tensor->data_ && tensor->get_element_count() > 0) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    tn_struct->network.add_tensor(*tensor);
    return ROCQ_STATUS_SUCCESS;
}

/* rocTensorNetworkAddContraction is removed as contractions are found dynamically.
rocqStatus_t rocTensorNetworkAddContraction(rocTensorNetworkHandle_t handle,
                                            int tensor_idx_A, int mode_idx_A,
                                            int tensor_idx_B, int mode_idx_B) {
    return ROCQ_STATUS_NOT_IMPLEMENTED;
}
*/

rocqStatus_t rocTensorNetworkContract(rocTensorNetworkHandle_t handle,
                                      rocquantum::util::rocTensor* result_tensor,
                                      rocblas_handle blas_handle,
                                      hipStream_t stream) {
    if (!handle || !result_tensor || !blas_handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = static_cast<rocTnStruct*>(handle);
    return tn_struct->network.contract(result_tensor, blas_handle, stream);
}

} // extern "C"


rocqStatus_t rocquantum::TensorNetwork::contract(
    rocquantum::util::rocTensor* result_tensor_out,
    rocblas_handle blas_handle,
    hipStream_t stream) {

    if (initial_tensors_.empty()) {
        if (result_tensor_out) { // Should return a scalar 1 if network is empty? Or error?
            result_tensor_out->dimensions_ = {1};
            result_tensor_out->labels_ = {"scalar"};
            result_tensor_out->calculate_strides();
            // Caller should allocate if they expect a scalar. For now, error.
        }
        return ROCQ_STATUS_INVALID_VALUE;
    }

    // Clear previous active tensors and start with copies of initial tensors.
    // Free any owned data from a previous partial contraction attempt.
    for(auto& t : active_tensors_during_contraction_) {
        rocquantum::util::rocTensorFree(&t);
    }
    active_tensors_during_contraction_.clear();
    for(const auto& t_init : initial_tensors_){
        active_tensors_during_contraction_.push_back(t_init); // These are views, owned_ = false
    }

    while (active_tensors_during_contraction_.size() > 1) {
        std::vector<ContractionCandidate> candidates;

        for (size_t i = 0; i < active_tensors_during_contraction_.size(); ++i) {
            for (size_t j = i + 1; j < active_tensors_during_contraction_.size(); ++j) {
                const auto& t1 = active_tensors_during_contraction_[i];
                const auto& t2 = active_tensors_during_contraction_[j];

                std::vector<std::pair<int, int>> shared_modes = find_shared_mode_indices(t1, t2);

                if (!shared_modes.empty()) {
                    ContractionCandidate cand;
                    cand.tensor_idx1 = static_cast<int>(i);
                    cand.tensor_idx2 = static_cast<int>(j);
                    cand.mode_pairs_to_contract = shared_modes;

                    std::vector<long long> res_dims;
                    std::vector<std::string> res_labels;
                    get_resulting_tensor_metadata(t1, t2, shared_modes, res_dims, res_labels);

                    cand.resulting_tensor_size = 1;
                    if (res_dims.empty()){
                        cand.resulting_tensor_size = 1;
                    } else {
                        for (long long dim : res_dims) {
                            if (dim == 0) {
                                cand.resulting_tensor_size = 0;
                                break;
                            }
                            // Check for potential overflow before multiplication
                            if (dim > 0 && cand.resulting_tensor_size > LLONG_MAX / dim) {
                                return ROCQ_STATUS_INVALID_VALUE; // Overflow in size calculation
                            }
                            cand.resulting_tensor_size *= dim;
                        }
                    }
                    candidates.push_back(cand);
                }
            }
        }

        if (candidates.empty()) {
            if (active_tensors_during_contraction_.size() > 1) {
                return ROCQ_STATUS_FAILURE; // Disconnected network
            }
            break;
        }

        std::sort(candidates.begin(), candidates.end());
        const ContractionCandidate& best_candidate = candidates[0];

        // These are views/copies from active_tensors_during_contraction_
        rocquantum::util::rocTensor tensor_A_view = active_tensors_during_contraction_[best_candidate.tensor_idx1];
        rocquantum::util::rocTensor tensor_B_view = active_tensors_during_contraction_[best_candidate.tensor_idx2];

        rocquantum::util::rocTensor intermediate_result_tensor; // This will be the new tensor
        std::vector<long long> new_dims;
        std::vector<std::string> new_labels;
        get_resulting_tensor_metadata(tensor_A_view, tensor_B_view, best_candidate.mode_pairs_to_contract, new_dims, new_labels);

        intermediate_result_tensor.dimensions_ = new_dims;
        intermediate_result_tensor.labels_ = new_labels;
        intermediate_result_tensor.calculate_strides();
        // intermediate_result_tensor.owned_ will be set by rocTensorAllocate

        // TODO FUTURE: Workspace Memory Management
        // The allocation and deallocation of intermediate tensors here using
        // rocTensorAllocate and rocTensorFree can be inefficient due to repeated
        // hipMalloc/hipFree calls. A future optimization would be to implement
        // a workspace memory manager.
        // 1. A WorkspaceManager class would pre-allocate a large device memory pool.
        // 2. rocTensorAllocate would be replaced by workspace_manager.allocate().
        // 3. rocTensorFree for these intermediates would not call hipFree directly;
        //    instead, the workspace would be reset after the full network contraction.
        rocqStatus_t alloc_status = rocquantum::util::rocTensorAllocate(&intermediate_result_tensor);
        if (alloc_status != ROCQ_STATUS_SUCCESS) {
             // Clean up previously allocated intermediates if any error occurs.
             for(auto& t : active_tensors_during_contraction_) {
                if (t.owned_ && t.data_ != tensor_A_view.data_ && t.data_ != tensor_B_view.data_) { // Don't free inputs here
                    rocquantum::util::rocTensorFree(&t);
                }
             }
            return alloc_status;
        }

        std::vector<int> result_A_modes_order;
        std::vector<bool> tensorA_mode_is_contracted(tensor_A_view.rank(), false);
        for(const auto& p : best_candidate.mode_pairs_to_contract) tensorA_mode_is_contracted[p.first] = true;
        for(size_t i = 0; i < tensor_A_view.rank(); ++i) if(!tensorA_mode_is_contracted[i]) result_A_modes_order.push_back(i);

        std::vector<int> result_B_modes_order;
        std::vector<bool> tensorB_mode_is_contracted(tensor_B_view.rank(), false);
        for(const auto& p : best_candidate.mode_pairs_to_contract) tensorB_mode_is_contracted[p.second] = true;
        for(size_t i = 0; i < tensor_B_view.rank(); ++i) if(!tensorB_mode_is_contracted[i]) result_B_modes_order.push_back(i);

        rocqStatus_t contract_status = rocquantum::util::rocTensorContractPair_internal(
            &intermediate_result_tensor,
            &tensor_A_view,
            &tensor_B_view,
            best_candidate.mode_pairs_to_contract,
            result_A_modes_order,
            result_B_modes_order,
            blas_handle,
            stream
        );

        if (contract_status != ROCQ_STATUS_SUCCESS) {
            rocquantum::util::rocTensorFree(&intermediate_result_tensor);
            for(auto& t : active_tensors_during_contraction_) rocquantum::util::rocTensorFree(&t);
            return contract_status;
        }

        // Update active_tensors_during_contraction_ list
        std::vector<rocquantum::util::rocTensor> next_active_tensors;
        int idx1 = best_candidate.tensor_idx1;
        int idx2 = best_candidate.tensor_idx2;

        for (size_t k = 0; k < active_tensors_during_contraction_.size(); ++k) {
            if (k != static_cast<size_t>(idx1) && k != static_cast<size_t>(idx2)) {
                next_active_tensors.push_back(std::move(active_tensors_during_contraction_[k]));
            } else {
                // If the consumed tensor was an intermediate (owned its data), free it.
                rocquantum::util::rocTensorFree(&active_tensors_during_contraction_[k]);
            }
        }
        next_active_tensors.push_back(std::move(intermediate_result_tensor));
        active_tensors_during_contraction_ = std::move(next_active_tensors);
    }

    if (active_tensors_during_contraction_.size() == 1) {
        // Transfer data and ownership to result_tensor_out
        // The caller of rocTensorNetworkContract is responsible for allocating result_tensor_out struct,
        // but its data buffer should be allocated here if it's not already or if sizes mismatch.

        rocquantum::util::rocTensor& final_tensor = active_tensors_during_contraction_[0];

        // If result_tensor_out is not pre-allocated or shape mismatch, reallocate
        bool shape_match = result_tensor_out->dimensions_ == final_tensor.dimensions_;
        if (!result_tensor_out->data_ || !shape_match || result_tensor_out->get_element_count() != final_tensor.get_element_count()) {
            rocquantum::util::rocTensorFree(result_tensor_out); // Free if it owned different memory
            result_tensor_out->dimensions_ = final_tensor.dimensions_;
            result_tensor_out->labels_ = final_tensor.labels_; // Also copy labels
            result_tensor_out->calculate_strides();
            rocqStatus_t alloc_final_status = rocquantum::util::rocTensorAllocate(result_tensor_out);
            if (alloc_final_status != ROCQ_STATUS_SUCCESS) {
                 rocquantum::util::rocTensorFree(&final_tensor); // Free the computed intermediate
                 return alloc_final_status;
            }
        } else { // Already allocated and shape matches, just update metadata if necessary
            result_tensor_out->labels_ = final_tensor.labels_;
            result_tensor_out->calculate_strides(); // Ensure strides are up-to-date
        }

        // Copy data from the final active tensor to the user-provided result_tensor_out
        if (final_tensor.data_ && result_tensor_out->data_ && final_tensor.get_element_count() > 0) {
            hipError_t copy_err = hipMemcpyAsync(result_tensor_out->data_, final_tensor.data_,
                                         final_tensor.get_element_count() * sizeof(rocComplex),
                                         hipMemcpyDeviceToDevice, stream);
            hipError_t sync_err = hipStreamSynchronize(stream);
            rocquantum::util::rocTensorFree(&final_tensor); // Free the last intermediate

            if (copy_err != hipSuccess || sync_err != hipSuccess) {
                rocquantum::util::rocTensorFree(result_tensor_out); // Free result if copy failed
                return ROCQ_STATUS_HIP_ERROR;
            }
        } else if (final_tensor.get_element_count() == 0 && result_tensor_out->get_element_count() == 0) {
            // Both are 0-element tensors, success.
            rocquantum::util::rocTensorFree(&final_tensor);
        } else if (final_tensor.get_element_count() == 1 && result_tensor_out->get_element_count() == 1 && final_tensor.data_ && result_tensor_out->data_) {
            // Scalar case
            hipError_t copy_err = hipMemcpyAsync(result_tensor_out->data_, final_tensor.data_, sizeof(rocComplex), hipMemcpyDeviceToDevice, stream);
            hipError_t sync_err = hipStreamSynchronize(stream);
            rocquantum::util::rocTensorFree(&final_tensor);
             if (copy_err != hipSuccess || sync_err != hipSuccess) {
                rocquantum::util::rocTensorFree(result_tensor_out);
                return ROCQ_STATUS_HIP_ERROR;
            }
        }
         else if (final_tensor.data_ == nullptr && final_tensor.get_element_count() > 0) {
            // This implies rocTensorContractPair_internal did not produce data for some reason, should have failed earlier.
             rocquantum::util::rocTensorFree(&final_tensor);
             return ROCQ_STATUS_FAILURE;
        }


        return ROCQ_STATUS_SUCCESS;
    } else if (active_tensors_during_contraction_.empty() && initial_tensors_.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    } else {
        // Error: contraction finished with multiple tensors or no tensors.
        for(auto& t : active_tensors_during_contraction_) rocquantum::util::rocTensorFree(&t);
        return ROCQ_STATUS_FAILURE;
    }
}
