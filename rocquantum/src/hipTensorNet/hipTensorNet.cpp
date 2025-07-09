#include "rocquantum/hipTensorNet.h"
#include "rocquantum/hipStateVec.h" // For rocqStatus_t
#include "rocquantum/rocWorkspaceManager.h" // For WorkspaceManager
#include <vector>
#include <new> // For std::nothrow
#include <algorithm> // For std::sort, std::min, std::max
#include <iostream>  // For debugging (temporary)
#include <map>

// Define the opaque struct that rocTensorNetworkHandle_t points to
struct rocTnStruct {
    rocquantum::TensorNetwork network;
    // The network itself will own its workspace manager if created internally
};

extern "C" {

rocqStatus_t rocTensorNetworkCreate(rocTensorNetworkHandle_t* handle) {
    if (!handle) {
        return ROCQ_STATUS_INVALID_VALUE;
    }
    rocTnStruct* tn_struct = nullptr;
    try {
        // TensorNetwork constructor now creates a default WorkspaceManager
        tn_struct = new rocTnStruct();
    } catch (const std::bad_alloc& e) {
        *handle = nullptr;
        return ROCQ_STATUS_ALLOCATION_FAILED;
    } catch (const std::runtime_error& e_ws) { // Catch workspace allocation failure
        // std::cerr << "Error creating TensorNetwork: " << e_ws.what() << std::endl;
        if (tn_struct) delete tn_struct; // Should not happen if constructor throws
        *handle = nullptr;
        return ROCQ_STATUS_ALLOCATION_FAILED; // Or a more specific workspace error
    }

    if (!tn_struct) { // Should be caught by bad_alloc, but as a safeguard
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
    // TensorNetwork destructor will handle freeing its owned workspace
    // and any owned tensors in active_tensors_during_contraction_ (though they should be workspace managed)
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

/* rocTensorNetworkAddContraction is removed
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


// Implementation of TensorNetwork methods
rocqStatus_t rocquantum::TensorNetwork::contract(
    rocquantum::util::rocTensor* result_tensor_out,
    rocblas_handle blas_handle,
    hipStream_t stream) {

    if (initial_tensors_.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    }

    for(auto& t : active_tensors_during_contraction_) { // Clear any owned from previous failed attempts
        if (t.owned_) rocquantum::util::rocTensorFree(&t);
    }
    active_tensors_during_contraction_.clear();
    for(const auto& t_init : initial_tensors_){
        active_tensors_during_contraction_.push_back(t_init);
    }

    if (workspace_) {
        workspace_->reset();
    } else {
        // Fallback: no workspace available, proceed with individual hipMalloc/hipFree
        // This path should ideally emit a warning or be less preferred.
        // std::cout << "Warning: Contracting tensor network without a workspace manager. Performance may be impacted." << std::endl;
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
                    if (res_dims.empty()){ cand.resulting_tensor_size = 1; }
                    else {
                        for (long long dim : res_dims) {
                            if (dim == 0) { cand.resulting_tensor_size = 0; break; }
                            if (dim > 0 && cand.resulting_tensor_size > LLONG_MAX / dim) return ROCQ_STATUS_INVALID_VALUE;
                            cand.resulting_tensor_size *= dim;
                        }
                    }
                    candidates.push_back(cand);
                }
            }
        }

        if (candidates.empty()) {
            if (active_tensors_during_contraction_.size() > 1) return ROCQ_STATUS_FAILURE;
            break;
        }

        std::sort(candidates.begin(), candidates.end());
        const ContractionCandidate& best_candidate = candidates[0];

        rocquantum::util::rocTensor tensor_A_view = active_tensors_during_contraction_[best_candidate.tensor_idx1];
        rocquantum::util::rocTensor tensor_B_view = active_tensors_during_contraction_[best_candidate.tensor_idx2];

        rocquantum::util::rocTensor intermediate_result_tensor;
        std::vector<long long> new_dims;
        std::vector<std::string> new_labels;
        get_resulting_tensor_metadata(tensor_A_view, tensor_B_view, best_candidate.mode_pairs_to_contract, new_dims, new_labels);

        intermediate_result_tensor.dimensions_ = new_dims;
        intermediate_result_tensor.labels_ = new_labels;
        intermediate_result_tensor.calculate_strides();

        if (workspace_) {
            intermediate_result_tensor.data_ = workspace_->allocate(intermediate_result_tensor.get_element_count());
            if (!intermediate_result_tensor.data_ && intermediate_result_tensor.get_element_count() > 0) {
                for(auto& t : active_tensors_during_contraction_) if(t.owned_) rocquantum::util::rocTensorFree(&t);
                return ROCQ_STATUS_ALLOCATION_FAILED; // Workspace full
            }
            intermediate_result_tensor.owned_ = false; // Workspace manages this memory block
        } else {
            rocqStatus_t alloc_status = rocquantum::util::rocTensorAllocate(&intermediate_result_tensor);
            if (alloc_status != ROCQ_STATUS_SUCCESS) {
                 for(auto& t : active_tensors_during_contraction_) {
                    if (t.owned_ && t.data_ != tensor_A_view.data_ && t.data_ != tensor_B_view.data_) {
                        rocquantum::util::rocTensorFree(&t);
                    }
                 }
                return alloc_status;
            }
            // rocTensorAllocate sets owned_ = true
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
            // If allocated from workspace, memory is reclaimed on workspace_.reset().
            // If allocated individually (owned_=true), free it.
            if (intermediate_result_tensor.owned_ && intermediate_result_tensor.data_) {
                 rocquantum::util::rocTensorFree(&intermediate_result_tensor);
            }
            // Free any other owned intermediates from active_tensors_
            for(auto& t : active_tensors_during_contraction_) if(t.owned_) rocquantum::util::rocTensorFree(&t);
            return contract_status;
        }

        std::vector<rocquantum::util::rocTensor> next_active_tensors;
        int idx1 = best_candidate.tensor_idx1;
        int idx2 = best_candidate.tensor_idx2;

        for (size_t k = 0; k < active_tensors_during_contraction_.size(); ++k) {
            if (k != static_cast<size_t>(idx1) && k != static_cast<size_t>(idx2)) {
                next_active_tensors.push_back(std::move(active_tensors_during_contraction_[k]));
            } else {
                // If the consumed tensor was individually owned (not from workspace), free it.
                if (active_tensors_during_contraction_[k].owned_) {
                    rocquantum::util::rocTensorFree(&active_tensors_during_contraction_[k]);
                }
                // If it was from workspace, its memory will be reclaimed by workspace_->reset() later.
            }
        }
        next_active_tensors.push_back(std::move(intermediate_result_tensor));
        active_tensors_during_contraction_ = std::move(next_active_tensors);
    }

    if (active_tensors_during_contraction_.size() == 1) {
        rocquantum::util::rocTensor& final_computed_tensor = active_tensors_during_contraction_[0];

        bool shape_match = result_tensor_out->dimensions_ == final_computed_tensor.dimensions_;
        if (!result_tensor_out->data_ || !shape_match || result_tensor_out->get_element_count() != final_computed_tensor.get_element_count()) {
            rocquantum::util::rocTensorFree(result_tensor_out);
            result_tensor_out->dimensions_ = final_computed_tensor.dimensions_;
            result_tensor_out->labels_ = final_computed_tensor.labels_;
            result_tensor_out->calculate_strides();
            // The final result tensor_out should NOT be allocated from the workspace,
            // as its lifetime is managed by the caller.
            rocqStatus_t alloc_final_status = rocquantum::util::rocTensorAllocate(result_tensor_out);
            if (alloc_final_status != ROCQ_STATUS_SUCCESS) {
                 if (final_computed_tensor.owned_) rocquantum::util::rocTensorFree(&final_computed_tensor);
                 // Workspace memory for final_computed_tensor (if it used workspace) is reclaimed by reset.
                 return alloc_final_status;
            }
        } else {
            result_tensor_out->labels_ = final_computed_tensor.labels_;
            result_tensor_out->calculate_strides();
        }

        if (final_computed_tensor.data_ && result_tensor_out->data_ && final_computed_tensor.get_element_count() > 0) {
            hipError_t copy_err = hipMemcpyAsync(result_tensor_out->data_, final_computed_tensor.data_,
                                         final_computed_tensor.get_element_count() * sizeof(rocComplex),
                                         hipMemcpyDeviceToDevice, stream);
            if (copy_err != hipSuccess) {
                if (final_computed_tensor.owned_) rocquantum::util::rocTensorFree(&final_computed_tensor);
                return ROCQ_STATUS_HIP_ERROR;
            }
        } else if (final_computed_tensor.get_element_count() == 0 && result_tensor_out->get_element_count() == 0) {
            // Success
        } else if (final_computed_tensor.get_element_count() == 1 && result_tensor_out->get_element_count() == 1 && final_computed_tensor.data_ && result_tensor_out->data_) { // Scalar
            hipError_t copy_err = hipMemcpyAsync(result_tensor_out->data_, final_computed_tensor.data_, sizeof(rocComplex), hipMemcpyDeviceToDevice, stream);
            if (copy_err != hipSuccess) {
                if (final_computed_tensor.owned_) rocquantum::util::rocTensorFree(&final_computed_tensor);
                return ROCQ_STATUS_HIP_ERROR;
            }
        } else if (final_computed_tensor.data_ == nullptr && final_computed_tensor.get_element_count() > 0) {
             if (final_computed_tensor.owned_) rocquantum::util::rocTensorFree(&final_computed_tensor);
             return ROCQ_STATUS_FAILURE;
        }

        // Free the last intermediate if it was heap allocated (owned_ == true)
        // If it was workspace allocated (owned_ == false), its memory is managed by the workspace reset.
        if (final_computed_tensor.owned_) {
            rocquantum::util::rocTensorFree(&final_computed_tensor);
        }
        active_tensors_during_contraction_.clear();

        return ROCQ_STATUS_SUCCESS;
    } else if (active_tensors_during_contraction_.empty() && initial_tensors_.empty()) {
        return ROCQ_STATUS_INVALID_VALUE;
    } else {
        for(auto& t : active_tensors_during_contraction_) {
            if(t.owned_) rocquantum::util::rocTensorFree(&t);
        }
        active_tensors_during_contraction_.clear();
        return ROCQ_STATUS_FAILURE;
    }
}
