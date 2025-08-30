#ifndef HIP_TENSOR_NET_H
#define HIP_TENSOR_NET_H

#include "rocquantum/rocTensorUtil.h" // For rocTensor, rocqStatus_t
#include <vector>
#include <string>
#include <utility> // For std::pair

// Forward declaration of the internal implementation if PIMPL is used (not used initially for simplicity)
// struct rocTnInternalHandle;

// Opaque handle for the TensorNetwork object from the C API
typedef struct rocTnStruct* rocTensorNetworkHandle_t;


#ifdef __cplusplus
namespace rocquantum {

/**
 * @brief Represents a tensor network.
 *
 * Manages a collection of tensors and their specified contractions.
 * This is a preliminary structure; connectivity representation and contraction
 * algorithms will be expanded later.
 */
class TensorNetwork {
public:
    TensorNetwork() = default;
    ~TensorNetwork() = default; // Basic destructor

    // Delete copy constructor and assignment operator to prevent shallow copies with owned resources (if any later)
    // For now, it's simple, but good practice for classes managing resources.
    TensorNetwork(const TensorNetwork&) = delete;
    TensorNetwork& operator=(const TensorNetwork&) = delete;

    // Move constructor and assignment could be added if needed
    // TensorNetwork(TensorNetwork&&) noexcept;
    // TensorNetwork& operator=(TensorNetwork&&) noexcept;

    /**
     * @brief Adds a tensor to the network.
     * The TensorNetwork class will store a copy of the rocTensor metadata.
     * It assumes the rocTensor's data_ pointer is valid and managed externally
     * for the lifetime of its use in the network.
     * @param tensor The tensor to add.
     * @return Index of the added tensor in the network.
     */
    int add_tensor(const rocquantum::util::rocTensor& tensor) {
        initial_tensors_.push_back(tensor); // Makes a copy of the rocTensor struct
        return static_cast<int>(initial_tensors_.size() - 1);
    }

    /**
     * @brief Adds a contraction between two modes of two tensors.
     * Simplified: assumes modes are not already contracted.
     * @param tensor_idx_A Index of the first tensor.
     * @param mode_idx_A Mode index of the first tensor to contract.
     * @param tensor_idx_B Index of the second tensor.
     * @param mode_idx_B Mode index of the second tensor to contract.
     * @return rocqStatus_t indicating success or failure (e.g., invalid indices).
     */
    rocqStatus_t add_contraction(int tensor_idx_A, int mode_idx_A, int tensor_idx_B, int mode_idx_B) {
        if (tensor_idx_A < 0 || tensor_idx_A >= static_cast<int>(initial_tensors_.size()) ||
            tensor_idx_B < 0 || tensor_idx_B >= static_cast<int>(initial_tensors_.size()) ||
            tensor_idx_A == tensor_idx_B) { // Disallow self-contraction for this simple API
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (mode_idx_A < 0 || mode_idx_A >= static_cast<int>(initial_tensors_[tensor_idx_A].rank()) ||
            mode_idx_B < 0 || mode_idx_B >= static_cast<int>(initial_tensors_[tensor_idx_B].rank())) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        // Basic check: ensure dimensions of contracted modes match
        if (initial_tensors_[tensor_idx_A].dimensions_[mode_idx_A] != initial_tensors_[tensor_idx_B].dimensions_[mode_idx_B]) {
            return ROCQ_STATUS_INVALID_VALUE; // Dimension mismatch for contraction
        }

        // This function is deprecated in favor of using labels
        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }

    /**
     * @brief Performs the tensor network contraction.
     *
     * @param result_tensor Pointer to a rocTensor where the result will be stored.
     *                      The caller is responsible for ensuring it's allocated with
     *                      the correct dimensions for the final tensor.
     * @param blas_handle rocBLAS handle for GEMM operations.
     * @param stream HIP stream for operations.
     * @return rocqStatus_t
     */
    rocqStatus_t contract(rocquantum::util::rocTensor* result_tensor, rocblas_handle blas_handle, hipStream_t stream);


public:
    // Stores the initial tensors added to the network.
    std::vector<rocquantum::util::rocTensor> initial_tensors_;
    // Note: The `contractions_` member defined by pairs of (tensor_idx, mode_idx)
    // is too simplistic for dynamic pathfinding based on shared labels.
    // Pathfinding will dynamically identify contractions.
    // We might retain `contractions_` if it's used to declare *intended*
    // connections by label, which pathfinding can then verify or use as hints.
    // For now, let's assume pathfinding works primarily on shared labels.
    // The declared_contractions_ member is removed as pathfinding will be dynamic.


    // Helper structure for pathfinding
    struct ContractionCandidate {
        int tensor_idx1;
        int tensor_idx2;
        std::vector<std::pair<int, int>> mode_pairs_to_contract; // (mode_idx_in_tensor1, mode_idx_in_tensor2)
        long long resulting_tensor_size;
        // Could add other cost metrics here (e.g., FLOPs)

        bool operator<(const ContractionCandidate& other) const {
            return resulting_tensor_size < other.resulting_tensor_size;
        }
    };


private:
    /**
     * @brief Finds all shared labels between two tensors and the corresponding mode indices.
     * @param t1 First tensor.
     * @param t2 Second tensor.
     * @return A vector of pairs, where each pair contains {mode_idx_in_t1, mode_idx_in_t2} for a shared label.
     */
    std::vector<std::pair<int, int>> find_shared_mode_indices(
        const rocquantum::util::rocTensor& t1,
        const rocquantum::util::rocTensor& t2) const;

    /**
     * @brief Calculates the metadata (dimensions, labels) of a tensor resulting from contracting two tensors.
     * @param t1 First tensor.
     * @param t2 Second tensor.
     * @param contracted_mode_pairs Pairs of (mode_idx_t1, mode_idx_t2) that are contracted.
     * @param out_new_dims Output vector for dimensions of the resulting tensor.
     * @param out_new_labels Output vector for labels of the resulting tensor.
     */
    void get_resulting_tensor_metadata(
        const rocquantum::util::rocTensor& t1,
        const rocquantum::util::rocTensor& t2,
        const std::vector<std::pair<int, int>>& contracted_mode_pairs,
        std::vector<long long>& out_new_dims,
        std::vector<std::string>& out_new_labels) const;


public:
    std::vector<rocquantum::util::rocTensor> active_tensors_during_contraction_;

private:
    rocquantum::util::WorkspaceManager* workspace_ = nullptr;
    bool owns_workspace_ = false;

    // Default workspace size if none provided (e.g., 256 MB)
    static const size_t DEFAULT_WORKSPACE_SIZE_BYTES = 256 * 1024 * 1024;

public:
    // Constructor that takes an external workspace manager (optional)
    TensorNetwork(rocquantum::util::WorkspaceManager* external_workspace = nullptr, hipStream_t stream = 0) {
        if (external_workspace) {
            workspace_ = external_workspace;
            owns_workspace_ = false;
        } else {
            try {
                workspace_ = new rocquantum::util::WorkspaceManager(DEFAULT_WORKSPACE_SIZE_BYTES, stream);
                owns_workspace_ = true;
            } catch (const std::runtime_error& e) {
                // Failed to create default workspace, set to null and proceed without one,
                // or rethrow / handle error appropriately. For now, proceed without.
                workspace_ = nullptr;
                owns_workspace_ = false;
                // Consider logging this failure.
            }
        }
    }

    ~TensorNetwork() {
        if (owns_workspace_ && workspace_) {
            delete workspace_;
            workspace_ = nullptr;
        }
        // Ensure any tensors in active_tensors_during_contraction_ that might have been
        // allocated with rocTensorAllocate (not workspace) and are owned, are freed.
        // However, the design is moving towards intermediates being workspace managed.
        // Initial tensors are views. If contract() fails mid-way, it should clean up its intermediates.
        for(auto& t : active_tensors_during_contraction_) {
            if (t.owned_ && t.data_) { // If any tensor was somehow still marked owned and not from workspace
                 rocquantum::util::rocTensorFree(&t); // (rocTensorFree checks ownership again)
            }
        }
    }


};

} // namespace rocquantum

extern "C" {
#endif // __cplusplus

/**
 * @brief Creates a tensor network handle.
 * @param[out] handle Pointer to the handle to be created.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorNetworkCreate(rocTensorNetworkHandle_t* handle);

/**
 * @brief Destroys a tensor network handle and releases associated resources.
 * @param[in] handle The handle to be destroyed.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorNetworkDestroy(rocTensorNetworkHandle_t handle);

/**
 * @brief Adds a tensor to the tensor network.
 * The network stores a copy of the tensor metadata. Data management for the
 * tensor's device pointer is external.
 *
 * @param handle The tensor network handle.
 * @param tensor Pointer to the rocTensor struct (metadata view).
 * @return rocqStatus_t Status of the operation. Returns tensor_idx on success in a wrapper if needed.
 */
rocqStatus_t rocTensorNetworkAddTensor(rocTensorNetworkHandle_t handle, const rocquantum::util::rocTensor* tensor);

/**
 * @brief Defines a contraction between two modes of two tensors in the network.
 *
 * @param handle The tensor network handle.
 * @param tensor_idx_A Index of the first tensor.
 * @param mode_idx_A Mode index of the first tensor.
 * @param tensor_idx_B Index of the second tensor.
 * @param mode_idx_B Mode index of the second tensor.
 * @return rocqStatus_t Status of the operation.
 */
// rocStatus_t rocTensorNetworkAddContraction(rocTensorNetworkHandle_t handle,
//                                             int tensor_idx_A, int mode_idx_A,
//                                             int tensor_idx_B, int mode_idx_B); // Removed

/**
 * @brief Contracts the tensor network.
 *
 * @param handle The tensor network handle.
 * @param[out] result_tensor Pointer to an rocTensor struct where the result tensor's
 *                           metadata and data pointer will be set. The caller is responsible
 *                           for allocating this struct. The function will allocate device memory
 *                           for the result tensor data.
 * @param blas_handle rocBLAS handle for GEMM operations.
 * @param stream HIP stream for operations.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorNetworkContract(rocTensorNetworkHandle_t handle,
                                      rocquantum::util::rocTensor* result_tensor,
                                      rocblas_handle blas_handle, /* Pass from rocsvHandle? */
                                      hipStream_t stream         /* Pass from rocsvHandle? */
                                      );


#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // HIP_TENSOR_NET_H