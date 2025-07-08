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
        tensors_.push_back(tensor); // Makes a copy of the rocTensor struct
        return static_cast<int>(tensors_.size() - 1);
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
        if (tensor_idx_A < 0 || tensor_idx_A >= static_cast<int>(tensors_.size()) ||
            tensor_idx_B < 0 || tensor_idx_B >= static_cast<int>(tensors_.size()) ||
            tensor_idx_A == tensor_idx_B) { // Disallow self-contraction for this simple API
            return ROCQ_STATUS_INVALID_VALUE;
        }
        if (mode_idx_A < 0 || mode_idx_A >= static_cast<int>(tensors_[tensor_idx_A].rank()) ||
            mode_idx_B < 0 || mode_idx_B >= static_cast<int>(tensors_[tensor_idx_B].rank())) {
            return ROCQ_STATUS_INVALID_VALUE;
        }
        // Basic check: ensure dimensions of contracted modes match
        if (tensors_[tensor_idx_A].dimensions_[mode_idx_A] != tensors_[tensor_idx_B].dimensions_[mode_idx_B]) {
            return ROCQ_STATUS_INVALID_VALUE; // Dimension mismatch for contraction
        }

        contractions_.push_back({{tensor_idx_A, mode_idx_A}, {tensor_idx_B, mode_idx_B}});
        return ROCQ_STATUS_SUCCESS;
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
     * @note This is a STUB. Actual contraction logic is not yet implemented.
     */
    rocqStatus_t contract(rocquantum::util::rocTensor* result_tensor, rocblas_handle blas_handle, hipStream_t stream) {
        // TODO: Implement pathfinding (e.g., greedy).
        // TODO: Implement pairwise contraction using rocTensorContractWithRocBLAS.
        // TODO: Manage intermediate tensors and their memory.
        if (!result_tensor || !blas_handle) return ROCQ_STATUS_INVALID_VALUE;

        // For now, just print the network structure as a placeholder
        // std::cout << "TensorNetwork::contract() called. Network has " << tensors_.size() << " tensors." << std::endl;
        // for(size_t i=0; i < tensors_.size(); ++i) {
        //     std::cout << "  Tensor " << i << ": Rank " << tensors_[i].rank() << ", Elements " << tensors_[i].get_element_count() << std::endl;
        // }
        // std::cout << "Contractions to perform: " << contractions_.size() << std::endl;
        // for(const auto& p : contractions_) {
        //    std::cout << "  (" << p.first.first << "," << p.first.second << ") with ("
        //              << p.second.first << "," << p.second.second << ")" << std::endl;
        // }

        return ROCQ_STATUS_NOT_IMPLEMENTED;
    }


//private: // Make these public for C-API access initially, or provide getters
public:
    std::vector<rocquantum::util::rocTensor> tensors_;
    // Contraction specified as pairs of (tensor_index, mode_index)
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> contractions_;
    // TODO: Add more sophisticated connectivity representation (e.g., hypergraph or list of shared indices)
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
rocqStatus_t rocTensorNetworkAddContraction(rocTensorNetworkHandle_t handle,
                                            int tensor_idx_A, int mode_idx_A,
                                            int tensor_idx_B, int mode_idx_B);

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
 * @return rocqStatus_t Status of the operation. ROCQ_STATUS_NOT_IMPLEMENTED for now.
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
