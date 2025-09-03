#ifndef HIP_TENSOR_NET_H
#define HIP_TENSOR_NET_H

#include "rocquantum/rocTensorUtil.h" // For rocTensor, rocqStatus_t, rocDataType_t, rocComplex, etc.
#include <vector>
#include <string>
#include <utility>

// Opaque handle for the TensorNetwork object from the C API
typedef struct rocTnStruct* rocTensorNetworkHandle_t;

// Forward declarations from the C-API section might be needed if hipTensorNetContractionOptimizerConfig_t is defined there
// Assuming hipTensorNetContractionOptimizerConfig_t is defined elsewhere and available
struct hipTensorNetContractionOptimizerConfig_t;


#ifdef __cplusplus
namespace rocquantum {

// Abstract base class for type erasure in the C API
class TensorNetworkBase {
public:
    virtual ~TensorNetworkBase() = default;

    virtual int add_tensor(const rocquantum::util::rocTensor& tensor) = 0;

    virtual rocqStatus_t contract(const hipTensorNetContractionOptimizerConfig_t* config,
                                  rocquantum::util::rocTensor* result_tensor,
                                  rocblas_handle blas_handle,
                                  hipStream_t stream) = 0;
    
    // If other public methods need to be exposed via C-API, add them here.
};


/**
 * @brief Represents a tensor network for a specific data type.
 *
 * Manages a collection of tensors and their specified contractions.
 */
template <typename T>
class TensorNetwork : public TensorNetworkBase {
public:
    TensorNetwork(rocquantum::util::WorkspaceManager* external_workspace = nullptr, hipStream_t stream = 0);
    ~TensorNetwork() override;

    // Delete copy constructor and assignment operator
    TensorNetwork(const TensorNetwork&) = delete;
    TensorNetwork& operator=(const TensorNetwork&) = delete;

    /**
     * @brief Adds a tensor to the network.
     */
    int add_tensor(const rocquantum::util::rocTensor& tensor) override;

    /**
     * @brief Performs the tensor network contraction.
     */
    rocqStatus_t contract(const hipTensorNetContractionOptimizerConfig_t* config,
                          rocquantum::util::rocTensor* result_tensor,
                          rocblas_handle blas_handle,
                          hipStream_t stream) override;

private:
    // Helper structure for pathfinding
    struct ContractionCandidate {
        int tensor_idx1;
        int tensor_idx2;
        std::vector<std::pair<int, int>> mode_pairs_to_contract;
        long long resulting_tensor_size;
        bool operator<(const ContractionCandidate& other) const {
            return resulting_tensor_size < other.resulting_tensor_size;
        }
    };

    // Helper functions for pathfinding and metadata calculation
    std::vector<std::pair<int, int>> find_shared_mode_indices(
        const rocquantum::util::rocTensor& t1,
        const rocquantum::util::rocTensor& t2) const;

    void get_resulting_tensor_metadata(
        const rocquantum::util::rocTensor& t1,
        const rocquantum::util::rocTensor& t2,
        const std::vector<std::pair<int, int>>& contracted_mode_pairs,
        std::vector<long long>& out_new_dims,
        std::vector<std::string>& out_new_labels) const;

    // Member variables
    std::vector<rocquantum::util::rocTensor> initial_tensors_;
    std::vector<rocquantum::util::rocTensor> active_tensors_during_contraction_;
    
    rocquantum::util::WorkspaceManager* workspace_ = nullptr;
    bool owns_workspace_ = false;
    static const size_t DEFAULT_WORKSPACE_SIZE_BYTES = 256 * 1024 * 1024;
};

} // namespace rocquantum

extern "C" {
#endif // __cplusplus

/**
 * @brief Creates a tensor network handle for a specific data type.
 * @param[out] handle Pointer to the handle to be created.
 * @param[in] dtype The data type for the tensor network operations.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorNetworkCreate(rocTensorNetworkHandle_t* handle, rocDataType_t dtype);

/**
 * @brief Destroys a tensor network handle and releases associated resources.
 * @param[in] handle The handle to be destroyed.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorNetworkDestroy(rocTensorNetworkHandle_t handle);

/**
 * @brief Adds a tensor to the tensor network.
 * @param handle The tensor network handle.
 * @param tensor Pointer to the rocTensor struct (metadata view).
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorNetworkAddTensor(rocTensorNetworkHandle_t handle, const rocquantum::util::rocTensor* tensor);

/**
 * @brief Contracts the tensor network.
 * @param handle The tensor network handle.
 * @param config Optimizer configuration for pathfinding and slicing.
 * @param[out] result_tensor Pointer to a rocTensor struct for the result.
 * @param blas_handle rocBLAS handle for GEMM operations.
 * @param stream HIP stream for operations.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorNetworkContract(rocTensorNetworkHandle_t handle,
                                      const hipTensorNetContractionOptimizerConfig_t* config,
                                      rocquantum::util::rocTensor* result_tensor,
                                      rocblas_handle blas_handle,
                                      hipStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // HIP_TENSOR_NET_H