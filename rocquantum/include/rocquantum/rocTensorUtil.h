#ifndef ROC_TENSOR_UTIL_H
#define ROC_TENSOR_UTIL_H

#include <vector>
#include <string>
#include <numeric>      // For std::accumulate
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <hip/hip_runtime.h> // For rocComplex definition if not already included via hipStateVec.h
                           // Assuming rocComplex is hipFloatComplex or hipDoubleComplex
#include "rocquantum/hipStateVec.h" // For rocqStatus_t and rocComplex (if not defined above)


namespace rocquantum {
namespace util {

// Forward declaration if rocTensor might be used in other utils not yet defined
// struct rocTensor; // Not strictly needed if all usage is within this header or after definition

/**
 * @brief Represents a tensor on the ROCm device.
 *
 * This structure provides a view of tensor data, including its dimensions,
 * data pointer, and optionally, labels for its modes and calculated strides.
 * Memory management of the `data_` pointer is generally considered external
 * to this struct, unless explicitly handled by creation/destruction utilities
 * that are part of a larger tensor management system.
 */
struct rocTensor {
    rocComplex* data_ = nullptr;             // Pointer to the tensor data on the GPU device
    std::vector<long long> dimensions_;      // Shape/dimensions of the tensor (e.g., {2, 2, 2} for a 3-qubit tensor)
    std::vector<std::string> labels_;        // Optional: Names/labels for each mode/index of the tensor
    std::vector<long long> strides_;         // Strides for accessing elements in each mode (column-major convention assumed for rocBLAS)
    bool owned_ = false;                     // True if this rocTensor struct instance owns the `data_` memory

    // Default constructor
    rocTensor() = default;

    // Constructor to initialize a tensor view (does not own memory by default)
    rocTensor(rocComplex* data,
              const std::vector<long long>& dims,
              const std::vector<std::string>& lbls = {},
              bool calculate_strides_on_construct = true,
              bool mem_owned = false)
        : data_(data), dimensions_(dims), labels_(lbls), owned_(mem_owned) {
        if (calculate_strides_on_construct) {
            if (!dimensions_.empty()) {
                strides_.resize(dimensions_.size());
                strides_[0] = 1;
                for (size_t i = 1; i < dimensions_.size(); ++i) {
                    strides_[i] = strides_[i-1] * dimensions_[i-1];
                }
            }
        }
    }

    // Destructor: Only frees memory if owned_ is true.
    ~rocTensor() {
        if (owned_ && data_) {
            // hipFree might not be safe in a header-only destructor if not careful
            // For now, this assumes if owned, it was allocated with hipMalloc
            // Consider moving memory management to dedicated functions if issues arise
            // hipFree(data_); // This can cause issues if header is included multiple times or if data_ was not from hipMalloc
            // For a simple struct, it's often better to leave memory management external
            // or use a dedicated manager class.
            // Let's comment this out for now to avoid potential double-free or non-hipMalloc frees.
            // The plan is for rocTensorUtil to provide alloc/free functions.
            data_ = nullptr;
        }
    }

    // Copy constructor (handle ownership carefully)
    rocTensor(const rocTensor& other)
        : data_(other.data_), dimensions_(other.dimensions_),
          labels_(other.labels_), strides_(other.strides_), owned_(false) {
        // By default, copy constructor creates a view and does NOT take ownership.
        // If deep copy with new memory allocation is needed, a separate clone() method is better.
    }

    // Move constructor
    rocTensor(rocTensor&& other) noexcept
        : data_(other.data_), dimensions_(std::move(other.dimensions_)),
          labels_(std::move(other.labels_)), strides_(std::move(other.strides_)),
          owned_(other.owned_) {
        other.data_ = nullptr;
        other.owned_ = false; // Ownership is transferred
    }

    // Copy assignment (handle ownership carefully)
    rocTensor& operator=(const rocTensor& other) {
        if (this != &other) {
            // If current instance owns memory, it should be freed before reassigning.
            // This is complex for a simple struct. For now, assume assignment creates a view.
            if (owned_ && data_) {
                // hipFree(data_); // Potential issue as noted in destructor
            }
            data_ = other.data_;
            dimensions_ = other.dimensions_;
            labels_ = other.labels_;
            strides_ = other.strides_;
            owned_ = false; // Assignment creates a view, does not transfer ownership by default
        }
        return *this;
    }

    // Move assignment
    rocTensor& operator=(rocTensor&& other) noexcept {
        if (this != &other) {
            if (owned_ && data_) {
                // hipFree(data_); // Potential issue
            }
            data_ = other.data_;
            dimensions_ = std::move(other.dimensions_);
            labels_ = std::move(other.labels_);
            strides_ = std::move(other.strides_);
            owned_ = other.owned_;

            other.data_ = nullptr;
            other.owned_ = false;
        }
        return *this;
    }


    /**
     * @brief Calculates the total number of elements in the tensor.
     * @return Total number of elements. Returns 0 if dimensions are empty.
     */
    long long get_element_count() const {
        if (dimensions_.empty()) {
            return 0;
        }
        return std::accumulate(dimensions_.begin(), dimensions_.end(), 1LL, std::multiplies<long long>());
    }

    /**
     * @brief Calculates strides for the tensor based on its dimensions.
     * Assumes column-major like strides (stride of first index is 1).
     * This is a common convention for Fortran-style arrays and some tensor libraries
     * when interfacing with BLAS (GEMM).
     */
    void calculate_strides() {
        if (dimensions_.empty()) {
            strides_.clear();
            return;
        }
        strides_.resize(dimensions_.size());
        strides_[0] = 1;
        for (size_t i = 1; i < dimensions_.size(); ++i) {
            strides_[i] = strides_[i-1] * dimensions_[i-1];
        }
    }

    /**
     * @brief Returns the rank (number of modes/dimensions) of the tensor.
     */
    size_t rank() const {
        return dimensions_.size();
    }
};


/**
 * @brief Allocates memory for a rocTensor on the device.
 *
 * @param tensor Pointer to the rocTensor struct. Its dimensions must be set.
 *               The `data_` field will be populated and `owned_` will be set to true.
 * @return rocqStatus_t Status of the operation.
 */
inline rocqStatus_t rocTensorAllocate(rocTensor* tensor) {
    if (!tensor) return ROCQ_STATUS_INVALID_VALUE;
    if (tensor->data_ && tensor->owned_) { // If it already owns memory, free it first
        hipFree(tensor->data_);
        tensor->data_ = nullptr;
        tensor->owned_ = false;
    } else if (tensor->data_ && !tensor->owned_){
        // Tensor is a view, but we are asked to allocate. Clear old view.
        tensor->data_ = nullptr;
    }


    long long num_elements = tensor->get_element_count();
    if (num_elements == 0 && !tensor->dimensions_.empty()) { // e.g. a dimension is zero
         tensor->data_ = nullptr; // No data to allocate
         tensor->owned_ = true; // Technically owns "nothing"
         tensor->calculate_strides(); // Strides might still be relevant conceptually
         return ROCQ_STATUS_SUCCESS;
    }
    if (num_elements < 0) return ROCQ_STATUS_INVALID_VALUE; // Should not happen with positive dims
    if (tensor->dimensions_.empty() && num_elements == 0) { // Scalar case, treat as 1 element for allocation
        num_elements = 1;
        // tensor->dimensions_ = {1}; // Optionally, represent scalar as 1D tensor of size 1
    }


    size_t size_bytes = num_elements * sizeof(rocComplex);
    if (size_bytes == 0) { // If truly 0 elements and 0 bytes (e.g. empty dimensions vector)
        tensor->data_ = nullptr;
        tensor->owned_ = true;
        tensor->strides_.clear();
        return ROCQ_STATUS_SUCCESS;
    }

    hipError_t hip_err = hipMalloc(&(tensor->data_), size_bytes);
    if (hip_err != hipSuccess) {
        tensor->data_ = nullptr;
        tensor->owned_ = false;
        return ROCQ_STATUS_ALLOCATION_FAILED;
    }
    tensor->owned_ = true;
    if (tensor->strides_.empty() && !tensor->dimensions_.empty()) { // Calculate strides if not already set
        tensor->calculate_strides();
    }
    return ROCQ_STATUS_SUCCESS;
}

/**
 * @brief Frees memory for a rocTensor on the device if it's owned by the struct.
 *
 * @param tensor Pointer to the rocTensor struct. If `owned_` is true, `data_` will be freed
 *               and set to nullptr, and `owned_` to false.
 * @return rocqStatus_t Status of the operation.
 */
inline rocqStatus_t rocTensorFree(rocTensor* tensor) {
    if (!tensor) return ROCQ_STATUS_INVALID_VALUE;
    if (tensor->owned_ && tensor->data_) {
        hipError_t hip_err = hipFree(tensor->data_);
        tensor->data_ = nullptr;
        tensor->owned_ = false;
        if (hip_err != hipSuccess) {
            return ROCQ_STATUS_HIP_ERROR;
        }
    } else if (!tensor->owned_ && tensor->data_) {
        // Not owned, just clear the view
        tensor->data_ = nullptr;
    }
    // If !tensor->owned_ and !tensor->data_, nothing to do.
    return ROCQ_STATUS_SUCCESS;
}

/**
 * @brief Permutes the modes of an input tensor and stores the result in an output tensor.
 *
 * The permutation is defined by `host_permutation_map`, where `host_permutation_map[new_mode_idx] = old_mode_idx`.
 * For example, to transpose a 2D matrix (modes 0, 1 -> 1, 0), host_permutation_map would be {1, 0}.
 * The output_tensor must be pre-allocated with the correct permuted dimensions and its data pointer set.
 * Input and output tensors must have the same rank (number of modes) and total number of elements.
 *
 * @param output_tensor Pointer to the rocTensor struct for the output (permuted) tensor.
 *                      Its `data_`, `dimensions_`, and `strides_` must be set appropriately for the permuted layout.
 * @param input_tensor Pointer to the const rocTensor struct for the input tensor.
 *                     Its `data_`, `dimensions_`, and `strides_` must be valid.
 * @param host_permutation_map A std::vector<int> on the host defining the permutation.
 *                             `host_permutation_map[i]` is the old mode index that maps to the new mode index `i`.
 * @return rocqStatus_t Status of the operation.
 */
rocqStatus_t rocTensorPermute(
    rocTensor* output_tensor,
    const rocTensor* input_tensor,
    const std::vector<int>& host_permutation_map);


/**
 * @brief Conceptually contracts two tensors (tensorA, tensorB) using rocBLAS GEMM
 *        and stores the result in `result_tensor`.
 *
 * @note This is a conceptual wrapper. The current implementation is a STUB and
 *       does NOT perform actual tensor permutation, reshaping, or correct contraction.
 *       It primarily serves to establish the API and ensure rocBLAS linkage.
 *       The `contraction_indices_spec` is not fully parsed or used yet.
 *       `result_tensor` must be pre-allocated by the caller with the expected dimensions.
 *
 * A full implementation would involve:
 * 1. Parsing `contraction_indices_spec` (e.g., an Einstein summation string or explicit index pairs).
 * 2. Permuting `tensorA` and `tensorB` so that contracted modes are contiguous
 *    and uncontracted modes are contiguous, suitable for GEMM.
 * 3. Reshaping (casting) the permuted tensors into 2D matrices (tensor_A_matrix, tensor_B_matrix).
 * 4. Performing the matrix multiplication: C = A * B using `rocblas_cgemm`.
 * 5. Reshaping the resulting matrix C back into the `result_tensor`'s correct higher-order shape.
 *
 * @param result_tensor Pointer to the rocTensor struct for the output. Must be pre-allocated.
 * @param tensorA Pointer to the first input rocTensor.
 * @param tensorB Pointer to the second input rocTensor.
 * @param contraction_indices_spec A string or other structure specifying how indices are contracted.
 *                                 (Currently a placeholder, not fully utilized).
 * @param blas_handle A rocBLAS handle, assumed to be initialized by the caller.
 * @param stream The HIP stream to use for rocBLAS operations.
 * @return rocqStatus_t Status of the operation. ROCQ_STATUS_NOT_IMPLEMENTED for actual contraction logic.
 */
rocqStatus_t rocTensorContractWithRocBLAS(
    rocTensor* result_tensor,
    const rocTensor* tensorA,
    const rocTensor* tensorB,
    const char* contraction_indices_spec, // Placeholder for actual spec
    rocblas_handle blas_handle,
    hipStream_t stream);

} // namespace util
} // namespace rocquantum

#endif // ROC_TENSOR_UTIL_H
