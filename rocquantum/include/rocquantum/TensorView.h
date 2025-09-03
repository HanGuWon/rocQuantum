#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include "rocquantum/rocTensorUtil.h" // For rocTensor definition

/**
 * @struct TensorView
 * @brief Describes a view into a larger tensor's GPU memory without owning the data.
 */
struct TensorView {
    void* data_ptr = nullptr;
    std::vector<long long> dimensions;
    std::vector<long long> strides;
    size_t offset_in_elements = 0;

    template<typename T>
    T* get_view_data_ptr() const {
        return static_cast<T*>(data_ptr) + offset_in_elements;
    }
};

/**
 * @brief Creates a sliced view of a parent tensor along a specific mode (label).
 */
TensorView create_sliced_view(
    const rocquantum::util::rocTensor& parent_tensor,
    const std::string& slice_label,
    int slice_start_index,
    int slice_width);

#endif // TENSOR_VIEW_H