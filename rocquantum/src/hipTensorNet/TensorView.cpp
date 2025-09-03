#include "rocquantum/TensorView.h"

TensorView create_sliced_view(
    const rocquantum::util::rocTensor& parent_tensor,
    const std::string& slice_label,
    int slice_start_index,
    int slice_width)
{
    // --- 1. Find the axis corresponding to the slice label ---
    int slice_axis = -1;
    for (size_t i = 0; i < parent_tensor.labels_.size(); ++i) {
        if (parent_tensor.labels_[i] == slice_label) {
            slice_axis = i;
            break;
        }
    }

    if (slice_axis == -1) {
        throw std::runtime_error("create_sliced_view error: Slice label '" + slice_label + "' not found in the parent tensor's labels.");
    }

    // --- 2. Validate slice boundaries ---
    if (slice_start_index < 0 || slice_width <= 0 || (slice_start_index + slice_width) > parent_tensor.dimensions_[slice_axis]) {
        throw std::out_of_range("create_sliced_view error: The requested slice is out of bounds.");
    }

    // --- 3. Calculate strides of the parent tensor (column-major) ---
    std::vector<long long> parent_strides(parent_tensor.dimensions_.size());
    if (!parent_strides.empty()) {
        parent_strides[0] = 1;
        for (size_t i = 1; i < parent_tensor.dimensions_.size(); ++i) {
            parent_strides[i] = parent_strides[i - 1] * parent_tensor.dimensions_[i - 1];
        }
    }

    // --- 4. Calculate the starting offset for the view ---
    size_t offset = static_cast<size_t>(slice_start_index) * parent_strides[slice_axis];

    // --- 5. Determine the dimensions of the new view ---
    std::vector<long long> view_dims = parent_tensor.dimensions_;
    view_dims[slice_axis] = slice_width;

    // --- 6. Assemble and return the TensorView ---
    TensorView view;
    view.data_ptr = parent_tensor.data_;
    view.dimensions = view_dims;
    view.strides = parent_strides;
    view.offset_in_elements = offset;

    return view;
}