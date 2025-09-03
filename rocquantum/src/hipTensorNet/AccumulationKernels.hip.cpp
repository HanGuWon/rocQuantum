#include <hip/hip_runtime.h>
#include <vector>
#include <string>
#include <numeric>
#include "rocquantum/TensorView.h" // Assumes TensorView struct is defined here

template <typename T>
__global__ void accumulate_sliced_result_kernel(
    T* destination,
    const T* source,
    size_t num_source_elements,
    const long long* dest_strides,
    const long long* source_dims,
    int source_rank,
    size_t dest_offset_in_elements)
{
    size_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_idx >= num_source_elements) {
        return;
    }

    size_t dest_relative_idx = 0;
    size_t temp_idx = source_idx;

    for (int i = 0; i < source_rank; ++i) {
        long long coord = temp_idx % source_dims[i];
        dest_relative_idx += coord * dest_strides[i];
        temp_idx /= source_dims[i];
    }

    size_t dest_idx = dest_offset_in_elements + dest_relative_idx;
    destination[dest_idx] += source[source_idx];
}

template <typename T>
void launch_accumulate_sliced_result(
    const TensorView& destination_view,
    const TensorView& source_view,
    size_t destination_slice_offset,
    hipStream_t stream)
{
    size_t num_source_elements = std::accumulate(
        source_view.dimensions.begin(),
        source_view.dimensions.end(),
        1LL,
        std::multiplies<long long>());

    if (num_source_elements == 0) {
        return;
    }

    const int threads_per_block = 256;
    const int num_blocks = (num_source_elements + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        accumulate_sliced_result_kernel<T>,
        dim3(num_blocks),
        dim3(threads_per_block),
        0,
        stream,
        static_cast<T*>(destination_view.data_ptr),
        static_cast<const T*>(source_view.data_ptr),
        num_source_elements,
        destination_view.strides.data(),
        source_view.dimensions.data(),
        static_cast<int>(source_view.dimensions.size()),
        destination_slice_offset
    );
}

// Explicit instantiations for common types
#include "rocquantum/rocComplex.h"
template void launch_accumulate_sliced_result<float>(const TensorView&, const TensorView&, size_t, hipStream_t);
template void launch_accumulate_sliced_result<rocComplex>(const TensorView&, const TensorView&, size_t, hipStream_t);