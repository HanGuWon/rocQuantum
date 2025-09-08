#ifndef PERMUTATION_KERNELS_H
#define PERMUTATION_KERNELS_H

#include <hip/hip_runtime.h>
#include <vector>

/**
 * @brief Host-side wrapper to launch the tensor permutation kernel.
 *
 * This function handles the logistics of setting up and launching the HIP kernel.
 * It calculates tensor strides, allocates and transfers metadata (dims, maps) to the GPU,
 * determines the optimal grid/block configuration, and launches the kernel.
 *
 * @tparam T The data type of the tensor elements.
 * @param output_tensor Pointer to the destination tensor in GPU memory.
 * @param input_tensor Pointer to the source tensor in GPU memory.
 * @param input_dims A std::vector containing the dimensions/extents of the input tensor.
 * @param permutation_map A std::vector where `map[i]` indicates that the i-th dimension
 *                        of the output tensor corresponds to the `map[i]`-th dimension
 *                        of the input tensor.
 * @param stream The HIP stream on which to execute the kernel.
 */
template<typename T>
void launch_permute_tensor(
    T* output_tensor,
    const T* input_tensor,
    const std::vector<long long>& input_dims,
    const std::vector<int>& permutation_map,
    hipStream_t stream
);

#endif // PERMUTATION_KERNELS_H
