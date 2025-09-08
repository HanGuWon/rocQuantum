#include "rocquantum/PermutationKernels.h"
#include <numeric>
#include <stdexcept>
#include <string>

// Error checking utility
#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + " in file " + __FILE__ + " at line " + std::to_string(__LINE__)); \
    } \
} while(0)

/**
 * @brief A high-performance HIP kernel to perform an arbitrary N-dimensional tensor permutation.
 *
 * This kernel is designed for memory-bound operations. Each thread is responsible for
 * computing the location of and copying a single element from the input tensor to the
 * output tensor. The mapping is done such that threads write to contiguous memory
 * locations in the output tensor, ensuring perfectly coalesced memory writes.
 *
 * @tparam T The data type of the tensor elements (e.g., float, rocComplex).
 * @param output_tensor Pointer to the destination tensor in GPU memory.
 * @param input_tensor Pointer to the source tensor in GPU memory.
 * @param d_input_strides Device pointer to an array of strides for the input tensor.
 * @param d_output_strides Device pointer to an array of strides for the output tensor.
 * @param d_permutation_map Device pointer to a map where `map[i]` indicates that the
 *                          i-th dimension of the output tensor corresponds to the
 *                          `map[i]`-th dimension of the input tensor.
 * @param rank The number of dimensions (rank) of the tensor.
 * @param total_elements The total number of elements in the tensor.
 */
template<typename T>
__global__ void permute_tensor_kernel(
    T* output_tensor,
    const T* input_tensor,
    const long long* d_input_strides,
    const long long* d_output_strides,
    const int* d_permutation_map,
    int rank,
    long long total_elements
) {
    // Calculate the global linear index for the current thread.
    // This index corresponds directly to the element's position in the output tensor.
    long long output_linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure the thread is not processing an element beyond the tensor's size.
    if (output_linear_idx < total_elements) {
        long long remaining_idx = output_linear_idx;
        long long source_linear_idx = 0;

        // This loop de-linearizes the output index into a multi-dimensional coordinate,
        // permutes it, and re-linearizes it into the source index in a single pass.
        #pragma unroll
        for (int i = 0; i < rank; ++i) {
            // Get the stride for the current dimension of the output tensor.
            long long output_stride = d_output_strides[i];
            
            // Calculate the coordinate value for this dimension.
            long long coord = remaining_idx / output_stride;
            remaining_idx %= output_stride;

            // Find the corresponding dimension in the input tensor using the permutation map.
            int source_dim = d_permutation_map[i];
            
            // Use the coordinate value to calculate the partial source index using the input tensor's strides.
            source_linear_idx += coord * d_input_strides[source_dim];
        }

        // Read from the calculated source index (gather) and write to the coalesced
        // destination index (scatter).
        output_tensor[output_linear_idx] = input_tensor[source_linear_idx];
    }
}

/**
 * @brief Host-side wrapper to launch the tensor permutation kernel.
 */
template<typename T>
void launch_permute_tensor(
    T* output_tensor,
    const T* input_tensor,
    const std::vector<long long>& input_dims,
    const std::vector<int>& permutation_map,
    hipStream_t stream
) {
    int rank = input_dims.size();
    if (rank == 0) return; // Nothing to do for a scalar.
    if (rank != permutation_map.size()) {
        throw std::invalid_argument("Dimension mismatch between input_dims and permutation_map.");
    }

    // Calculate total number of elements
    long long total_elements = 1;
    for(long long dim : input_dims) {
        total_elements *= dim;
    }
    if (total_elements == 0) return;

    // --- Prepare metadata on the host ---
    std::vector<long long> input_strides(rank);
    std::vector<long long> output_strides(rank);
    std::vector<long long> output_dims(rank);

    // Calculate input strides (assuming column-major layout)
    input_strides[0] = 1;
    for (int i = 1; i < rank; ++i) {
        input_strides[i] = input_strides[i - 1] * input_dims[i - 1];
    }

    // Calculate output dimensions and strides based on the permutation
    for (int i = 0; i < rank; ++i) {
        output_dims[i] = input_dims[permutation_map[i]];
    }
    output_strides[0] = 1;
    for (int i = 1; i < rank; ++i) {
        output_strides[i] = output_strides[i - 1] * output_dims[i - 1];
    }

    // --- Allocate and transfer metadata to the GPU ---
    long long* d_input_strides;
    long long* d_output_strides;
    int* d_permutation_map;

    HIP_CHECK(hipMallocAsync(&d_input_strides, rank * sizeof(long long), stream));
    HIP_CHECK(hipMallocAsync(&d_output_strides, rank * sizeof(long long), stream));
    HIP_CHECK(hipMallocAsync(&d_permutation_map, rank * sizeof(int), stream));

    HIP_CHECK(hipMemcpyAsync(d_input_strides, input_strides.data(), rank * sizeof(long long), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_output_strides, output_strides.data(), rank * sizeof(long long), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_permutation_map, permutation_map.data(), rank * sizeof(int), hipMemcpyHostToDevice, stream));

    // --- Launch Kernel ---
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        permute_tensor_kernel<T>,
        dim3(grid_size),
        dim3(block_size),
        0, // shared memory
        stream,
        output_tensor,
        input_tensor,
        d_input_strides,
        d_output_strides,
        d_permutation_map,
        rank,
        total_elements
    );

    // --- Free temporary device memory ---
    HIP_CHECK(hipFreeAsync(d_input_strides, stream));
    HIP_CHECK(hipFreeAsync(d_output_strides, stream));
    HIP_CHECK(hipFreeAsync(d_permutation_map, stream));
}

// Explicit template instantiations
#include "rocquantum/hipStateVec.h" // For rocComplex types
template void launch_permute_tensor<rocComplex>(T* output_tensor, const T* input_tensor, const std::vector<long long>& input_dims, const std::vector<int>& permutation_map, hipStream_t stream);
template void launch_permute_tensor<rocDoubleComplex>(T* output_tensor, const T* input_tensor, const std::vector<long long>& input_dims, const std::vector<int>& permutation_map, hipStream_t stream);
