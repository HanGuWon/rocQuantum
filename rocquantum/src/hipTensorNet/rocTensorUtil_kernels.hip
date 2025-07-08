#include "rocquantum/hipStateVec.h" // For rocComplex
#include "rocquantum/rocTensorUtil.h" // For rocTensor struct (though it's in include path)
#include <hip/hip_runtime.h>

// Helper to calculate flat index from multi-dimensional indices and strides
__device__ inline long long calculate_flat_index(
    const long long* multi_dim_indices,
    const long long* strides,
    int num_modes) {
    long long flat_index = 0;
    for (int i = 0; i < num_modes; ++i) {
        flat_index += multi_dim_indices[i] * strides[i];
    }
    return flat_index;
}

// Helper to get multi-dimensional indices from a flat index
__device__ inline void get_multi_dim_indices(
    long long flat_index,
    const long long* dimensions,
    long long* multi_dim_indices,
    int num_modes) {
    long long current_flat_index = flat_index;
    for (int i = 0; i < num_modes; ++i) {
        multi_dim_indices[i] = current_flat_index % dimensions[i];
        current_flat_index /= dimensions[i];
    }
}


/**
 * @brief Generic tensor permutation kernel.
 *
 * @param output_data Pointer to the output tensor data on device.
 * @param input_data Pointer to the input tensor data on device.
 * @param d_input_dims Device pointer to input tensor dimensions.
 * @param d_input_strides Device pointer to input tensor strides.
 * @param d_output_strides Device pointer to output tensor strides (based on permuted dims).
 * @param d_permutation_map Device pointer to the permutation map (new_pos_i = perm_map[old_pos_i]).
 *                          Example: if mode 0 moves to 2, mode 1 to 0, mode 2 to 1 for a 3-mode tensor,
 *                          permutation_map would be {2, 0, 1}. So, old_indices[0] goes to new_indices[2], etc.
 *                          NO, this is incorrect. permutation_map[new_idx_pos] = old_idx_pos
 *                          So, new_indices[i] = old_indices[permutation_map[i]]
 *                          Corrected: permutation_map[old_mode_idx] = new_mode_idx
 *                          So, if old_mode 0 maps to new_mode 2, permutation_map[0] = 2.
 *                          When calculating output index using input's multi-dim-indices:
 *                          output_multi_indices[permutation_map[j]] = input_multi_indices[j]
 *
 * Let's redefine permutation_map: permutation_map[old_input_mode_idx] = corresponding_new_output_mode_idx
 * Example: Input (i,j,k), Output (k,i,j). permutation_map = {1, 2, 0} (i->pos1, j->pos2, k->pos0 in output)
 *
 * Alternative permutation_map: p[new_idx] = old_idx.
 * So, output_coords[k] = input_coords[p[k]].
 * E.g., input (d0, d1, d2), output (d1, d2, d0). p = {1, 2, 0}.
 * output_coord[0] (for d1) = input_coord[p[0]=1]
 * output_coord[1] (for d2) = input_coord[p[1]=2]
 * output_coord[2] (for d0) = input_coord[p[2]=0]
 * This seems more standard for gather-like operations.
 *
 * @param num_modes Number of modes/dimensions in the tensor.
 * @param total_elements Total number of elements in the tensor.
 */
__global__ void permute_tensor_kernel(
    rocComplex* output_data,
    const rocComplex* input_data,
    const long long* d_input_dims,      // device array
    const long long* d_input_strides,   // device array
    const long long* d_output_dims,     // device array (permuted input_dims)
    const long long* d_output_strides,  // device array
    const int* d_permutation_map,       // device array: p[new_mode_idx] = old_mode_idx
    int num_modes,
    long long total_elements) {

    long long output_flat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_flat_idx < total_elements) {
        long long input_multi_indices[16]; // Max 16 modes, adjust if necessary
        long long output_multi_indices[16];

        // 1. Get multi-dimensional indices for the current output element
        long long temp_flat_idx = output_flat_idx;
        for (int i = 0; i < num_modes; ++i) {
            output_multi_indices[i] = temp_flat_idx % d_output_dims[i];
            temp_flat_idx /= d_output_dims[i];
        }

        // 2. Determine the corresponding multi-dimensional indices in the input tensor
        //    using the permutation map: output_multi_indices[new_pos] = input_multi_indices[old_pos]
        //    No, it's: input_multi_indices[old_pos] = output_multi_indices[new_pos] where new_pos = map[old_pos]
        //    Using p[new_idx] = old_idx:
        //    input_multi_indices[d_permutation_map[i]] = output_multi_indices[i]
        //    This means we are effectively "scattering" output_multi_indices to input_multi_indices based on map.
        //    This is for constructing the input coordinates from output coordinates.
        for (int i = 0; i < num_modes; ++i) {
            input_multi_indices[d_permutation_map[i]] = output_multi_indices[i];
        }

        // 3. Calculate the flat index for the input tensor
        long long input_flat_idx = 0;
        for (int i = 0; i < num_modes; ++i) {
            input_flat_idx += input_multi_indices[i] * d_input_strides[i];
        }

        // 4. Perform the copy
        output_data[output_flat_idx] = input_data[input_flat_idx];
    }
}

/**
 * @brief Simpler permutation kernel where input and output dimensions are known to be permutations
 *        of each other. The permutation array `p` indicates for each dimension `i` of the output tensor,
 *        which dimension `p[i]` of the input tensor it corresponds to.
 *
 * @param p_output_data Device pointer to the output tensor data.
 * @param p_input_data Device pointer to the input tensor data.
 * @param p_input_dims Device pointer to array of input dimensions.
 * @param p_output_dims Device pointer to array of output dimensions.
 * @param p_permutation Device pointer to array p, where p[i] is the input dimension corresponding to output dimension i.
 * @param num_dims Number of dimensions.
 * @param num_elements Total number of elements in the tensor.
 */
__global__ void general_permute_kernel(rocComplex* p_output_data, const rocComplex* p_input_data,
                                       const long long* p_input_dims, const long long* p_output_dims,
                                       const int* p_permutation, int num_dims, long long num_elements) {
    long long gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < num_elements) {
        long long temp_gid = gid;
        long long current_input_idx = 0;
        long long factor = 1;

        // Calculate output coordinates
        long long output_coords[16]; // Max 16 dims, consistent with above
        for (int i = 0; i < num_dims; ++i) {
            output_coords[i] = temp_gid % p_output_dims[i];
            temp_gid /= p_output_dims[i];
        }

        // Calculate input index based on permuted output coordinates
        // output_coord[i] is the coordinate for the i-th dimension of the output tensor.
        // This dimension corresponds to the p_permutation[i]-th dimension of the input tensor.
        for (int i = 0; i < num_dims; ++i) {
            long long current_coord_in_input_dim_space = output_coords[i]; // This is coord for output_dims[i]
            // We need to find where this p_permutation[i]-th dimension of input is in the factor calculation
            long long input_dim_stride = 1;
            for(int j=0; j < p_permutation[i]; ++j){
                input_dim_stride *= p_input_dims[j];
            }
            current_input_idx += current_coord_in_input_dim_space * input_dim_stride;
        }
        // The above stride calculation is for packed C-order.
        // If using precomputed strides (passed as d_input_strides), it's simpler:
        // for (int i=0; i < num_dims; ++i) {
        //    input_coords[p_permutation[i]] = output_coords[i];
        // }
        // current_input_idx = calculate_flat_index(input_coords, d_input_strides, num_dims);
        // For now, let's use the easier indexing from the first kernel if strides are available.
        // This general_permute_kernel is a bit redundant if the first one is robust.
        // Sticking to permute_tensor_kernel which uses strides directly.

        // This kernel (general_permute_kernel) will be removed in favor of the one above.
        // The logic for permute_tensor_kernel is more aligned with how strides are typically used.
    }
}
