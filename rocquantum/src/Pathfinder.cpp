#include "Pathfinder.h"
#include "hipTensorNet.h" // Assuming this is where TensorNetwork and rocTensor are fully defined.
#include <stdexcept>      // For std::runtime_error
#include <limits>         // For std::numeric_limits
#include <algorithm>      // For std::find, std::sort, std::set_difference
#include <numeric>        // For std::accumulate
#include <map>            // For std::map
#include <iostream>       // For debugging
#include <unordered_map>  // For label_to_vertex map

// Conditionally include and define KaHyPar related code only if the library was found by CMake.
#ifdef HAS_KAHYPAR

// --- Mock KaHyPar C API Definition ---
// In a real project, this would be replaced by including the actual KaHyPar header.
// e.g., #include <kahypar-capi.h>
extern "C" {
    typedef void* kahypar_context_t;
    typedef int32_t kahypar_hyperedge_id_t;
    typedef int32_t kahypar_hypernode_id_t;
    typedef int32_t kahypar_partition_id_t;

    kahypar_context_t kahypar_context_new() { return new int(1); }
    void kahypar_context_free(kahypar_context_t context) { if(context) delete static_cast<int*>(context); }
    void kahypar_set_parameter(kahypar_context_t, int, const char*) {}
    void kahypar_partition(
        kahypar_context_t context,
        const kahypar_hypernode_id_t num_vertices,
        const kahypar_hyperedge_id_t num_hyperedges,
        const double imbalance,
        const kahypar_partition_id_t k, // Number of blocks to partition into
        const kahypar_hypernode_id_t* vertex_weights,
        const kahypar_hyperedge_id_t* hyperedge_weights,
        const size_t* hyperedge_indices_starts,
        const kahypar_hypernode_id_t* hyperedge_indices,
        double* objective,
        kahypar_partition_id_t* partition_ids)
    {
        // Mock implementation: Simply bisects the hyperedges.
        // The real KaHyPar uses a sophisticated algorithm to find an optimal cut.
        if (num_hyperedges > 0) {
            for (kahypar_hyperedge_id_t i = 0; i < num_hyperedges; ++i) {
                partition_ids[i] = (i < num_hyperedges / 2) ? 0 : 1;
            }
        }
        *objective = 0.0; // Mock objective value
    }
}
// --- End Mock API Definition ---

#endif // HAS_KAHYPAR


namespace rocquantum {

// Helper structure to hold temporary tensor metadata during pathfinding.
// This avoids modifying the original network and is lighter than rocTensor.
struct TensorInfo {
    int original_index; // Index in the initial tensor list, or a new ID for an intermediate
    std::vector<long long> dims;
    std::vector<std::string> labels;
};

/**
 * @brief A helper function to analyze a potential pairwise contraction.
 *
 * @param t1 First tensor metadata.
 * @param t2 Second tensor metadata.
 * @param out_dims [out] The dimensions of the resulting tensor.
 * @param out_labels [out] The labels of the resulting tensor.
 * @param out_flops [out] The FLOP count for this contraction.
 */
static void getContractionResultInfo(
    const TensorInfo& t1, const TensorInfo& t2,
    std::vector<long long>& out_dims,
    std::vector<std::string>& out_labels,
    long long& out_flops)
{
    out_dims.clear();
    out_labels.clear();

    std::map<std::string, long long> t1_label_to_dim, t2_label_to_dim;
    for (size_t i = 0; i < t1.labels.size(); ++i) t1_label_to_dim[t1.labels[i]] = t1.dims[i];
    for (size_t i = 0; i < t2.labels.size(); ++i) t2_label_to_dim[t2.labels[i]] = t2.dims[i];

    std::vector<std::string> shared_labels;
    std::vector<std::string> t1_unique_labels;
    std::vector<std::string> t2_unique_labels;

    // Find shared and unique labels for t1
    for (const auto& label : t1.labels) {
        if (t2_label_to_dim.count(label)) {
            if (std::find(shared_labels.begin(), shared_labels.end(), label) == shared_labels.end())
                shared_labels.push_back(label);
        } else {
            t1_unique_labels.push_back(label);
        }
    }
    // Find unique labels for t2
    for (const auto& label : t2.labels) {
        if (!t1_label_to_dim.count(label)) {
            t2_unique_labels.push_back(label);
        }
    }

    // The resulting tensor has the unique labels from both tensors
    out_labels.insert(out_labels.end(), t1_unique_labels.begin(), t1_unique_labels.end());
    out_labels.insert(out_labels.end(), t2_unique_labels.begin(), t2_unique_labels.end());

    // Populate the dimensions for the resulting tensor
    for (const auto& label : out_labels) {
        if (t1_label_to_dim.count(label)) {
            out_dims.push_back(t1_label_to_dim[label]);
        } else {
            out_dims.push_back(t2_label_to_dim[label]);
        }
    }

    // Calculate FLOPs: 2 * product of all dimension sizes involved in the contraction
    long long total_dim_product = 1;
    for (const auto& dim : t1.dims) total_dim_product *= dim;
    for (const auto& dim : t2.dims) total_dim_product *= dim;

    long long shared_dim_product = 1;
    for (const auto& label : shared_labels) {
        shared_dim_product *= t1_label_to_dim[label];
    }

    out_flops = (shared_dim_product > 0) ? (2 * total_dim_product / shared_dim_product) : 0;
}

// Forward declaration for the recursive helper function
#ifdef HAS_KAHYPAR
namespace {
    template <typename T>
    TensorInfo build_plan_recursively(
        const std::vector<int>& tensor_indices_to_contract,
        std::vector<TensorInfo>& all_tensors_info, // Pass by non-const ref to add intermediates
        internal::ContractionPlan& plan,
        const hipTensorNetContractionOptimizerConfig_t& config,
        int& next_intermediate_id);
}
#endif // HAS_KAHYPAR

template <typename T>
internal::ContractionPlan Pathfinder::findOptimalPath(
    const TensorNetwork<T>& network,
    const hipTensorNetContractionOptimizerConfig_t& config)
{
    // Delegate to the appropriate private method based on the selected algorithm.
    switch (config.pathfinder_algorithm) {
        case ROCTN_PATHFINDER_ALGO_GREEDY:
            return findGreedyPath(network, config);

        case ROCTN_PATHFINDER_ALGO_KAHYPAR:
            #ifdef HAS_KAHYPAR
            return findKaHyParPath(network, config);
            #else
            throw std::runtime_error("rocQuantum was compiled without KaHyPar support. Please recompile with KaHyPar to use this feature.");
            #endif

        case ROCTN_PATHFINDER_ALGO_METIS:
            return findMetisPath(network, config);

        default:
            throw std::runtime_error("Unknown or unsupported pathfinder algorithm specified.");
    }
}

template <typename T>
internal::ContractionPlan Pathfinder::findGreedyPath(
    const TensorNetwork<T>& network,
    const hipTensorNetContractionOptimizerConfig_t& config)
{
    internal::ContractionPlan plan;
    plan.algorithm = ROCTN_PATHFINDER_ALGO_GREEDY;

    // 1. Initialization: Create a working copy of tensor metadata.
    const auto& initial_tensors = network.getInitialTensors();
    if (initial_tensors.size() <= 1) {
        return plan; // Nothing to contract.
    }

    std::vector<TensorInfo> active_tensors;
    for (size_t i = 0; i < initial_tensors.size(); ++i) {
        active_tensors.push_back({
            static_cast<int>(i),
            initial_tensors[i].getDims(),
            initial_tensors[i].getLabels()
        });
    }

    // 2. Main Loop: Continue until only one tensor remains.
    while (active_tensors.size() > 1) {
        long long best_cost = std::numeric_limits<long long>::max();
        int best_i = -1, best_j = -1;
        internal::ContractionStep best_step_candidate;

        // 3. Cost Calculation: Evaluate every possible pair.
        for (size_t i = 0; i < active_tensors.size(); ++i) {
            for (size_t j = i + 1; j < active_tensors.size(); ++j) {
                std::vector<long long> current_dims;
                std::vector<std::string> current_labels;
                long long current_flops;

                getContractionResultInfo(active_tensors[i], active_tensors[j],
                                         current_dims, current_labels, current_flops);

                // 4. Selection: Use FLOP count as the cost metric.
                if (current_flops < best_cost) {
                    best_cost = current_flops;
                    best_i = i;
                    best_j = j;
                    best_step_candidate.flops = current_flops;
                    best_step_candidate.resulting_dims = current_dims;
                    best_step_candidate.resulting_labels = current_labels;
                }
            }
        }

        if (best_i == -1) {
            // This can happen if the network is disconnected.
            throw std::runtime_error("Could not find a valid pair to contract in the network.");
        }

        // 5. Plan Update: Record the best contraction for this iteration.
        best_step_candidate.tensor_index_1 = active_tensors[best_i].original_index;
        best_step_candidate.tensor_index_2 = active_tensors[best_j].original_index;
        plan.steps.push_back(best_step_candidate);

        // 6. Network Update: "Contract" the pair in our metadata copy.
        TensorInfo new_tensor;
        new_tensor.original_index = initial_tensors.size() + plan.steps.size() - 1; // New unique ID for the intermediate
        new_tensor.dims = best_step_candidate.resulting_dims;
        new_tensor.labels = best_step_candidate.resulting_labels;

        // Remove the contracted tensors. Erase the one with the larger index first
        // to keep the smaller index valid.
        active_tensors.erase(active_tensors.begin() + best_j);
        active_tensors.erase(active_tensors.begin() + best_i);

        // Add the new tensor representing the result.
        active_tensors.push_back(new_tensor);
    }

    // 7. Finalization: Calculate summary statistics for the plan.
    plan.total_flops = 0;
    plan.largest_intermediate_size_bytes = 0;
    size_t sizeof_T = sizeof(T);

    for (const auto& step : plan.steps) {
        plan.total_flops += step.flops;
        long long intermediate_size = std::accumulate(
            step.resulting_dims.begin(), step.resulting_dims.end(),
            1LL, std::multiplies<long long>());
        
        long long intermediate_bytes = intermediate_size * sizeof(T);
        if (intermediate_bytes > plan.largest_intermediate_size_bytes) {
            plan.largest_intermediate_size_bytes = intermediate_bytes;
        }
    }

    return plan;
}

#ifdef HAS_KAHYPAR
template <typename T>
internal::ContractionPlan Pathfinder::findKaHyParPath(
    const TensorNetwork<T>& network,
    const hipTensorNetContractionOptimizerConfig_t& config)
{
    internal::ContractionPlan plan;
    plan.algorithm = ROCTN_PATHFINDER_ALGO_KAHYPAR;

    const auto& initial_tensors = network.getInitialTensors();
    if (initial_tensors.size() <= 1) {
        return plan; // Nothing to contract.
    }

    // Create a working copy of tensor metadata that can be expanded with intermediates.
    std::vector<TensorInfo> all_tensors_info;
    std::vector<int> initial_indices;
    for (size_t i = 0; i < initial_tensors.size(); ++i) {
        all_tensors_info.push_back({
            static_cast<int>(i),
            initial_tensors[i].getDims(),
            initial_tensors[i].getLabels()
        });
        initial_indices.push_back(i);
    }

    // Start the recursive pathfinding process.
    int next_intermediate_id = initial_tensors.size();
    build_plan_recursively<T>(initial_indices, all_tensors_info, plan, config, next_intermediate_id);

    // Finalization Stage: Calculate summary statistics for the completed plan.
    plan.total_flops = 0;
    plan.largest_intermediate_size_bytes = 0;
    size_t sizeof_T = sizeof(T);

    for (const auto& step : plan.steps) {
        plan.total_flops += step.flops;
        long long intermediate_elements = std::accumulate(
            step.resulting_dims.begin(), step.resulting_dims.end(),
            1LL, std::multiplies<long long>());
        
        long long intermediate_bytes = intermediate_elements * sizeof(T);
        if (intermediate_bytes > plan.largest_intermediate_size_bytes) {
            plan.largest_intermediate_size_bytes = intermediate_bytes;
        }
    }

    return plan;
}
#endif // HAS_KAHYPAR

template <typename T>
internal::ContractionPlan Pathfinder::findMetisPath(
    const TensorNetwork<T>& network,
    const hipTensorNetContractionOptimizerConfig_t& config)
{
    internal::ContractionPlan plan;
    plan.algorithm = ROCTN_PATHFINDER_ALGO_METIS;

    // TODO:
    // 1. Convert the tensor network's hypergraph into a standard graph
    //    (e.g., a line graph where indices are vertices and tensors are cliques).
    // 2. Prepare the graph data structures required by METIS (e.g., adjacency lists).
    // 3. Configure METIS options from `config.algo_config.metis_config`.
    // 4. Call METIS's graph partitioning function (e.g., METIS_PartGraphKway)
    //    recursively to determine the bisection tree.
    // 5. Translate the resulting partitioning back into a sequence of
    //    ContractionStep objects and populate the `plan`.
    // 6. Calculate total_flops and largest_intermediate_size_bytes for the plan.

    throw std::runtime_error("METIS pathfinder is not yet implemented.");
    return plan;
}

#ifdef HAS_KAHYPAR
namespace {
/**
 * @brief Helper function that recursively builds a contraction plan using bisection.
 *
 * This function takes a list of tensors, partitions them into two groups using KaHyPar,
 * recursively calls itself on each group, and then adds the final contraction step
 * between the two resulting subgroups to the plan.
 *
 * @param tensor_indices_to_contract List of indices (into all_tensors_info) to be contracted in this step.
 * @param all_tensors_info Metadata for all tensors (initial and intermediate). Intermediates are added here.
 * @param plan The final ContractionPlan object being built.
 * @param config The optimizer configuration.
 * @param next_intermediate_id The next available unique ID for an intermediate tensor.
 * @return The TensorInfo for the single tensor resulting from this recursive contraction.
 */
template <typename T>
TensorInfo build_plan_recursively(
    const std::vector<int>& tensor_indices_to_contract,
    std::vector<TensorInfo>& all_tensors_info,
    internal::ContractionPlan& plan,
    const hipTensorNetContractionOptimizerConfig_t& config,
    int& next_intermediate_id)
{
    // Base Case: If only one tensor is left, it's the result of this branch.
    if (tensor_indices_to_contract.size() == 1) {
        return all_tensors_info[tensor_indices_to_contract[0]];
    }

    // Base Case: If two tensors are left, contract them directly.
    if (tensor_indices_to_contract.size() == 2) {
        const auto& t1_info = all_tensors_info[tensor_indices_to_contract[0]];
        const auto& t2_info = all_tensors_info[tensor_indices_to_contract[1]];

        internal::ContractionStep step;
        step.tensor_index_1 = t1_info.original_index;
        step.tensor_index_2 = t2_info.original_index;

        getContractionResultInfo(t1_info, t2_info, step.resulting_dims, step.resulting_labels, step.flops);
        plan.steps.push_back(step);

        TensorInfo result_info;
        result_info.original_index = next_intermediate_id++;
        result_info.dims = step.resulting_dims;
        result_info.labels = step.resulting_labels;
        all_tensors_info.push_back(result_info); // Add intermediate to the global list
        return result_info;
    }

    // --- Stage 1: Convert Tensor Network to Hypergraph ---
    // Each tensor index (label) is a hypergraph vertex.
    // Each tensor is a hyperedge connecting the vertices (indices) it contains.
    std::unordered_map<std::string, kahypar_hypernode_id_t> label_to_vertex;
    kahypar_hypernode_id_t next_vertex_id = 0;

    std::vector<size_t> hyperedge_indices_starts;
    std::vector<kahypar_hypernode_id_t> hyperedge_indices;
    std::vector<kahypar_hyperedge_id_t> hyperedge_weights;

    for (int tensor_idx : tensor_indices_to_contract) {
        const auto& tensor = all_tensors_info[tensor_idx];
        hyperedge_indices_starts.push_back(hyperedge_indices.size());

        for (const auto& label : tensor.labels) {
            if (label_to_vertex.find(label) == label_to_vertex.end()) {
                label_to_vertex[label] = next_vertex_id++;
            }
            hyperedge_indices.push_back(label_to_vertex[label]);
        }
        // The weight of a hyperedge (tensor) can be its size (number of elements).
        // For simplicity, we use 1 here. A more advanced implementation might use log(size).
        hyperedge_weights.push_back(1);
    }
    hyperedge_indices_starts.push_back(hyperedge_indices.size());

    const kahypar_hypernode_id_t num_vertices = next_vertex_id;
    const kahypar_hyperedge_id_t num_hyperedges = tensor_indices_to_contract.size();
    std::vector<kahypar_hypernode_id_t> vertex_weights(num_vertices, 1);

    // --- Stage 2: Interact with KaHyPar API ---
    kahypar_context_t* context = kahypar_context_new();
    // kahypar_set_parameter(context, ...); // Set parameters if needed from config

    double imbalance = config.algo_config.kahypar_config.imbalance_factor;
    kahypar_partition_id_t k = 2; // Always bisect into two partitions.
    double objective = 0.0;
    std::vector<kahypar_partition_id_t> partition_ids(num_hyperedges);

    // Call KaHyPar to partition the hyperedges (tensors) into two groups.
    // Note: KaHyPar partitions vertices, but the result can be mapped back to hyperedges.
    // Here, we assume our mock API partitions hyperedges directly for simplicity.
    kahypar_partition(context, num_vertices, num_hyperedges, imbalance, k,
                      vertex_weights.data(), hyperedge_weights.data(),
                      hyperedge_indices_starts.data(), hyperedge_indices.data(),
                      &objective, partition_ids.data());
    
    kahypar_context_free(context);

    // --- Stage 3: Translate Partition Result back to Contraction Plan ---
    std::vector<int> group_a_indices, group_b_indices;
    for (size_t i = 0; i < num_hyperedges; ++i) {
        if (partition_ids[i] == 0) {
            group_a_indices.push_back(tensor_indices_to_contract[i]);
        } else {
            group_b_indices.push_back(tensor_indices_to_contract[i]);
        }
    }
    
    // This can happen if one partition is empty, just contract greedily.
    if (group_a_indices.empty() || group_b_indices.empty()) {
        // Fallback for skewed partitions: contract the first tensor with the rest.
        // A more robust implementation would handle this more gracefully.
        group_a_indices = {tensor_indices_to_contract[0]};
        group_b_indices.assign(tensor_indices_to_contract.begin() + 1, tensor_indices_to_contract.end());
    }

    // Recursive calls for the two new subgroups.
    TensorInfo result_a = build_plan_recursively<T>(group_a_indices, all_tensors_info, plan, config, next_intermediate_id);
    TensorInfo result_b = build_plan_recursively<T>(group_b_indices, all_tensors_info, plan, config, next_intermediate_id);

    // Add the final contraction step for the results of the two subgroups.
    internal::ContractionStep final_step;
    final_step.tensor_index_1 = result_a.original_index;
    final_step.tensor_index_2 = result_b.original_index;
    getContractionResultInfo(result_a, result_b, final_step.resulting_dims, final_step.resulting_labels, final_step.flops);
    plan.steps.push_back(final_step);

    TensorInfo final_result_info;
    final_result_info.original_index = next_intermediate_id++;
    final_result_info.dims = final_step.resulting_dims;
    final_result_info.labels = final_step.resulting_labels;
    all_tensors_info.push_back(final_result_info); // Add the final result to the list
    
    return final_result_info;
}
} // end anonymous namespace
#endif // HAS_KAHYPAR

// Explicit template instantiation for supported types to avoid linker errors.
// Add other types like double, hipFloatComplex, etc., as they become supported.
template internal::ContractionPlan Pathfinder::findOptimalPath<float>(
    const TensorNetwork<float>&, const hipTensorNetContractionOptimizerConfig_t&);
template internal::ContractionPlan Pathfinder::findOptimalPath<double>(
    const TensorNetwork<double>&, const hipTensorNetContractionOptimizerConfig_t&);

template internal::ContractionPlan Pathfinder::findGreedyPath<float>(
    const TensorNetwork<float>&, const hipTensorNetContractionOptimizerConfig_t&);
template internal::ContractionPlan Pathfinder::findGreedyPath<double>(
    const TensorNetwork<double>&, const hipTensorNetContractionOptimizerConfig_t&);

#ifdef HAS_KAHYPAR
template internal::ContractionPlan Pathfinder::findKaHyParPath<float>(
    const TensorNetwork<float>&, const hipTensorNetContractionOptimizerConfig_t&);
template internal::ContractionPlan Pathfinder::findKaHyParPath<double>(
    const TensorNetwork<double>&, const hipTensorNetContractionOptimizerConfig_t&);
#endif

template internal::ContractionPlan Pathfinder::findMetisPath<float>(
    const TensorNetwork<float>&, const hipTensorNetContractionOptimizerConfig_t&);
template internal::ContractionPlan Pathfinder::findMetisPath<double>(
    const TensorNetwork<double>&, const hipTensorNetContractionOptimizerConfig_t&);


} // namespace rocquantum
