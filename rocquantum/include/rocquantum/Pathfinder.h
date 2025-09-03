#ifndef PATHFINDER_H
#define PATHFINDER_H

#include "hipTensorNet_internal_types.h" // For ContractionPlan
#include "hipTensorNet_api.h"            // For hipTensorNetContractionOptimizerConfig_t

// Forward declaration to avoid including the full TensorNetwork definition
// in this header, reducing compilation dependencies.
namespace rocquantum {
    template <typename T>
    class TensorNetwork;
}

namespace rocquantum {

/**
 * @class Pathfinder
 * @brief A class responsible for finding an efficient contraction path for a tensor network.
 *
 * This class encapsulates the logic for different pathfinding algorithms (Greedy, KaHyPar, METIS)
 * and produces a ContractionPlan that can be executed by the TensorNetwork contractor.
 */
class Pathfinder {
public:
    /**
     * @brief Finds the optimal contraction path for a given tensor network.
     *
     * This is the main entry point for the pathfinding process. It takes the tensor network
     * and optimizer configuration, selects the appropriate algorithm, and returns a
     * detailed contraction plan.
     *
     * @tparam T The data type of the tensors in the network (e.g., float, double).
     * @param network The TensorNetwork object containing the tensors to be contracted.
     * @param config The optimizer configuration specifying which algorithm to use and its settings.
     * @return A ContractionPlan object detailing the steps for contraction.
     */
    template <typename T>
    internal::ContractionPlan findOptimalPath(
        const TensorNetwork<T>& network,
        const hipTensorNetContractionOptimizerConfig_t& config);

private:
    /**
     * @brief Finds a contraction path using a simple greedy algorithm.
     * @tparam T Data type of the tensors.
     * @param network The tensor network.
     * @param config The optimizer configuration.
     * @return A ContractionPlan.
     */
    template <typename T>
    internal::ContractionPlan findGreedyPath(
        const TensorNetwork<T>& network,
        const hipTensorNetContractionOptimizerConfig_t& config);

    /**
     * @brief Finds a contraction path using the KaHyPar hypergraph partitioning library.
     * @tparam T Data type of the tensors.
     * @param network The tensor network.
     * @param config The optimizer configuration.
     * @return A ContractionPlan.
     */
    template <typename T>
    internal::ContractionPlan findKaHyParPath(
        const TensorNetwork<T>& network,
        const hipTensorNetContractionOptimizerConfig_t& config);

    /**
     * @brief Finds a contraction path using the METIS graph partitioning library.
     * @tparam T Data type of the tensors.
     * @param network The tensor network.
     * @param config The optimizer configuration.
     * @return A ContractionPlan.
     */
    template <typename T>
    internal::ContractionPlan findMetisPath(
        const TensorNetwork<T>& network,
        const hipTensorNetContractionOptimizerConfig_t& config);
};

} // namespace rocquantum

#endif // PATHFINDER_H
