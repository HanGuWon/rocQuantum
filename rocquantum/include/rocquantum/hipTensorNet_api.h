// The updated C-API struct with new members for slicing control.
typedef struct hipTensorNetContractionOptimizerConfig_t {
    // --- Existing Fields ---
    rocPathfinderAlgorithm_t pathfinder_algorithm;
    union {
        struct {
            double imbalance_factor;
        } kahypar_config;
        struct {
            int num_iterations;
        } metis_config;
    } algo_config;

    // --- New Slicing Configuration Fields ---

    /**
     * @brief The memory limit in bytes for any single intermediate tensor.
     *
     * If the pathfinder determines that a pairwise contraction will produce an
     * intermediate tensor larger than this limit, slicing will be triggered for
     * that specific contraction.
     * A value of 0 (the default) disables the slicing feature entirely.
     */
    size_t memory_limit_bytes;

    /**
     * @brief (Optional) Manually specifies the number of slices to use.
     *
     * If this value is greater than 0, the library will use this fixed number
     * of slices for the identified slicing contraction.
     * If this value is 0 (the default), the library will automatically calculate
     * the minimum number of slices required to ensure the largest sliced
     * intermediate fits within 'memory_limit_bytes'.
     */
    int num_slices;

} hipTensorNetContractionOptimizerConfig_t;