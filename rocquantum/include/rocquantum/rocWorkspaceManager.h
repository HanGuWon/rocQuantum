#ifndef ROC_WORKSPACE_MANAGER_H
#define ROC_WORKSPACE_MANAGER_H

#include "rocquantum/hipStateVec.h" // For rocComplex, rocqStatus_t
#include <hip/hip_runtime.h>
#include <cstddef> // For size_t
#include <stdexcept> // For std::runtime_error

namespace rocquantum {
namespace util {

class WorkspaceManager {
public:
    /**
     * @brief Constructs a WorkspaceManager.
     * @param initial_size_bytes The initial size of the workspace to allocate on the device.
     * @param stream The HIP stream to associate with allocations if needed by underlying mechanisms
     *               (though basic bump allocator might not use it directly for allocation itself).
     *               Currently not used in this simple implementation.
     */
    WorkspaceManager(size_t initial_size_bytes, hipStream_t stream = 0);
    ~WorkspaceManager();

    // Disable copy constructor and assignment
    WorkspaceManager(const WorkspaceManager&) = delete;
    WorkspaceManager& operator=(const WorkspaceManager&) = delete;

    /**
     * @brief Allocates a block of memory from the workspace.
     * Uses a simple bump allocation strategy.
     * @param num_elements The number of rocComplex elements to allocate.
     * @return Device pointer to the allocated memory, or nullptr if allocation fails or not enough space.
     */
    rocComplex* allocate(size_t num_elements);

    /**
     * @brief Resets the workspace, making all its memory available again.
     * For the bump allocator, this simply resets the current offset.
     */
    void reset();

    /**
     * @brief Gets the total size of the workspace in bytes.
     */
    size_t get_total_size_bytes() const;

    /**
     * @brief Gets the currently used size of the workspace in bytes.
     */
    size_t get_used_size_bytes() const;

    /**
     * @brief Gets the underlying HIP stream (currently unused by allocator but stored).
     */
    hipStream_t get_stream() const;

private:
    rocComplex* d_workspace_ptr_ = nullptr;
    size_t total_size_bytes_ = 0;
    size_t current_offset_bytes_ = 0;
    hipStream_t stream_ = 0; // Associated stream, for future use or if allocators need it
    const size_t alignment_ = 256; // Common alignment, e.g., for texture memory or performance
};

} // namespace util
} // namespace rocquantum

#endif // ROC_WORKSPACE_MANAGER_H
