#include "rocquantum/rocWorkspaceManager.h"
#include "rocquantum/hipStateVec.h" // For checkHipError

namespace rocquantum {
namespace util {

WorkspaceManager::WorkspaceManager(size_t initial_size_bytes, hipStream_t stream)
    : d_workspace_ptr_(nullptr),
      total_size_bytes_(initial_size_bytes),
      current_offset_bytes_(0),
      stream_(stream) {
    if (total_size_bytes_ > 0) {
        hipError_t err = hipMalloc(&d_workspace_ptr_, total_size_bytes_);
        if (err != hipSuccess) {
            // In a real scenario, might throw or log. For now, ptr remains null.
            // checkHipError(err, "WorkspaceManager hipMalloc"); // Assuming checkHipError exists and handles logging/throwing
            d_workspace_ptr_ = nullptr; // Ensure it's null on failure
            total_size_bytes_ = 0;      // and size is 0
            // Consider throwing std::runtime_error here
            throw std::runtime_error("WorkspaceManager: hipMalloc failed to allocate workspace of size " + std::to_string(initial_size_bytes));
        }
    }
}

WorkspaceManager::~WorkspaceManager() {
    if (d_workspace_ptr_) {
        hipFree(d_workspace_ptr_);
        d_workspace_ptr_ = nullptr;
    }
}

rocComplex* WorkspaceManager::allocate(size_t num_elements) {
    if (!d_workspace_ptr_ || num_elements == 0) {
        return nullptr;
    }

    size_t requested_bytes = num_elements * sizeof(rocComplex);
    if (requested_bytes == 0) return nullptr; // Should not happen if num_elements > 0

    // Align the current offset
    size_t aligned_offset = current_offset_bytes_;
    if (alignment_ > 0 && (current_offset_bytes_ % alignment_) != 0) {
        aligned_offset = ((current_offset_bytes_ + alignment_ - 1) / alignment_) * alignment_;
    }

    if (aligned_offset + requested_bytes <= total_size_bytes_) {
        rocComplex* ptr = reinterpret_cast<rocComplex*>(
            reinterpret_cast<char*>(d_workspace_ptr_) + aligned_offset
        );
        current_offset_bytes_ = aligned_offset + requested_bytes;
        return ptr;
    } else {
        // Not enough space with simple bump allocator
        // A more sophisticated manager might try to grow the pool or find free blocks.
        return nullptr;
    }
}

void WorkspaceManager::reset() {
    current_offset_bytes_ = 0;
}

size_t WorkspaceManager::get_total_size_bytes() const {
    return total_size_bytes_;
}

size_t WorkspaceManager::get_used_size_bytes() const {
    // Return the aligned offset as used, as that's the next allocation start
    size_t aligned_offset = current_offset_bytes_;
     if (alignment_ > 0 && (current_offset_bytes_ % alignment_) != 0) {
        aligned_offset = ((current_offset_bytes_ + alignment_ - 1) / alignment_) * alignment_;
    }
    return aligned_offset; // Or current_offset_bytes_ if strict usage without alignment padding is preferred for "used"
}

hipStream_t WorkspaceManager::get_stream() const {
    return stream_;
}

} // namespace util
} // namespace rocquantum
