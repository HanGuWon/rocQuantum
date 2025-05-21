#include "rocquantum/hipStateVec.h"
#include <iostream>
#include <vector>
#include <complex> // For std::complex
#include <cassert>
#include <cmath>   // For fabs, sqrt, log2
#include <numeric> // For std::iota
#include <algorithm> // For std::reverse

// Test-visible internal handle structure.
// WARNING: This is a HACK and relies on the internal structure of rocsvInternalHandle
// defined in hipStateVec.cpp NOT CHANGING. This makes tests fragile.
struct rocsvInternalHandle_TestVisible {
    int numGpus;
    unsigned globalNumQubits;
    unsigned numLocalQubitsPerGpu;
    unsigned numGlobalSliceQubits;
    std::vector<int> deviceIds;
    std::vector<hipStream_t> streams;
    std::vector<rocblas_handle> blasHandles;
    std::vector<rcclComm_t> comms;
    std::vector<rocComplex*> d_local_state_slices;
    std::vector<size_t> localStateSizes;
    std::vector<rocComplex*> d_swap_buffers; // This was confirmed to be in the main struct
};

// Helper to compare complex numbers
bool compareComplex(rocComplex a, rocComplex b, double tol = 1e-6) {
    return (fabs(a.x - b.x) < tol) && (fabs(a.y - b.y) < tol);
}

// Helper to copy a local slice to host for verification
std::vector<rocComplex> getLocalSliceData(rocsvHandle_t handle, int gpuRankToInspect) {
    if (!handle) {
        std::cerr << "getLocalSliceData: Invalid handle provided." << std::endl;
        return {};
    }
    rocsvInternalHandle_TestVisible* h_tv = reinterpret_cast<rocsvInternalHandle_TestVisible*>(handle);

    if (gpuRankToInspect < 0 || gpuRankToInspect >= h_tv->numGpus) {
        std::cerr << "getLocalSliceData: gpuRankToInspect " << gpuRankToInspect 
                  << " is out of bounds for numGpus " << h_tv->numGpus << std::endl;
        return {};
    }
     if (static_cast<size_t>(gpuRankToInspect) >= h_tv->deviceIds.size() ||
        static_cast<size_t>(gpuRankToInspect) >= h_tv->d_local_state_slices.size() ||
        static_cast<size_t>(gpuRankToInspect) >= h_tv->localStateSizes.size() ) {
        std::cerr << "getLocalSliceData: Handle vectors not properly sized for rank " << gpuRankToInspect << std::endl;
        return {};
    }
    if (!h_tv->d_local_state_slices[gpuRankToInspect]) {
        std::cerr << "getLocalSliceData: Slice pointer for rank " 
                  << gpuRankToInspect << " is null." << std::endl;
        // Return empty if slice pointer is null, unless size is also 0.
        if (h_tv->localStateSizes[gpuRankToInspect] > 0) return {}; 
    }

    size_t slice_size_elements = h_tv->localStateSizes[gpuRankToInspect];
    if (slice_size_elements == 0) {
        return {}; 
    }

    std::vector<rocComplex> host_slice(slice_size_elements);
    
    hipError_t err_set_dev = hipSetDevice(h_tv->deviceIds[gpuRankToInspect]);
    if (err_set_dev != hipSuccess) {
        std::cerr << "getLocalSliceData: hipSetDevice to " << h_tv->deviceIds[gpuRankToInspect] 
                  << " failed: " << hipGetErrorString(err_set_dev) << std::endl;
        return {};
    }
    
    hipError_t err_memcpy = hipMemcpy(host_slice.data(), h_tv->d_local_state_slices[gpuRankToInspect], 
                                   slice_size_elements * sizeof(rocComplex), hipMemcpyDeviceToHost);
    if (err_memcpy != hipSuccess) {
        std::cerr << "getLocalSliceData: HIP memcpy DtoH error on GPU " << h_tv->deviceIds[gpuRankToInspect] 
                  << " for slice " << gpuRankToInspect << ": " << hipGetErrorString(err_memcpy) << std::endl;
        return {};
    }
    return host_slice;
}

// Test 1: Multi-GPU Allocation and Initialization
void test_allocation_and_initialization(int minRequiredGpusForTest) {
    std::cout << "Starting Test 1: Multi-GPU Allocation and Initialization..." << std::endl;
    rocsvHandle_t handle;
    rocqStatus_t status = rocsvCreate(&handle);
    assert(status == ROCQ_STATUS_SUCCESS && "Test 1: rocsvCreate failed");
    assert(handle != nullptr && "Test 1: Handle is null after create");

    rocsvInternalHandle_TestVisible* h_tv = reinterpret_cast<rocsvInternalHandle_TestVisible*>(handle);
    
    std::cout << "  GPUs detected by handle: " << h_tv->numGpus << std::endl;
    if (h_tv->numGpus < minRequiredGpusForTest && minRequiredGpusForTest > 0) { // Allow minRequiredGpusForTest = 0 for any number of GPUs
        std::cout << "  Skipping Test 1: Needs at least " << minRequiredGpusForTest << " GPUs in handle for this specific test configuration. Found " << h_tv->numGpus << std::endl;
        rocsvDestroy(handle);
        return;
    }
    
    // For this test, we specifically want to test a 2-GPU distribution if possible.
    // The number of GPUs used in calculations will be h_tv->numGpus from the created handle.
    int effective_num_gpus_for_test_logic = (minRequiredGpusForTest == 2 && h_tv->numGpus >=2) ? 2 : h_tv->numGpus;
    if ( (effective_num_gpus_for_test_logic > 0) && ((effective_num_gpus_for_test_logic & (effective_num_gpus_for_test_logic - 1)) != 0) && effective_num_gpus_for_test_logic != 1) {
         std::cout << "  Skipping Test 1: Test logic requires a power-of-2 number of GPUs (or 1) for distribution. Effective GPUs for test: " << effective_num_gpus_for_test_logic << std::endl;
        rocsvDestroy(handle);
        return;
    }


    unsigned totalNumQubits = 3; // Example: q2 q1 q0
    status = rocsvAllocateDistributedState(handle, totalNumQubits);
    assert(status == ROCQ_STATUS_SUCCESS && "Test 1: rocsvAllocateDistributedState failed");

    unsigned expectedNumGlobalSliceQubits = (h_tv->numGpus > 1) ? static_cast<unsigned>(std::log2(h_tv->numGpus)) : 0;
    unsigned expectedNumLocalQubitsPerGpu = totalNumQubits - expectedNumGlobalSliceQubits;
    size_t expectedLocalStateSize = (1ULL << expectedNumLocalQubitsPerGpu);
    if (totalNumQubits < expectedNumGlobalSliceQubits) { // Not enough qubits to distribute
        expectedLocalStateSize = 0; // Each slice would be empty.
        expectedNumLocalQubitsPerGpu = 0; // Correct this based on how the library handles it. Assuming 0 for now.
    }


    assert(h_tv->globalNumQubits == totalNumQubits);
    assert(h_tv->numGlobalSliceQubits == expectedNumGlobalSliceQubits);
    assert(h_tv->numLocalQubitsPerGpu == expectedNumLocalQubitsPerGpu);
    assert(h_tv->localStateSizes.size() == static_cast<size_t>(h_tv->numGpus));

    for (int i = 0; i < h_tv->numGpus; ++i) {
        assert(h_tv->localStateSizes[i] == expectedLocalStateSize && "Test 1: Local state size incorrect");
        if (expectedLocalStateSize > 0) {
            assert(h_tv->d_local_state_slices[i] != nullptr && "Test 1: Device slice pointer is null for non-empty slice");
        }
    }
    std::cout << "  Allocation parameters verified." << std::endl;

    status = rocsvInitializeDistributedState(handle);
    assert(status == ROCQ_STATUS_SUCCESS && "Test 1: rocsvInitializeDistributedState failed");

    for (int i = 0; i < h_tv->numGpus; ++i) {
        std::vector<rocComplex> slice_data = getLocalSliceData(handle, i);
        if (expectedLocalStateSize == 0) {
             assert(slice_data.empty() && "Test 1: Slice data should be empty");
             continue;
        }
        assert(slice_data.size() == expectedLocalStateSize && "Test 1: Slice data size mismatch");
        if (slice_data.empty() && expectedLocalStateSize > 0) { // Should have been caught by assert above
             std::cerr << "Test 1: Slice data empty for GPU " << i << " when it should not be." << std::endl; continue;
        }

        if (i == 0) { // GPU 0 (rank 0)
            assert(compareComplex(slice_data[0], {1.0f, 0.0f}) && "Test 1: GPU0 Slice[0] incorrect");
            for (size_t j = 1; j < slice_data.size(); ++j) {
                assert(compareComplex(slice_data[j], {0.0f, 0.0f}) && "Test 1: GPU0 Slice non-zero tail incorrect");
            }
        } else { // Other GPUs
            for (size_t j = 0; j < slice_data.size(); ++j) {
                assert(compareComplex(slice_data[j], {0.0f, 0.0f}) && "Test 1: Other GPU Slice non-zero element incorrect");
            }
        }
    }
    std::cout << "  Test 1 state verified." << std::endl;
    rocsvDestroy(handle);
    std::cout << "Test 1 Finished." << std::endl;
}

void test_local_apply_x(int requiredGpus) {
    std::cout << "Starting Test 2: Local rocsvApplyX..." << std::endl;
    rocsvHandle_t handle;
    rocqStatus_t status = rocsvCreate(&handle); assert(status == ROCQ_STATUS_SUCCESS);
    rocsvInternalHandle_TestVisible* h_tv = reinterpret_cast<rocsvInternalHandle_TestVisible*>(handle);
    if (h_tv->numGpus < requiredGpus || ((h_tv->numGpus > 0) && ((h_tv->numGpus & (h_tv->numGpus - 1)) != 0) && h_tv->numGpus != 1) ) { // Power of 2 or 1 GPU
        std::cout << "  Skipping Test 2: Requires " << requiredGpus << " (power of 2, or 1) GPUs for this test logic. Found " << h_tv->numGpus << std::endl;
        rocsvDestroy(handle); return;
    }
    if (h_tv->numGpus == 0) { rocsvDestroy(handle); return; } // Cannot run test with 0 GPUs

    unsigned totalNumQubits = 3; // q2 q1 q0
    unsigned targetQubit = 0;    // X on q0 (global index)
    
    status = rocsvAllocateDistributedState(handle, totalNumQubits); assert(status == ROCQ_STATUS_SUCCESS);
    status = rocsvInitializeDistributedState(handle); assert(status == ROCQ_STATUS_SUCCESS);

    status = rocsvApplyX(handle, nullptr, totalNumQubits, targetQubit);
    assert(status == ROCQ_STATUS_SUCCESS && "Test 2: rocsvApplyX failed");

    // Expected: |...01> (q0 flipped)
    // If numGpus = 2 (numGlobalSliceQubits = 1, numLocalQubitsPerGpu = 2):
    //   GPU 0 (slice q2=0): local state |01> (q1,q0). Index 1 should be 1.0.
    //   GPU 1 (slice q2=1): local state |00>. Index 0 should be 0.0 (no change from init).
    unsigned numSliceQubits = h_tv->numGlobalSliceQubits;
    unsigned numLocalQubits = h_tv->numLocalQubitsPerGpu;
    size_t localStateSize = (1ULL << numLocalQubits);

    for (int i = 0; i < h_tv->numGpus; ++i) {
        std::vector<rocComplex> slice_data = getLocalSliceData(handle, i);
        assert(slice_data.size() == localStateSize);
        
        // Determine if this slice (rank i) should contain the |1> state for the target qubit
        // This depends on how targetQubit maps into slice vs local bits.
        // Here, targetQubit (0) is local, so it affects index within each slice.
        // The |00...0> state becomes |00...1> (targetQubit 0 flips).
        // This means global index 1 is 1.0.
        // Global index `g = rank * localStateSize + local_idx`.
        // We expect state |1> = global index 1.
        // If rank_for_state_1 == i, then slice_data[local_idx_for_state_1] == 1.0
        
        int rank_for_state_1 = (1 >> numLocalQubits); // Rank where global state |1> resides
        size_t local_idx_for_state_1 = 1 & ((1ULL << numLocalQubits) - 1);

        if (i == rank_for_state_1) {
            for (size_t j = 0; j < slice_data.size(); ++j) {
                if (j == local_idx_for_state_1) {
                    assert(compareComplex(slice_data[j], {1.0f, 0.0f}) && "Test 2: Target slice element incorrect");
                } else {
                    assert(compareComplex(slice_data[j], {0.0f, 0.0f}) && "Test 2: Target slice non-target incorrect");
                }
            }
        } else { // Other GPU slices should remain all zero
            for (const auto& val : slice_data) {
                assert(compareComplex(val, {0.0f, 0.0f}) && "Test 2: Other slice incorrect");
            }
        }
    }
    std::cout << "  Test 2 state verified." << std::endl;
    rocsvDestroy(handle);
    std::cout << "Test 2 Finished." << std::endl;
}

void test_local_apply_cnot(int requiredGpus) {
    std::cout << "Starting Test 3: Local rocsvApplyCNOT..." << std::endl;
    rocsvHandle_t handle;
    rocqStatus_t status = rocsvCreate(&handle); assert(status == ROCQ_STATUS_SUCCESS);
    rocsvInternalHandle_TestVisible* h_tv = reinterpret_cast<rocsvInternalHandle_TestVisible*>(handle);
    if (h_tv->numGpus < requiredGpus || ((h_tv->numGpus > 0) && ((h_tv->numGpus & (h_tv->numGpus - 1)) != 0) && h_tv->numGpus != 1) ) {
        std::cout << "  Skipping Test 3: Requires " << requiredGpus << " (power of 2, or 1) GPUs. Found " << h_tv->numGpus << std::endl;
        rocsvDestroy(handle); return;
    }
    if (h_tv->numGpus == 0) { rocsvDestroy(handle); return; }

    unsigned totalNumQubits = 3; 
    unsigned controlQubit = 0; // global index
    unsigned targetQubit = 1;  // global index
    status = rocsvAllocateDistributedState(handle, totalNumQubits); assert(status == ROCQ_STATUS_SUCCESS);
    status = rocsvInitializeDistributedState(handle); assert(status == ROCQ_STATUS_SUCCESS);
    
    status = rocsvApplyX(handle, nullptr, totalNumQubits, controlQubit); // State is |...01> (q0=1)
    assert(status == ROCQ_STATUS_SUCCESS);

    status = rocsvApplyCNOT(handle, nullptr, totalNumQubits, controlQubit, targetQubit);
    assert(status == ROCQ_STATUS_SUCCESS && "Test 3: rocsvApplyCNOT failed");

    // Expected: CNOT(0,1) on |...01> -> |...11> (q1 flipped because q0=1)
    // Global state is 2^targetQubit | 2^controlQubit = 2^1 | 2^0 = 3. So global index 3 should be 1.0.
    unsigned numSliceQubits = h_tv->numGlobalSliceQubits;
    unsigned numLocalQubits = h_tv->numLocalQubitsPerGpu;
    size_t localStateSize = 1ULL << numLocalQubits;
    size_t final_state_global_idx = (1ULL << targetQubit) | (1ULL << controlQubit);

    int rank_for_final_state = (final_state_global_idx >> numLocalQubits);
    size_t local_idx_for_final_state = final_state_global_idx & ((1ULL << numLocalQubits) - 1);

    for (int i = 0; i < h_tv->numGpus; ++i) {
        std::vector<rocComplex> slice_data = getLocalSliceData(handle, i);
        assert(slice_data.size() == localStateSize);
        if (i == rank_for_final_state) {
            for (size_t j = 0; j < slice_data.size(); ++j) {
                if (j == local_idx_for_final_state) {
                    assert(compareComplex(slice_data[j], {1.0f, 0.0f}) && "Test 3: Target slice element incorrect");
                } else {
                    assert(compareComplex(slice_data[j], {0.0f, 0.0f}) && "Test 3: Target slice non-target incorrect");
                }
            }
        } else {
             for (const auto& val : slice_data) {
                assert(compareComplex(val, {0.0f, 0.0f}) && "Test 3: Other slice incorrect");
            }
        }
    }
    std::cout << "  Test 3 state verified." << std::endl;
    rocsvDestroy(handle);
    std::cout << "Test 3 Finished." << std::endl;
}

void test_local_fused_matrix(int requiredGpus) {
    std::cout << "Starting Test 4: Local rocsvApplyFusedSingleQubitMatrix..." << std::endl;
    rocsvHandle_t handle;
    rocqStatus_t status = rocsvCreate(&handle); assert(status == ROCQ_STATUS_SUCCESS);
    rocsvInternalHandle_TestVisible* h_tv = reinterpret_cast<rocsvInternalHandle_TestVisible*>(handle);
    if (h_tv->numGpus < requiredGpus || ((h_tv->numGpus > 0) && ((h_tv->numGpus & (h_tv->numGpus - 1)) != 0) && h_tv->numGpus != 1) ) {
        std::cout << "  Skipping Test 4: Requires " << requiredGpus << " (power of 2, or 1) GPUs. Found " << h_tv->numGpus << std::endl;
        rocsvDestroy(handle); return;
    }
     if (h_tv->numGpus == 0) { rocsvDestroy(handle); return; }


    unsigned totalNumQubits = 3;
    unsigned targetQubit = 0; // global index
    status = rocsvAllocateDistributedState(handle, totalNumQubits); assert(status == ROCQ_STATUS_SUCCESS);
    status = rocsvInitializeDistributedState(handle); assert(status == ROCQ_STATUS_SUCCESS);

    rocComplex h_H_matrix[4] = {{float(1/sqrt(2)),0}, {float(1/sqrt(2)),0}, {float(1/sqrt(2)),0}, {float(-1/sqrt(2)),0}};
    rocComplex* d_H_matrix;
    hipError_t hip_s = hipMalloc(&d_H_matrix, 4 * sizeof(rocComplex)); assert(hip_s == hipSuccess);
    hip_s = hipMemcpy(d_H_matrix, h_H_matrix, 4 * sizeof(rocComplex), hipMemcpyHostToDevice); assert(hip_s == hipSuccess);
    
    status = rocsvApplyFusedSingleQubitMatrix(handle, targetQubit, d_H_matrix);
    assert(status == ROCQ_STATUS_SUCCESS && "Test 4: rocsvApplyFusedSingleQubitMatrix failed");

    // Expected: H|...00> = 1/sqrt(2) * (|...00> + |...01>)
    // Global state |0> (idx 0) and |1> (idx 1) will have amp 1/sqrt(2)
    unsigned numSliceQubits = h_tv->numGlobalSliceQubits;
    unsigned numLocalQubits = h_tv->numLocalQubitsPerGpu;
    size_t localStateSize = 1ULL << numLocalQubits;
    float expected_amp_val = 1.0f/sqrt(2.0f);

    int rank_for_state_0 = (0 >> numLocalQubits);
    size_t local_idx_for_state_0 = 0 & ((1ULL << numLocalQubits) - 1);
    int rank_for_state_1 = (1 >> numLocalQubits);
    size_t local_idx_for_state_1 = 1 & ((1ULL << numLocalQubits) - 1);
    
    for (int i = 0; i < h_tv->numGpus; ++i) {
        std::vector<rocComplex> slice_data = getLocalSliceData(handle, i);
        assert(slice_data.size() == localStateSize);
        for (size_t j = 0; j < slice_data.size(); ++j) {
            bool should_be_H_amp = false;
            if (i == rank_for_state_0 && j == local_idx_for_state_0) should_be_H_amp = true;
            if (i == rank_for_state_1 && j == local_idx_for_state_1) should_be_H_amp = true;

            if (should_be_H_amp) {
                 assert(compareComplex(slice_data[j], {expected_amp_val, 0.0f}) && "Test 4: Hadamard amplitude incorrect");
            } else {
                 assert(compareComplex(slice_data[j], {0.0f, 0.0f}) && "Test 4: Zero amplitude incorrect");
            }
        }
    }
    std::cout << "  Test 4 state verified." << std::endl;
    hipFree(d_H_matrix);
    rocsvDestroy(handle);
    std::cout << "Test 4 Finished." << std::endl;
}

int main(int argc, char **argv) {
    int numGpusForTestLogic = 2; // Default to 2 for tests designed around 2-GPU distribution
    if (argc > 1) {
        try {
            int cli_gpus = std::stoi(argv[1]);
            if (cli_gpus > 0) numGpusForTestLogic = cli_gpus;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse GPU count argument '" << argv[1] << "'. Defaulting to " << numGpusForTestLogic << std::endl;
        }
    }
    std::cout << "Test harness: Specific test logic may assume " << numGpusForTestLogic << " GPUs for result verification." << std::endl;
    std::cout << "Note: rocquantum library's rocsvCreate currently uses all available GPUs reported by hipGetDeviceCount." << std::endl;
    std::cout << "Tests will attempt to run; assertions depend on the actual number of GPUs used by the handle vs test expectations." << std::endl;

    test_allocation_and_initialization(numGpusForTestLogic);
    test_local_apply_x(numGpusForTestLogic);          
    test_local_apply_cnot(numGpusForTestLogic);       
    test_local_fused_matrix(numGpusForTestLogic);     
    
    std::cout << "\nAll multi-GPU tests finished." << std::endl;
    return 0;
}
