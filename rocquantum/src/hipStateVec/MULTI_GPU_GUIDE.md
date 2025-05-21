# hipStateVec: Multi-GPU Support Guide

**IMPORTANT NOTE: The multi-GPU support in `hipStateVec` is PARTIALLY IMPLEMENTED AND CURRENTLY BLOCKED from further development due to persistent tooling issues. Key functionalities, particularly inter-GPU data exchange for global gate operations, are INCOMPLETE. This guide describes the intended architecture and the current state.**

## Overview

`hipStateVec` has been enhanced with the goal of supporting distributed quantum state vector simulations across multiple AMD GPUs on a single node. This would allow for larger qubit systems to be simulated by leveraging the combined memory and compute power of available GPUs. The distribution strategy relies on bit-slicing and AMD's RCCL library for inter-GPU communication.

## Multi-GPU Architecture

### Handle (`rocsvHandle_t`)
The `rocsvHandle_t` has been refactored to manage resources for multiple GPUs. A single handle now oversees:
- A list of HIP device IDs.
- Per-GPU HIP streams and rocBLAS handles.
- Per-GPU RCCL communicators (`rcclComm_t`), initialized via `rcclCommInitRank`.
- Pointers to per-GPU device memory slices for the state vector (`d_local_state_slices`).
- Temporary swap buffers (`d_swap_buffers`) for data exchange operations.

### Data Distribution: Bit-Slicing
The state vector is distributed using a bit-slicing technique:
- For a system with `N` global qubits and `P` GPUs (where `P` must be a power of 2), `M = log2(P)` qubits are designated as "slice-determining" or "global slice" qubits.
- The remaining `L = N - M` qubits are "local" to each GPU slice.
- The state of the `M` slice-determining qubits dictates which GPU holds a particular part of the state vector. For example, if qubits `q_N-1, ..., q_L` are slice qubits, their combined state `|s_M-1 ... s_0>` forms an integer `S` that maps to GPU rank `S`.
- Each GPU `S` stores `2^L` amplitudes, corresponding to all possible states of the `L` local qubits for its designated slice configuration.

### RCCL Integration
RCCL (AMD's Communications Collective Library) is intended for all inter-GPU data transfers. Key operations like the index-bit swap (see below) would rely on RCCL collectives (e.g., `rcclAlltoallv`). The build system has been configured to find and link against RCCL.

## API Functions for Multi-GPU

### Creation and Destruction
- **`rocqStatus_t rocsvCreate(rocsvHandle_t* handle);`**
  Initializes a handle for multi-GPU operations. Detects all available GPUs, sets up per-GPU streams, BLAS handles, and RCCL communicators.
- **`rocqStatus_t rocsvDestroy(rocsvHandle_t handle);`**
  Releases all multi-GPU resources, including RCCL communicators and device memory.

### Distributed State Management
- **`rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle, unsigned totalNumQubits);`**
  Allocates memory for the state vector, distributed across all GPUs managed by the handle. Calculates slice sizes based on `totalNumQubits` and the number of GPUs. Also allocates necessary temporary swap buffers (`d_swap_buffers`).
- **`rocqStatus_t rocsvInitializeDistributedState(rocsvHandle_t handle);`**
  Initializes the distributed state vector to the |0...0> state. The amplitude for the |0...0> state resides on GPU 0.

### Index-Bit Swap (Communication)
- **`rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle, unsigned qubit_idx1, unsigned qubit_idx2);`**
  **CRITICAL LIMITATION: This function is INCOMPLETE.**
    - Stage 1 (data preparation): The logic for determining send/receive counts and shuffling data into temporary swap buffers using custom HIP kernels (`calculate_swap_counts_kernel`, `shuffle_data_for_swap_kernel`) **is implemented**. The local-only swap path using `local_bit_swap_permutation_kernel` is also implemented.
    - Stage 2 (RCCL Communication): The integration of `rcclAlltoallv` for actual data exchange between GPUs **could NOT be implemented** due to persistent tooling errors preventing modifications to the `hipStateVec.cpp` file.
  **As a result, this function currently does not perform any inter-GPU data exchange and returns `ROCQ_STATUS_NOT_IMPLEMENTED` for distributed swap cases (where one qubit is local and one is slice-determining).** This is a major blocker for global gate operations.

### Gate Application
Gate application functions have been partially adapted for multi-GPU operation:
- **Local Gates:** If all target qubits of a gate are "local" to each GPU's slice (i.e., their global indices are less than `h->numLocalQubitsPerGpu`), the gate is applied in parallel on each GPU's local data slice without communication.
    - Successfully adapted for local operations: `rocsvApplyMatrix` (for local cases), `rocsvApplyX`, `rocsvApplyRx`, `rocsvApplyCNOT`.
    - **Not Adapted:** The following specific gate functions were **not** successfully refactored for multi-GPU local operation due to the aforementioned tooling errors and remain as placeholders: `rocsvApplyY`, `rocsvApplyZ`, `rocsvApplyH`, `rocsvApplyS`, `rocsvApplyT`, `rocsvApplyRy`, `rocsvApplyRz`, `rocsvApplyCZ`, `rocsvApplySWAP`. These may default to prior single-GPU logic (if any existed for them in a multi-GPU context) or more likely return `ROCQ_STATUS_NOT_IMPLEMENTED`.
- **Global Gates:** If a gate targets non-local (slice-determining) qubits, it would conceptually require a functional `rocsvSwapIndexBits` to make them local, apply the gate, then swap back. **Currently, operations on global gates will return `ROCQ_STATUS_NOT_IMPLEMENTED` due to the incomplete `rocsvSwapIndexBits`.**

## Gate Fusion

- **`rocqStatus_t rocsvApplyFusedSingleQubitMatrix(rocsvHandle_t handle, unsigned targetQubit, const rocComplex* d_fusedMatrix);`**
  Applies a pre-fused 2x2 unitary matrix (provided in device memory) to the `targetQubit`.
  - The fusion of multiple single-qubit gates into this single matrix is expected to be performed by the caller.
  - This function works for local `targetQubit` applications. For global `targetQubit`, it will return `ROCQ_STATUS_NOT_IMPLEMENTED` due to the `rocsvSwapIndexBits` limitation.

## Building with RCCL
The `CMakeLists.txt` for `hipStateVec` has been updated to find the RCCL package and link against it.

## Current Limitations & Blockers

The multi-GPU support for `hipStateVec` is **severely limited and blocked from completion** due to the following:

1.  **Incomplete `rocsvSwapIndexBits`:**
    *   The core data exchange routine (`rcclAlltoallv`) **is not integrated** into `rocsvSwapIndexBits`.
    *   **Reason:** Persistent tooling errors ("diff did not apply") prevented the necessary C++ code modifications in `hipStateVec.cpp` to implement the RCCL communication stage.
    *   **Impact:** True multi-GPU operation for gates acting on non-local qubits (global gates) is not functional.

2.  **Incomplete Refactoring of Specific Gate Functions:**
    *   Many specific gate functions (e.g., `rocsvApplyY`, `rocsvApplyZ`, `rocsvApplyH`, `rocsvApplyS`, `rocsvApplyT`, `rocsvApplyRy`, `rocsvApplyRz`, `rocsvApplyCZ`, `rocsvApplySWAP`) were not successfully refactored for multi-GPU local operations.
    *   **Reason:** The same persistent tooling errors prevented modifications to `hipStateVec.cpp`.
    *   **Impact:** These gates may not function correctly or at all in a multi-GPU context, likely returning `ROCQ_STATUS_NOT_IMPLEMENTED`. Only `rocsvApplyMatrix`, `rocsvApplyX`, `rocsvApplyRx`, `rocsvApplyCNOT`, and `rocsvApplyFusedSingleQubitMatrix` have updated local path logic.

3.  **Overall Status:** As a result of these blockers, the multi-GPU capability is limited to:
    *   Distributed state vector representation across GPUs.
    *   Application of a subset of gates (`rocsvApplyMatrix`, `X`, `Rx`, `CNOT`, `FusedSingleQubitMatrix`) *only when all target qubits are local to each GPU slice*.
    *   Global gate operations are non-functional.

4.  **Testing:** Existing tests cover multi-GPU allocation, initialization, and local gate application for the successfully refactored gates. Testing for global gate operations is blocked.

Further progress on multi-GPU features is contingent on resolving the underlying tooling issues that prevent reliable code modification of `hipStateVec.cpp`.
