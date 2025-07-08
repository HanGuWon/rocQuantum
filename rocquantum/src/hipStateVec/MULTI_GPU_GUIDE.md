# hipStateVec: Multi-GPU Support Guide

**IMPORTANT NOTE: Multi-GPU support in `hipStateVec` is under active development. While foundational components are in place and some functionalities have been implemented, certain complex operations or edge cases might still be evolving or marked as `ROCQ_STATUS_NOT_IMPLEMENTED`.**

## Overview

`hipStateVec` supports distributed quantum state vector simulations across multiple AMD GPUs on a single node. This allows for larger qubit systems to be simulated by leveraging the combined memory and compute power of available GPUs. The distribution strategy relies on bit-slicing, and AMD's RCCL library is used for inter-GPU communication.

## Multi-GPU Architecture

### Handle (`rocsvHandle_t`)
The `rocsvHandle_t` has been refactored to manage resources for multiple GPUs. A single handle now oversees:
- A list of HIP device IDs.
- Per-GPU HIP streams and rocBLAS handles.
- Per-GPU RCCL communicators (`rcclComm_t`), initialized via `rcclCommInitRank`.
- Pointers to per-GPU device memory slices for the state vector (`d_local_state_slices`).
- Temporary swap buffers (`d_swap_buffers`) per GPU, used for data exchange operations like `rocsvSwapIndexBits`.

### Data Distribution: Bit-Slicing
The state vector is distributed using a bit-slicing technique:
- For a system with `N` global qubits and `P` GPUs (where `P` must be a power of 2), `M = log2(P)` qubits are designated as "slice-determining" or "global slice" qubits. These are typically the most significant qubits in the global indexing scheme.
- The remaining `L = N - M` qubits are "local" to each GPU slice. These are typically the least significant qubits.
- The state of the `M` slice-determining qubits dictates which GPU (rank) holds a particular part of the state vector. For example, if qubits `q_N-1, ..., q_L` are slice qubits, their combined state `|s_M-1 ... s_0>` forms an integer `S` that maps to GPU rank `S`.
- Each GPU `S` stores `2^L` amplitudes, corresponding to all possible states of the `L` local qubits for its designated slice configuration.

### RCCL Integration
RCCL (AMD's Communications Collective Library) is used for inter-GPU data transfers. The `rocsvSwapIndexBits` function relies on `rcclAlltoallv` for efficient data redistribution when swapping a local-domain bit with a slice-domain bit. The build system is configured to find and link against RCCL.

## API Functions for Multi-GPU

### Creation and Destruction
- **`rocqStatus_t rocsvCreate(rocsvHandle_t* handle);`**
  Initializes a handle for multi-GPU operations. Detects all available GPUs, sets up per-GPU streams, BLAS handles, and RCCL communicators.
- **`rocqStatus_t rocsvDestroy(rocsvHandle_t handle);`**
  Releases all multi-GPU resources, including RCCL communicators, device memory for state slices, and swap buffers.

### Distributed State Management
- **`rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle, unsigned totalNumQubits);`**
  Allocates memory for the state vector, distributed across all GPUs managed by the handle. Calculates slice sizes based on `totalNumQubits` and the number of GPUs. Also allocates necessary temporary swap buffers (`d_swap_buffers`) for each GPU.
- **`rocqStatus_t rocsvInitializeDistributedState(rocsvHandle_t handle);`**
  Initializes the distributed state vector to the |0...0> state. The amplitude for the |0...0> state (global index 0) resides on GPU 0's slice at local index 0.

### Index-Bit Swap (Communication)
- **`rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle, unsigned qubit_idx1, unsigned qubit_idx2);`**
  This function is crucial for enabling gates that span across GPU data slices by rearranging data.
    - **Local-Local Swap:** If both `qubit_idx1` and `qubit_idx2` are within the local domain of each GPU (i.e., `idx < numLocalQubitsPerGpu`), this function performs an efficient in-place permutation on each GPU slice using the `local_bit_swap_permutation_kernel` and the per-GPU swap buffer as temporary space. This part is implemented.
    - **Local-Slice Swap (RCCL Communication):** If one qubit is in the local domain and the other is a slice-determining bit, this function performs a multi-stage data redistribution:
        1.  **Data Preparation:** Each GPU launches `calculate_swap_counts_kernel` to determine how many of its local amplitudes need to be sent to every other GPU. These counts are gathered on the host. Then, `shuffle_data_for_swap_kernel` is launched on each GPU to pack its data into its `d_swap_buffer` according to destination ranks and calculated displacements (derived from the counts). This stage is implemented.
        2.  **RCCL Alltoallv:** The `rcclAlltoallv` collective is used to exchange the packed data from each GPU's `d_swap_buffer` to the `d_local_state_slices` of the target GPUs. This stage is implemented.
    - **Slice-Slice Swap:** Swapping two slice-determining bits is currently **`ROCQ_STATUS_NOT_IMPLEMENTED`** as it requires a more complex rank re-mapping strategy beyond a simple Alltoallv of current slice contents.
  **Status:** The local-local swap and the local-slice swap (including data prep and RCCL communication) are now implemented.

### Gate Application
Gate application functions have been adapted for multi-GPU operation:
- **Local Gates:** If all target qubits of a gate are "local" to each GPU's slice (i.e., their global indices are less than `h->numLocalQubitsPerGpu`), the gate is applied in parallel on each GPU's local data slice without communication.
    - All specific single-qubit gates (`rocsvApplyX`, `Y`, `Z`, `H`, `S`, `T`, `Rx`, `Ry`, `Rz`) and two-qubit gates (`rocsvApplyCNOT`, `CZ`, `SWAP`) have been refactored to support this local multi-GPU execution.
    - `rocsvApplyMatrix` also supports this for local targets.
- **Global Gates:** If a gate targets one or more non-local (slice-determining) qubits, it would conceptually require calls to `rocsvSwapIndexBits` to make the target qubits local to a processing unit (which might be a single GPU or a pair for 2-qubit gates after swaps), apply the gate locally, and then swap back.
    - Currently, direct application of gates involving slice-determining bits (where `are_qubits_local` returns false) will result in `ROCQ_STATUS_NOT_IMPLEMENTED` from the gate application functions themselves. The user or a higher-level library is responsible for orchestrating `rocsvSwapIndexBits` calls around local gate applications to simulate global gates.

### Measurement (`rocsvMeasure`)
- **`rocqStatus_t rocsvMeasure(rocsvHandle_t handle, rocComplex* d_state, unsigned numQubits, unsigned qubitToMeasure, int* h_outcome, double* h_probability);`**
  This function has been refactored to support multi-GPU measurement **if `qubitToMeasure` is a local-domain bit** (i.e., `qubitToMeasure < h->numLocalQubitsPerGpu`).
    - **Probability Calculation:** New kernels (`calculate_local_slice_probabilities_kernel`) are used on each GPU to calculate partial probabilities for measuring '0' and '1' for the target qubit within its slice. These kernels use block-level reduction with shared memory. The partial probabilities from all blocks on a GPU are then further summed (currently on host after D2H copy of block sums) to get per-GPU slice probabilities. These are then summed across GPUs on the host to determine the global measurement outcome.
    - **State Collapse:** The `collapse_state_kernel` (which is element-wise) is applied on each GPU slice based on the global outcome and the local index of `qubitToMeasure`.
    - **Renormalization:** New kernels (`calculate_local_slice_sum_sq_mag_kernel`) calculate the sum of squared magnitudes on each slice post-collapse (also using block-level reduction). These are aggregated on the host to find the global normalization factor, which is then applied via `renormalize_state_kernel` (element-wise) on each slice.
    - **Single-GPU Path:** The single-GPU path within `rocsvMeasure` has also been updated to use these new, more robust reduction kernels.
    - **Limitation:** Measuring a slice-determining qubit directly (i.e. `qubitToMeasure >= h->numLocalQubitsPerGpu` in a multi-GPU context) is `ROCQ_STATUS_NOT_IMPLEMENTED`. It would require `rocsvSwapIndexBits` to make that qubit local first. The `d_state` parameter is legacy for multi-GPU mode; the function uses the handle's distributed state.

## Gate Fusion
- **`rocqStatus_t rocsvApplyFusedSingleQubitMatrix(rocsvHandle_t handle, unsigned targetQubit, const rocComplex* d_fusedMatrix);`**
  Applies a pre-fused 2x2 unitary matrix to the `targetQubit`.
  - Works for local `targetQubit` applications across multiple GPUs.
  - For global `targetQubit`, it will return `ROCQ_STATUS_NOT_IMPLEMENTED` (requires `rocsvSwapIndexBits`).

## Building with RCCL
The `CMakeLists.txt` for `hipStateVec` is configured to find and link against the RCCL library.

## Current Status Summary
- Multi-GPU handle and distributed state allocation/initialization are functional.
- **Local gate operations** (where all target qubits are within each GPU's local domain) are functional for all specific named gates and generic `rocsvApplyMatrix`.
- **`rocsvSwapIndexBits`** is functional for:
    - Swapping two local-domain bits (local permutation on each GPU).
    - Swapping one local-domain bit with one slice-determining bit (uses kernel-based data prep and `rcclAlltoallv`).
    - Swapping two slice-domain bits is `ROCQ_STATUS_NOT_IMPLEMENTED`.
- **`rocsvMeasure`** is functional for multi-GPU when measuring a local-domain qubit. The single-GPU path also uses improved reduction kernels. Measuring slice-domain qubits directly is `ROCQ_STATUS_NOT_IMPLEMENTED`.
- **Global gate operations** (targeting slice-determining qubits) require manual orchestration of `rocsvSwapIndexBits` and local gate applications by the user/higher-level library. Direct calls to gate functions with non-local targets will return `ROCQ_STATUS_NOT_IMPLEMENTED`.

The primary path for enabling arbitrary global gates is now through the implemented `rocsvSwapIndexBits` function. Further work may involve integrating `rocsvSwapIndexBits` calls directly within gate application functions for convenience or optimizing the host-side aggregations in `rocsvMeasure` with RCCL collectives.
