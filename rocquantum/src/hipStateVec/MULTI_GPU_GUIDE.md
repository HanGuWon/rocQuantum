# hipStateVec: Multi-GPU Support Guide

This document outlines the multi-GPU capabilities, architecture, and usage of the `hipStateVec` module.

## Overview

`hipStateVec` has been enhanced to support distributed quantum state vector simulations across multiple AMD GPUs on a single node. This allows for larger qubit systems to be simulated by leveraging the combined memory and compute power of available GPUs. The distribution strategy relies on bit-slicing and AMD's RCCL library for inter-GPU communication.

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
RCCL (AMD's Communications Collective Library) is used for all inter-GPU data transfers. Key operations like the index-bit swap (see below) rely on RCCL collectives (e.g., `rcclAlltoallv`). The build system must be configured to find and link against RCCL.

## API Functions for Multi-GPU

### Creation and Destruction
- **`rocqStatus_t rocsvCreate(rocsvHandle_t* handle);`**
  Initializes a handle for multi-GPU operations. Detects all available GPUs, sets up per-GPU streams, BLAS handles, and RCCL communicators.
- **`rocqStatus_t rocsvDestroy(rocsvHandle_t handle);`**
  Releases all multi-GPU resources, including RCCL communicators and device memory.

### Distributed State Management
- **`rocqStatus_t rocsvAllocateDistributedState(rocsvHandle_t handle, unsigned totalNumQubits);`**
  Allocates memory for the state vector, distributed across all GPUs managed by the handle. Calculates slice sizes based on `totalNumQubits` and the number of GPUs. Also allocates necessary temporary swap buffers.
- **`rocqStatus_t rocsvInitializeDistributedState(rocsvHandle_t handle);`**
  Initializes the distributed state vector to the |0...0> state. The amplitude for the |0...0> state resides on GPU 0.

### Index-Bit Swap (Communication)
- **`rocqStatus_t rocsvSwapIndexBits(rocsvHandle_t handle, unsigned qubit_idx1, unsigned qubit_idx2);`**
  **Note: This function is currently a stub and returns `ROCQ_STATUS_NOT_IMPLEMENTED`.**
  Its intended purpose is to swap the roles of two global qubit indices in the state vector's distributed representation. If one qubit is local and the other is a slice-determining qubit, this operation requires redistributing data across GPUs using RCCL (`Alltoallv`). This is essential for applying gates to non-local qubits by making them temporarily local.

### Gate Application
Gate application functions (`rocsvApplyMatrix`, `rocsvApplyX`, `rocsvApplyCNOT`, etc.) have been adapted:
- **Local Gates:** If all target qubits of a gate are currently "local" to each GPU's slice (i.e., they are not slice-determining qubits in the current data layout), the gate is applied in parallel on each GPU's local data slice without communication.
- **Global Gates:** If a gate targets non-local (slice-determining) qubits, it would conceptually require `rocsvSwapIndexBits` to make them local, then apply the gate, then swap back. **Currently, operations on global gates will return `ROCQ_STATUS_NOT_IMPLEMENTED` due to the status of `rocsvSwapIndexBits`.**

The following specific gate functions have been partially refactored for multi-GPU local operations: `rocsvApplyX`, `rocsvApplyRx`, `rocsvApplyCNOT`. Others may still use single-GPU logic or return `ROCQ_STATUS_NOT_IMPLEMENTED` for multi-GPU scenarios.

## Gate Fusion

- **`rocqStatus_t rocsvApplyFusedSingleQubitMatrix(rocsvHandle_t handle, unsigned targetQubit, const rocComplex* d_fusedMatrix);`**
  Applies a pre-fused 2x2 unitary matrix (provided in device memory) to the `targetQubit`.
  - The fusion of multiple single-qubit gates into this single matrix (e.g., Rz(c)Rx(b)Rz(a)) is expected to be performed by the caller (CPU-side).
  - This function follows the same local/global logic as other gate applications. It works for local `targetQubit` applications in the current multi-GPU setup.

## Building with RCCL
The `CMakeLists.txt` for `hipStateVec` has been updated to find the RCCL package and link against it. Ensure RCCL is installed in a location discoverable by CMake.

## Current Limitations
- **`rocsvSwapIndexBits`:** The core data exchange routine is not implemented. This prevents true multi-GPU operation for gates acting on non-local qubits.
- **Specific Gate Refactoring:** Due to development tool issues, not all specific gate functions (e.g., `rocsvApplyY`, `rocsvApplyZ`, etc.) have been fully refactored for multi-GPU local operation. They may default to single-GPU behavior or return errors in a multi-GPU context.
- **Testing:** Tests cover multi-GPU allocation, initialization, and local gate application for X, CNOT, and fused gates. Global gate application testing is blocked by `rocsvSwapIndexBits`.
