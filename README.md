# rocQuantum
A quantum simulation library for AMD's ROCm, a counterpart to NVIDIA's CuQuantum.

## Features
- High-performance quantum state vector simulation.
- Multi-GPU support for distributed simulation on a single node.
- C++ API for integration into existing workflows.

## Multi-GPU Support

The `hipStateVec` module within rocQuantum has been enhanced with foundational multi-GPU support. The architecture allows for distributing the quantum state vector across multiple AMD GPUs on a single node, leveraging bit-slicing and AMD's RCCL library. Key data structures, resource management (`rocsvHandle_t`), distributed state allocation (`rocsvAllocateDistributedState`), and initialization (`rocsvInitializeDistributedState`) are in place.

However, the multi-GPU functionality is **PARTIALLY IMPLEMENTED AND CURRENTLY BLOCKED** from full completion due to persistent tooling issues that have prevented necessary code modifications in core C++ files.

**Current Capabilities:**
- Distributed state vector representation.
- Application of a subset of gates (`rocsvApplyMatrix` for local targets, `rocsvApplyX`, `rocsvApplyRx`, `rocsvApplyCNOT`, and `rocsvApplyFusedSingleQubitMatrix` for local targets) *only when all target qubits are local to each GPU's data slice*.

**Critical Limitations:**
- The core data exchange function, `rocsvSwapIndexBits`, which is essential for applying gates to non-local ("global") qubits, is incomplete. The RCCL communication stage could not be implemented.
- Consequently, global gate operations are non-functional.
- Refactoring of many specific gate functions (e.g., `rocsvApplyY`, `Z`, `H`, etc.) for multi-GPU local operation is also incomplete due to the same tooling issues.

For detailed information on the intended multi-GPU architecture, implemented API functions, and the specific limitations, please see the [Multi-GPU Support Guide](./rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md).

## Building
(Details about building the library would go here - e.g., CMake instructions, dependencies like ROCm, rocBLAS, RCCL)

## Usage
(Basic usage examples would go here)
