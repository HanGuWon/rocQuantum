# rocQuantum
A quantum simulation library for AMD's ROCm, a counterpart to NVIDIA's CuQuantum.

## Features
- High-performance quantum state vector simulation.
- Multi-GPU support for distributed simulation on a single node.
- C++ API for integration into existing workflows.

## Multi-GPU Support
The `hipStateVec` module within rocQuantum provides multi-GPU capabilities, allowing for the simulation of larger qubit systems by distributing the state vector across multiple AMD GPUs. This feature leverages bit-slicing for data distribution and AMD's RCCL library for inter-GPU communication.

For detailed information on the multi-GPU architecture, API functions, and current limitations, please see the [Multi-GPU Support Guide](./rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md).

## Building
(Details about building the library would go here - e.g., CMake instructions, dependencies like ROCm, rocBLAS, RCCL)

## Usage
(Basic usage examples would go here)
