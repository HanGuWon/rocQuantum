# rocQuantum
A quantum simulation library for AMD's ROCm, a counterpart to NVIDIA's CuQuantum.

## Features
- High-performance quantum state vector simulation (`hipStateVec`).
- Multi-GPU support for distributed simulation on a single node.
- C++ API for integration into existing workflows.
- Python API (`rocq`) for ease of use.

## Multi-GPU Support

The `hipStateVec` module within rocQuantum now features significantly enhanced multi-GPU support for distributed quantum state simulation on a single node. This leverages bit-slicing of the state vector across multiple AMD GPUs and uses the RCCL library for inter-GPU communication.

**Key Implemented Multi-GPU Features:**
- **Distributed State Management:**
    - `rocsvHandle_t` correctly manages resources (streams, BLAS handles, RCCL communicators, state slices, swap buffers) for all available GPUs.
    - `rocsvAllocateDistributedState` and `rocsvInitializeDistributedState` properly allocate and initialize the state vector across GPUs.
- **Local Gate Operations:**
    - All specific single-qubit gates (X, Y, Z, H, S, T, Rx, Ry, Rz) and two-qubit gates (CNOT, CZ, SWAP) are now fully functional for multi-GPU execution when all target qubits are within the "local domain" of each GPU's data slice (i.e., no communication is required between GPUs for these operations).
    - `rocsvApplyMatrix` (for applying generic matrices) and `rocsvApplyFusedSingleQubitMatrix` also support these local multi-GPU operations.
- **Index-Bit Swapping (`rocsvSwapIndexBits`):**
    - This critical function for data redistribution is now largely implemented:
        - Swapping two local-domain bits: Performs efficient local permutations on each GPU.
        - Swapping one local-domain bit and one slice-determining bit: Implemented using HIP kernels for data preparation (`calculate_swap_counts_kernel`, `shuffle_data_for_swap_kernel`) and `rcclAlltoallv` for the actual data exchange between GPUs.
    - Swapping two slice-determining bits remains `ROCQ_STATUS_NOT_IMPLEMENTED` (conceptual plan documented in the C++ source code).
- **Measurement (`rocsvMeasure`):**
    - Supports multi-GPU measurement when the qubit being measured is a "local-domain" bit for each slice.
    - Utilizes new HIP kernels with two-stage local reductions on each GPU (`calculate_local_slice_probabilities_kernel` followed by `reduce_block_sums_to_slice_total_probs_kernel`, and similarly for sum-of-squares for renormalization).
    - Global aggregation of probabilities and sum-of-squares (for renormalization) is performed efficiently using `rcclAllReduce`.
    - The single-GPU path also benefits from these improved local reduction kernels.
    - Direct measurement of a slice-determining bit in multi-GPU mode is `ROCQ_STATUS_NOT_IMPLEMENTED` (requires swaps first).
- **Python API (`rocq.Circuit`):**
    - Now supports a `multi_gpu=True` flag during initialization to enable distributed state simulation.

**Path to Global Gate Operations:**
With the implementation of `rocsvSwapIndexBits` for local-slice swaps, applying gates to non-local ("global") qubits is now achievable by orchestrating calls to `rocsvSwapIndexBits` (to make target qubits local to a processing domain) around local gate applications. `rocsvApplyMatrix` includes conceptual comments for how such orchestration could be integrated. Direct application of gate functions to non-local targets will typically return `ROCQ_STATUS_NOT_IMPLEMENTED`.

For more detailed information on the multi-GPU architecture, specific API functions, and current capabilities, please refer to the [Multi-GPU Support Guide](./rocquantum/src/hipStateVec/MULTI_GPU_GUIDE.md).

## Building
(Details about building the library would go here - e.g., CMake instructions, dependencies like ROCm, rocBLAS, RCCL)
```
mkdir build
cd build
cmake ..
make
```

## Usage
(Basic usage examples would go here)
```python
import rocq
import numpy as np

# Initialize simulator (can be reused for multiple circuits)
sim = rocq.Simulator()

# Create a single-GPU circuit
circuit_single = rocq.Circuit(num_qubits=3, simulator=sim)
circuit_single.h(0)
circuit_single.cx(0, 1)
outcome, prob = circuit_single.measure(0)
print(f"Single-GPU: Q0 measured: {outcome} with prob {prob}")

# Create a multi-GPU circuit (if multiple GPUs are available and configured)
# Note: Ensure enough qubits for meaningful distribution, e.g., > log2(num_gpus)
try:
    # Assuming 2 GPUs, 3 qubits: 1 slice bit, 2 local bits.
    # Qubit 0 and 1 are local to each slice.
    circuit_multi = rocq.Circuit(num_qubits=3, simulator=sim, multi_gpu=True)
    circuit_multi.h(0) # Local operation
    circuit_multi.cx(0, 1) # Local operation
    outcome_multi, prob_multi = circuit_multi.measure(0) # Measure local qubit
    print(f"Multi-GPU: Q0 measured: {outcome_multi} with prob {prob_multi}")
except RuntimeError as e:
    print(f"Multi-GPU example failed (possibly due to setup or NOT_IMPLEMENTED paths): {e}")

```
