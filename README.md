# rocQuantum
A quantum simulation library for AMD's ROCm, a counterpart to NVIDIA's CuQuantum.

## Features
- High-performance quantum state vector simulation (`hipStateVec`).
- Multi-GPU support for distributed simulation on a single node.
- Tensor network contraction (`hipTensorNet`).
- C++ API for integration into existing workflows.
- Python API (`rocq`) for ease of use.

## Building

### Dependencies
- ROCm
- rocBLAS
- RCCL

### Instructions

```
mkdir build
cd build
cmake ..
make
```

## Usage

### Tensor Network Contraction

```python
import rocq
import numpy as np

# Create a simulator
sim = rocq.Simulator()

# Create a tensor network
tn = rocq.TensorNetwork(simulator=sim)

# Create tensors
tensor_a = np.random.rand(2, 2).astype(np.complex64)
tensor_b = np.random.rand(2, 2).astype(np.complex64)

# Add tensors to the network
tn.add_tensor(tensor_a, ["a", "b"])
tn.add_tensor(tensor_b, ["b", "c"])

# Contract the network
result = tn.contract()

# Print the result
print(result)
```

### Multi-GPU State Vector Simulation

```python
import rocq
import numpy as np

# Create a simulator
sim = rocq.Simulator()

# Create a multi-GPU circuit
circuit = rocq.Circuit(num_qubits=3, simulator=sim, multi_gpu=True)

# Apply some gates
circuit.h(0)
circuit.cx(0, 1)

# Swap qubit 0 and 2
circuit.swap(0, 2)

# Measure qubit 0
outcome, prob = circuit.measure(0)

# Print the result
print(f"Measured outcome: {outcome} with probability {prob}")
```

## Running the Examples

To run the examples, first build the `rocQuantum` library. Then, you can run the example scripts from the `examples` directory:

```
cd examples
python tensornet_example.py
python multi_gpu_swap_example.py
```