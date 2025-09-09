# PennyLane-rocq: PennyLane device for the rocQuantum Simulator

## Overview

This package provides a PennyLane device that uses the [rocQuantum-1](https://github.com/ROCm/rocQuantum-1) C++/HIP library for high-performance quantum circuit simulation on AMD GPUs. It acts as a direct backend for PennyLane, allowing users to seamlessly offload quantum computations to a powerful, GPU-accelerated statevector simulator.

## Features

- **High-performance statevector simulation**: Leverages the rocQuantum C++ backend for fast and efficient calculations.
- **Full PennyLane Integration**: Supports core PennyLane measurement types, including `qml.state()` and `qml.counts()`.
- **Hardware-like Sampling**: Simulates measurement shots to provide realistic count dictionaries.
- **Seamless Workflow**: Integrates with PennyLane's automatic differentiation and optimization capabilities.

## Installation

### Prerequisites

Before installing this package, you must have the `rocQuantum-1` library and its Python binding (`rocquantum_bind`) already built and installed. Please follow the instructions in the `rocQuantum-1` project repository.

### Installation Steps

Once the prerequisites are met, you can install this package from the root directory using pip:

```bash
pip install .
```

PennyLane will automatically discover the `rocquantum.qpu` device upon successful installation.

## Usage Example

The following example demonstrates how to use the `rocquantum.qpu` device to simulate a Bell state circuit, first retrieving the final statevector and then getting measurement counts.

```python
import pennylane as qml
from pennylane import numpy as np

# 1. Load the rocQuantum device for statevector simulation
dev_state = qml.device("rocquantum.qpu", wires=2)

@qml.qnode(dev_state)
def bell_state_circuit_state():
    """A QNode that creates a Bell state and returns the final statevector."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

# Run the QNode and print the statevector
final_state = bell_state_circuit_state()
print("Final State Vector:")
print(final_state)
print("-" * 20)

# 2. Load the rocQuantum device for measurement sampling
shots = 1000
dev_counts = qml.device("rocquantum.qpu", wires=2, shots=shots)

@qml.qnode(dev_counts)
def bell_state_circuit_counts():
    """A QNode that creates a Bell state and returns measurement counts."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts()

# Run the QNode and print the counts
counts = bell_state_circuit_counts()
print(f"Measurement Counts (for {shots} shots):")
print(counts)

# Expected output:
# Final State Vector:
# [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
# --------------------
# Measurement Counts (for 1000 shots):
# {'00': 498, '11': 502}  (Note: Counts will vary slightly on each run)
```
