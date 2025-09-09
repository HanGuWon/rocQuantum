# Qiskit-rocQuantum-Provider

## Overview

This package provides a Qiskit Provider that allows users to run quantum circuits on the [rocQuantum-1](https://github.com/ROCm/rocQuantum-1) high-performance simulator. It follows the modern `BackendV2` provider plugin architecture, ensuring stability and compatibility with the latest Qiskit features. This provider enables Qiskit users to seamlessly accelerate their simulations using AMD GPUs.

## Features

- **Fully Compliant `BackendV2` Interface**: A modern, robust, and maintainable provider implementation.
- **Statevector Simulation**: Supports ideal statevector simulation via the `save_statevector()` instruction.
- **Measurement Sampling**: Supports realistic measurement outcomes and provides a counts dictionary via `get_counts()`.
- **Automatic Discovery**: Once installed, Qiskit can automatically discover and list this provider's backends.

## Installation

### Prerequisites

Before installing this package, you must have the `rocQuantum-1` library and its Python binding (`rocquantum_bind`) already built and installed. Please follow the instructions in the `rocQuantum-1` project repository.

### Installation Steps

Once the prerequisites are met, you can install this package from the root directory using pip:

```bash
pip install .
```

After installation, Qiskit will automatically discover the `rocquantum_simulator` backend.

## Usage Example

The following example demonstrates how to use the `RocQuantumBackend` to get both a final statevector and measurement counts for a GHZ state circuit.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_rocquantum_provider import RocQuantumProvider

# 1. Instantiate the provider and get the backend
# Although Qiskit can discover the backend, direct instantiation is also clear.
provider = RocQuantumProvider()
backend = provider.get_backend("rocquantum_simulator")

# 2. Create a circuit and run for the statevector
qc_state = QuantumCircuit(3)
qc_state.h(0)
qc_state.cx(0, 1)
qc_state.cx(0, 2)
qc_state.save_statevector()

print("Running statevector simulation...")
job_state = backend.run(qc_state)
result_state = job_state.result()
final_state = result_state.get_statevector()

print("Final State Vector:")
print(np.around(final_state, 3))
print("-" * 20)


# 3. Create a similar circuit and run for measurement counts
shots = 2000
qc_counts = QuantumCircuit(3, 3)
qc_counts.h(0)
qc_counts.cx(0, 1)
qc_counts.cx(0, 2)
qc_counts.measure_all()

print(f"Running measurement simulation ({shots} shots)...")
job_counts = backend.run(qc_counts, shots=shots)
result_counts = job_counts.result()
counts = result_counts.get_counts()

print("Measurement Counts:")
print(counts)


# Expected output:
# Running statevector simulation...
# Final State Vector:
# [0.707+0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.707+0.j]
# --------------------
# Running measurement simulation (2000 shots)...
# Measurement Counts:
# {'000': 1005, '111': 995} (Note: Counts will vary slightly on each run)
```
