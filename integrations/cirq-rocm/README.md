# Cirq-ROCm: Cirq simulator for the rocQuantum Backend

## Overview

This package provides a `cirq.Simulator` implementation that offloads quantum circuit calculations to the [rocQuantum-1](https://github.com/ROCm/rocQuantum-1) C++/HIP library. It is designed to provide a seamless way for Cirq users to leverage AMD GPU acceleration for their quantum simulations.

## Features

- **GPU-Accelerated Simulation**: Executes simulations on the rocQuantum C++ backend.
- **Implements `SimulatesFinalState`**: Provides fast access to the final statevector of a circuit via the `simulate()` method.
- **Implements `SimulatesSamples`**: Provides efficient measurement sampling for hardware-like results via the `run()` method.
- **Standard Cirq Interface**: Behaves like any other Cirq simulator for easy integration into existing workflows.

## Installation

### Prerequisites

Before installing this package, you must have the `rocQuantum-1` library and its Python binding (`rocquantum_bind`) already built and installed. Please follow the instructions in the `rocQuantum-1` project repository.

### Installation Steps

Once the prerequisites are met, you can install this package from the root directory using pip:

```bash
pip install .
```

## Usage Example

The following example demonstrates how to use the `RocQuantumSimulator` for both statevector simulation and measurement runs.

```python
import cirq
import numpy as np
from cirq_rocm import RocQuantumSimulator

# 1. Create a quantum circuit (e.g., a Bell state)
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)

# 2. Instantiate the rocQuantum simulator
sim = RocQuantumSimulator()

# 3. Use simulate() to get the final statevector
result_state = sim.simulate(circuit)
final_state = result_state.final_state_vector

print("Final State Vector:")
print(np.around(final_state, 3))
print("-" * 20)

# 4. Use run() to get measurement samples
# Add a measurement gate to the circuit
circuit.append(cirq.measure(q0, q1, key='result'))
repetitions = 1000
result_counts = sim.run(circuit, repetitions=repetitions)

# Get and print the histogram of results
counts = result_counts.histogram(key='result')
print(f"Measurement Counts (for {repetitions} repetitions):")
print(counts)

# Expected output:
# Final State Vector:
# [0.707+0.j 0.   +0.j 0.   +0.j 0.707+0.j]
# --------------------
# Measurement Counts (for 1000 repetitions):
# Counter({0: 505, 3: 495}) (Note: Counts will vary slightly on each run)
```
