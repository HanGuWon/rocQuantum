# rocQuantum-1: A High-Performance Quantum Circuit Simulator for AMD GPUs

## Overview

rocQuantum-1 is a high-performance quantum circuit simulator designed to leverage the computational power of AMD GPUs through the ROCm HIP programming model. Its primary goal is to provide a fast, seamless backend for leading quantum machine learning and computing frameworks, including PennyLane and Qiskit, enabling researchers and developers to simulate larger and more complex quantum systems than is feasible on traditional CPU-based simulators.

## Features

*   **HIP-based GPU Acceleration:** Core quantum operations are executed on AMD GPUs for significant performance gains.
*   **Seamless PennyLane Integration:** Implements the PennyLane `Device` API for easy use as a backend in PennyLane workflows.
*   **Seamless Qiskit Integration:** Implements the Qiskit `BackendV2` API, allowing it to function as a standard backend via a custom provider.
*   **Core Gate Support:** Includes high-performance kernels for essential quantum gates, including Hadamard, Pauli gates (X, Y, Z), parameterized rotations (RX, RY, RZ), and multi-qubit CNOT gates.

## Benchmark Results

Performance benchmarks show a significant advantage for rocQuantum-1 over standard CPU-based simulators, especially as the number of qubits increases. The simulation of the Quantum Fourier Transform (QFT) circuit highlights this performance gap.

![PennyLane Benchmark Results](benchmarks/benchmark_results_pennylane.png)

*As shown above, the logarithmic scale on the y-axis indicates an exponential speedup for the GPU-based rocQuantum-1 simulator.*

## Build Instructions

To build the rocQuantum-1 C++ core and the Python bindings, you will need a system with the ROCm toolkit, CMake, and a C++ compiler installed.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rocQuantum-1
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure the project with CMake:**
    ```bash
    cmake ..
    ```

4.  **Compile the project:**
    ```bash
    cmake --build . --config Release
    ```

This will create the `rocquantum.a` static library and the `rocquantum_bind.so` (or `.pyd`) Python module inside the `build` directory.

## Installation & Usage

After building the project, you can run the verification tests and use the simulator in your own Python projects.

### Running Tests

To verify that the simulator and framework plugins are working correctly, run the end-to-end test script from the project's root directory:

```bash
python tests/test_frameworks.py
```

### Usage in PennyLane

To use rocQuantum-1 as a PennyLane device, simply specify `"rocq.pennylane"` as the device name.

```python
import pennylane as qml
import numpy as np

# The build directory must be in your Python path
dev = qml.device("rocq.pennylane", wires=2, shots=1024)

@qml.qnode(dev)
def my_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

result = my_circuit()
print(f"Expectation value: {result}")
```

### Usage in Qiskit

To use rocQuantum-1 in Qiskit, import the custom provider and get the backend.

```python
from qiskit import QuantumCircuit, transpile
# Ensure the integrations and build directories are in your Python path
from qiskit_rocquantum_provider.provider import RocQuantumProvider

# 1. Get the backend from the provider
provider = RocQuantumProvider()
backend = provider.get_backend("rocq_simulator")

# 2. Create and run your circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

t_qc = transpile(qc, backend)
job = backend.run(t_qc, shots=1024)
result = job.result()
counts = result.get_counts()

print(f"Measurement counts: {counts}")
```