# rocQuantum-1
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

rocQuantum-1 is a modern, high-performance quantum computing simulation framework specifically designed and optimized for AMD GPUs using the ROCm software stack. It provides a user-friendly Python interface with a powerful C++/HIP backend, aiming for feature parity with leading quantum frameworks.

## Core Features
- **High-Performance Backend:** Leverages AMD's HIP for massively parallel execution of quantum circuit simulations on AMD GPUs.
- **Two Simulation Modes:**
  - `StateVectorBackend`: For ideal, noise-free simulations of quantum states.
  - `DensityMatrixBackend`: For realistic simulations including common quantum noise channels (e.g., depolarizing, bit-flip).
- **Python-First Interface:** A simple and intuitive Python API (`rocq`) for defining quantum kernels, building circuits, and running simulations.
- **Hybrid Workflow Support:** Includes high-level utilities for running hybrid quantum-classical algorithms like VQE, complete with integration for classical optimizers like `SciPy`.

## Quick Start: Expectation Value Calculation

Here is a complete example of defining a quantum circuit, specifying a Hamiltonian, and calculating its expectation value.

```python
import rocq

# 1. Define a parameterized kernel to prepare a quantum state.
@rocq.kernel
def create_bell_state(theta: float, phi: float):
    """A kernel to create a generalized Bell state."""
    q = rocq.qvec(2)
    rocq.h(q[0])
    rocq.ry(theta, q[0])
    rocq.cnot(q[0], q[1])
    rocq.rz(phi, q[1])

# 2. Define a Hamiltonian (e.g., H = 0.5*X0*Z1 + 0.2*Y0)
hamiltonian = 0.5 * rocq.PauliOperator("X0 Z1") + 0.2 * rocq.PauliOperator("Y0")

# 3. Execute the kernel and compute the expectation value.
# The C++ backend is mocked, so this will return a placeholder value.
exp_val = rocq.get_expectation_value(
    create_bell_state,
    hamiltonian,
    backend='state_vector',
    theta=0.0,
    phi=0.0
)

print(f"Computed Expectation Value: <H> = {exp_val}")
```

## Installation
(Placeholder) Currently, installation requires building from source. Ensure you have a compatible ROCm environment (e.g., ROCm 5.x) and all necessary dependencies (e.g., CMake, C++ compiler).

Clone the repository:
```bash
git clone https://your-repo-url/rocQuantum-1.git
cd rocQuantum-1
```
Build the project:
```bash
mkdir build && cd build
cmake ..
make -j
```

## Running Tests
The project uses CTest and GTest for its C++ backend tests. To run the tests, execute the following command from the `build` directory:
```bash
ctest
```

## Project Structure
- **rocquantum/**: Core C++ source code and Python bindings.
- **rocq/**: The user-facing Python package.
- **examples/**: Example scripts demonstrating framework features.
- **tests/**: C++ unit tests.
