# rocQuantum-1

A high-performance, CUDA Quantum-like quantum computing simulation framework for AMD GPUs. rocQuantum-1 provides a high-level, user-centric programming model built on top of a powerful C++/HIP backend, designed for modern quantum algorithm development.

## Core Features

*   **Modern, kernel-based programming model:** Define reusable, parameterized quantum circuits with the intuitive `@rocq.kernel` decorator.
*   **High-performance C++/HIP backends:** Leverage the full power of AMD GPUs for accelerated simulation.
*   **State-vector simulation:** For ideal, noise-free quantum systems.
*   **Density-matrix simulation:** For realistic simulations featuring a high-level noise model (`rocq.NoiseModel`).
*   **Foundation for operator algebra:** Easily define Hamiltonians and other observables using `rocq.PauliOperator` for expectation value calculations.

## Installation

(Installation instructions will be provided here in the future.)

## Quick Start

The following example demonstrates the primary workflow for defining a quantum kernel, specifying a Hamiltonian, and calculating its expectation value.

```python
import rocq

# 1. Define a parameterized kernel to prepare a quantum state.
@rocq.kernel
def prepare_ghz_state(theta: float):
    """Prepares a generalized GHZ state."""
    q = rocq.qvec(3)
    rocq.h(q[0])
    rocq.cnot(q[0], q[1])
    rocq.ry(theta, q[2])
    rocq.cnot(q[1], q[2])

# 2. Define a Hamiltonian (e.g., H = 0.5*X0*Z1 + 0.3*Z0*Y2)
hamiltonian = 0.5 * rocq.PauliOperator("X0 Z1") + 0.3 * rocq.PauliOperator("Z0 Y2")

# 3. Execute the kernel and compute the expectation value of the Hamiltonian.
# The C++ backend is mocked, so this will return a placeholder value.
exp_val = rocq.get_expectation_value(
    prepare_ghz_state,
    hamiltonian,
    backend='state_vector',
    theta=1.57  # Provide a concrete value for the 'theta' parameter
)

print(f"Computed Expectation Value: {exp_val}")
```

## Running Tests

To verify the installation and the integrity of the framework, run the test suite from the project's root directory:

```shell
python -m unittest tests/test_framework.py
```
