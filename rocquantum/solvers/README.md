# VQE Solver Framework

## Overview

This document describes the high-level Variational Quantum Eigensolver (VQE) framework within the `rocQuantum` library. The `VQE_Solver` class provides a powerful, user-friendly interface to run VQE experiments, abstracting away the boilerplate of the classical optimization loop.

The framework is designed for extensibility, particularly regarding the classical optimizer, allowing researchers and developers to easily plug in custom optimization routines.

## Core Concepts

The framework is built on three key components:

1.  **`VQE_Solver`**: The main entry point for the user. This class orchestrates the entire VQE algorithm, managing the interaction between the classical optimizer and the quantum backend. It takes a problem Hamiltonian and an ansatz and returns the optimal energy and parameters.

2.  **`@roc_q.kernel` Ansatz**: The variational form (ansatz) is not defined by a class, but rather as a Python function decorated with `@roc_q.kernel`. This function defines the sequence of parameterized quantum gates. The `VQE_Solver` will call `roc_q.build` on this kernel during each optimization step.

3.  **`Optimizer` (Strategy)**: The classical optimizer is implemented using a Strategy design pattern. The `VQE_Solver` accepts an `Optimizer` object, which must conform to a simple interface. This allows any optimization library (e.g., SciPy, NLopt, or a custom gradient-based method) to be wrapped and used with the solver. The default implementation, `SciPyOptimizer`, is provided for convenience.

## Basic Usage

Here is a complete example of how to find the ground state energy of a simple Hamiltonian using the `VQE_Solver`.

```python
import numpy as np
import rocquantum.python.rocq as roc_q
from rocquantum.python.rocq import PauliOperator
from rocquantum.solvers.vqe_solver import VQE_Solver, SciPyOptimizer

# 1. Initialize the rocQuantum Simulator
try:
    sim = roc_q.Simulator()
    print("rocQuantum Simulator initialized successfully.")
except Exception as e:
    print(f"Failed to initialize simulator: {e}")
    exit()

# 2. Problem Definition
# Define the Hamiltonian using roc_q.PauliOperator
hamiltonian = PauliOperator({
    "Z0 Z1": -1.0,
    "X0": -0.5,
    "X1": -0.5
})

# Define a parameterized ansatz using the @roc_q.kernel decorator
@roc_q.kernel
def simple_ansatz(q, theta_0: float, theta_1: float):
    q.h(0)
    q.h(1)
    q.rx(theta_0, 0)
    q.rx(theta_1, 1)
    q.cx(0, 1)

# Provide an initial guess for the parameters
initial_parameters = np.array([0.5, 0.5])
num_qubits_for_problem = 2

# 3. Instantiate and Run the Solver
# Plug in the desired optimizer strategy.
scipy_optimizer = SciPyOptimizer(options={'method': 'COBYLA', 'tol': 1e-6})
vqe_solver = VQE_Solver(simulator=sim, optimizer=scipy_optimizer)

# Run the VQE algorithm with a single call to solve()
vqe_result = vqe_solver.solve(
    hamiltonian=hamiltonian,
    ansatz_kernel=simple_ansatz,
    num_qubits=num_qubits_for_problem,
    initial_params=initial_parameters
)

# 4. Print the Results
print("\n--- VQE Results ---")
print(f"Optimal Energy: {vqe_result['optimal_energy']:.8f}")
print(f"Optimal Parameters: {vqe_result['optimal_parameters']}")
print("-------------------")
```

## Extensibility Guide

### Adding a New Optimizer

The solver's design makes it simple to add a new classical optimizer. You only need to create a new class that inherits from the `Optimizer` abstract base class and implements the `minimize` method.

For example, if you wanted to wrap a fictional optimization library called `custom_opt`, you would do the following:

```python
import numpy as np
from scipy.optimize import OptimizeResult
from rocquantum.solvers.vqe_solver import Optimizer
# Fictional third-party library
import custom_opt

class CustomOptimizer(Optimizer):
    """
    An example of a custom optimizer wrapper.
    """
    def __init__(self, max_iters=100):
        self.max_iters = max_iters

    def minimize(self, fun, x0, args=()):
        # Call the fictional library's optimization function
        # Note: You are responsible for adapting the function signature and
        # return type to match the Optimizer interface.
        raw_result = custom_opt.find_minimum(
            objective=fun,
            initial_guess=x0,
            callback_args=args,
            max_iterations=self.max_iters
        )

        # Wrap the result in a SciPy-compatible OptimizeResult object
        result = OptimizeResult(
            fun=raw_result['final_value'],
            x=raw_result['best_params'],
            success=raw_result['converged'],
            message=raw_result['reason']
        )
        return result

# --- Usage with the VQE_Solver ---
# my_custom_optimizer = CustomOptimizer(max_iters=500)
# vqe_solver = VQE_Solver(simulator=sim, optimizer=my_custom_optimizer)
#
# # ... then call vqe_solver.solve(...) as usual
```

