# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
High-level Variational Quantum Eigensolver (VQE) using rocQuantum primitives.
"""

from typing import Callable, List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

# --- Real rocQuantum Imports ---
import rocquantum.python.rocq as roc_q
from rocquantum.python.rocq import PauliOperator

# --- Third-party Imports ---
from scipy.optimize import minimize, OptimizeResult

# --- Type Hinting Placeholders ---
AnsatzKernel = Callable[..., None]  # An ansatz is a kernel function

# --- Optimizer Strategy Pattern Definition ---

class Optimizer(ABC):
    """
    Abstract base class for classical optimizers (Strategy Pattern).

    This interface allows for different optimization algorithms to be seamlessly
    plugged into the VQE_Solver. To add a new optimizer, create a concrete
    class that inherits from this one and implement the `minimize` method.
    """
    @abstractmethod
    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        args: tuple = ()
    ) -> OptimizeResult:
        """
        Executes the minimization routine.

        Args:
            fun (Callable): The objective function to minimize.
            x0 (np.ndarray): The initial guess for the parameters.
            args (tuple): Extra arguments to pass to the objective function.

        Returns:
            OptimizeResult: The result of the optimization.
        """
        pass

class SciPyOptimizer(Optimizer):
    """
    A concrete implementation of the Optimizer strategy that wraps
    `scipy.optimize.minimize`.
    """
    def __init__(self, options: Dict[str, Any] = None):
        """
        Initializes the SciPyOptimizer.

        Args:
            options (Dict[str, Any], optional): A dictionary of options
                (e.g., {'method': 'BFGS', 'tol': 1e-6}) to be passed to
                `scipy.optimize.minimize`. Defaults to a standard configuration.
        """
        if options is None:
            self.options = {'method': 'COBYLA', 'tol': 1e-6}
        else:
            self.options = options

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        args: tuple = ()
    ) -> OptimizeResult:
        """
        Minimizes the objective function using scipy.optimize.minimize.
        """
        return minimize(
            fun=fun,
            x0=x0,
            args=args,
            **self.options
        )

# --- VQE Solver ---

class VQE_Solver:
    """
    A high-level solver for the Variational Quantum Eigensolver (VQE) algorithm.

    This class uses a Strategy Pattern for its optimizer, allowing for easy
    extensibility with different classical optimization routines.
    """

    def __init__(
        self,
        simulator: roc_q.Simulator,
        optimizer: Optimizer = None
    ):
        """
        Initializes the VQE_Solver.

        Args:
            simulator (roc_q.Simulator): A rocQuantum simulator instance.
            optimizer (Optimizer, optional): A concrete optimizer instance that
                adheres to the Optimizer interface. If None, a default
                `SciPyOptimizer` is used.
        """
        if not isinstance(simulator, roc_q.Simulator):
            raise TypeError("A valid roc_q.Simulator instance is required.")
        self.simulator = simulator
        self.optimizer = optimizer if optimizer is not None else SciPyOptimizer()

        self._intermediate_results = []

    def _objective_function(
        self,
        params: np.ndarray,
        hamiltonian: PauliOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int
    ) -> float:
        """
        Internal objective function evaluated by the classical optimizer.
        """
        program = roc_q.build(ansatz_kernel, num_qubits, self.simulator, *params)
        energy = roc_q.get_expval(program=program, hamiltonian=hamiltonian)

        self._intermediate_results.append({'params': params.tolist(), 'energy': energy})
        print(f"Evaluated parameters {params.tolist()}, Energy: {energy:.8f}")

        return energy

    def solve(
        self,
        hamiltonian: PauliOperator,
        ansatz_kernel: AnsatzKernel,
        num_qubits: int,
        initial_params: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes the VQE algorithm.
        """
        print("Starting VQE optimization...")
        self._intermediate_results = []

        result = self.optimizer.minimize(
            fun=self._objective_function,
            x0=initial_params,
            args=(hamiltonian, ansatz_kernel, num_qubits)
        )

        print("VQE optimization finished.")

        solution = {
            'optimal_energy': result.fun,
            'optimal_parameters': result.x,
            'optimizer_result': result,
            'intermediate_results': self._intermediate_results
        }
        return solution

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Initialize the rocQuantum Simulator
    try:
        sim = roc_q.Simulator()
        print("rocQuantum Simulator initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize simulator: {e}")
        print("This example requires a functional rocQuantum installation.")
        exit()

    # 2. Problem Definition
    hamiltonian = PauliOperator({
        "Z0 Z1": -1.0,
        "X0": -0.5,
        "X1": -0.5
    })
    print("Hamiltonian:\n", hamiltonian)

    @roc_q.kernel
    def simple_ansatz(q, theta_0: float, theta_1: float):
        q.h(0)
        q.h(1)
        q.rx(theta_0, 0)
        q.rx(theta_1, 1)
        q.cx(0, 1)

    initial_parameters = np.array([0.5, 0.5])
    num_qubits_for_problem = 2

    # 3. Instantiate and Run the Solver
    # Plug in the desired optimizer strategy.
    scipy_optimizer = SciPyOptimizer(options={'method': 'COBYLA', 'tol': 1e-6})
    vqe_solver = VQE_Solver(simulator=sim, optimizer=scipy_optimizer)

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
