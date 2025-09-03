# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
VQE for the Hydrogen Molecule (H2)

This script implements the Variational Quantum Eigensolver (VQE) algorithm
to find the ground state energy of the H2 molecule. It serves as an end-to-end
acceptance test and a practical demonstration of the rocQuantum framework.

The process includes:
1. Defining the molecular Hamiltonian for H2.
2. Creating a parameterized quantum circuit (ansatz).
3. Using a classical optimizer (SciPy's L-BFGS-B) to minimize the
   expectation value of the Hamiltonian.
4. Calculating the energy and its gradient using rocQuantum's capabilities.
5. Benchmarking the execution time and comparing the result to the known
   theoretical value.
"""

import time
import numpy as np
import rocquantum as rocq
from scipy.optimize import minimize

# --- 1. Hamiltonian Definition ---
# Define the Pauli string Hamiltonian for the H2 molecule at the equilibrium
# bond distance of 0.7414 Å, using the STO-3G basis.
# The coefficients are pre-computed from quantum chemistry calculations.
h2_hamiltonian = {
    'II': -0.81054798,
    'IZ':  0.17141281,
    'ZI':  0.17141281,
    'ZZ':  0.1206252,
    'XX':  0.0453222,
}

# --- 2. Quantum Kernel (Ansatz) Definition ---
# Define a simple hardware-efficient, parameterized quantum circuit (ansatz)
# for a 2-qubit system, which is sufficient for the minimal basis H2 molecule.
# The `rocq.kernel` decorator marks this function for JIT compilation by the framework.
@rocq.kernel
def ansatz(params):
    """
    A simple 2-qubit ansatz for VQE.
    
    Args:
        params (list[float]): A list of variational parameters for the gates.
    """
    # Layer of single-qubit rotations
    rocq.ry(params[0], 0)
    rocq.ry(params[1], 1)
    
    # Entangling layer
    rocq.cnot(0, 1)
    
    # Another layer of single-qubit rotations
    rocq.ry(params[2], 0)
    rocq.ry(params[3], 1)

# --- 3. Expectation Value and Gradient Calculation ---
def calculate_energy(params):
    """
    Objective function for the optimizer.
    
    Calculates the total energy of the H2 molecule for a given set of
    variational parameters by summing the expectation values of each
    term in the Hamiltonian.
    
    Args:
        params (np.ndarray): The variational parameters from the optimizer.
        
    Returns:
        float: The total energy (expectation value).
    """
    total_energy = 0.0
    # Loop through each Pauli term in the Hamiltonian
    for pauli_string, coefficient in h2_hamiltonian.items():
        # Execute the kernel and measure the expectation value for the term
        exp_val = rocq.get_expval(ansatz, pauli_string, params)
        total_energy += coefficient * exp_val
    return total_energy

def calculate_gradient(params):
    """
    Gradient function for the optimizer.
    
    Calculates the gradient of the energy with respect to the variational
    parameters. This is done by summing the gradients of the expectation
    value of each Hamiltonian term. Using gradients significantly
    accelerates the optimization process.
    
    Args:
        params (np.ndarray): The variational parameters.
        
    Returns:
        np.ndarray: The gradient vector.
    """
    total_gradient = np.zeros_like(params)
    # Loop through each Pauli term in the Hamiltonian
    for pauli_string, coefficient in h2_hamiltonian.items():
        # Calculate the gradient for the current term and add it to the total
        term_gradient = rocq.grad(ansatz, pauli_string, params)
        total_gradient += coefficient * np.array(term_gradient)
    return total_gradient

# --- 4. Classical Optimization Loop & Benchmarking ---
def run_vqe():
    """
    Executes the main VQE optimization loop and reports the results.
    """
    print("--- Starting VQE for H2 Ground State Energy ---")
    
    # Set initial random parameters for the ansatz
    num_params = 4
    initial_params = np.random.uniform(0, 2 * np.pi, num_params)
    print(f"Initial parameters: {initial_params}")
    
    # Start the timer
    start_time = time.perf_counter()
    
    # Use SciPy's minimize function to find the optimal parameters.
    # We use the L-BFGS-B method, which is a good choice for this type of problem
    # and can leverage the analytic gradient we provide via the `jac` argument.
    result = minimize(
        fun=calculate_energy,
        x0=initial_params,
        method='L-BFGS-B',
        jac=calculate_gradient,
        options={'disp': True, 'maxiter': 100}
    )
    
    # Stop the timer
    end_time = time.perf_counter()
    
    # --- 5. Reporting ---
    final_energy = result.fun
    total_time = end_time - start_time
    
    # The known theoretical ground state energy for H2 in this basis
    theoretical_energy = -1.13728
    
    print("\n--- VQE Results ---")
    print(f"Optimization successful: {result.success}")
    print(f"Final variational parameters: {result.x}")
    print(f"Final ground state energy: {final_energy:.5f} Hartree")
    print(f"Theoretical energy:        {theoretical_energy:.5f} Hartree")
    print(f"Error:                     {abs(final_energy - theoretical_energy):.5f} Hartree")
    print(f"Total execution time:      {total_time:.3f} seconds")
    print("-------------------")

if __name__ == "__main__":
    run_vqe()
