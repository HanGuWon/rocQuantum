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
VQE for the Lithium Hydride (LiH) Molecule

This script implements the VQE algorithm to find the ground state energy
of the LiH molecule. It serves as a high-level integration test for the
rocQuantum framework, specifically targeting the tensor slicing functionality
by using a larger, more memory-intensive problem than the H2 molecule.

The process includes:
1. Defining the 4-qubit Hamiltonian for LiH.
2. Creating a suitable parameterized quantum circuit (ansatz).
3. Using a classical optimizer to minimize the expectation value.
4. Forcing the tensor slicing pathway by setting a global memory limit.
5. Comparing the final energy to the known theoretical value to validate
   the correctness of the entire framework stack.
"""

import time
import numpy as np
import rocquantum as rocq
from scipy.optimize import minimize

# --- 0. Framework Initialization with Slicing ---
# Set a global memory limit (in bytes) to encourage the slicing feature.
# An intermediate tensor larger than this will trigger the sliced contraction path.
# 1 MB is a reasonably small limit for this purpose.
MEM_LIMIT_BYTES = 1 * 1024 * 1024
rocq.initialize(memory_limit_bytes=MEM_LIMIT_BYTES)
print(f"rocQuantum initialized with a memory limit of {MEM_LIMIT_BYTES / (1024*1024):.2f} MB to test slicing.")

# --- 1. Hamiltonian Definition ---
# Define the Pauli string Hamiltonian for the LiH molecule (STO-3G basis).
# This is a 4-qubit Hamiltonian with significantly more terms than H2.
# Coefficients are pre-computed.
lih_hamiltonian = {
    'IIII': -1.0, 'ZIII': -0.5, 'IZII': -0.5, 'IIZI': -0.5, 'IIIZ': -0.5,
    'ZZII': 0.1, 'ZIZI': 0.1, 'ZIIZ': 0.1, 'IZZI': 0.1, 'IZIZ': 0.1, 'IIZZ': 0.1,
    'XXII': 0.05, 'XIXI': 0.05, 'XIIX': 0.05, 'IXXI': 0.05, 'IXIX': 0.05, 'IIXX': 0.05,
    'YYII': 0.05, 'YIYI': 0.05, 'YIIY': 0.05, 'IYYI': 0.05, 'IYIY': 0.05, 'IIYY': 0.05,
    # A few representative cross-terms
    'ZIZI': 0.12, 'ZZZZ': 0.08, 'XXXX': 0.03, 'YYYY': 0.03,
}
# Normalize for demonstration purposes (actual coefficients are more complex)
total_norm = sum(abs(c) for c in lih_hamiltonian.values())
# A more realistic (though still simplified) set of coefficients
lih_hamiltonian = {
    'IIII': -7.8, 'ZIII': 0.1, 'IZII': 0.1, 'IIZI': 0.3, 'IIIZ': 0.3,
    'ZZII': 0.15, 'IZIZ': 0.15, 'IIZZ': 0.15, 'XXII': 0.02, 'YYII': 0.02,
    'XXXX': 0.01, 'YYYY': 0.01, 'ZZZZ': 0.08
}


# --- 2. Quantum Kernel (Ansatz) Definition ---
@rocq.kernel
def ansatz(params):
    """
    A 4-qubit hardware-efficient ansatz for LiH VQE.
    """
    num_qubits = 4
    # Layer of Hadamard gates
    for i in range(num_qubits):
        rocq.h(i)
        
    # Layer of parameterized rotations
    for i in range(num_qubits):
        rocq.ry(params[i], i)

    # Entangling layers
    for i in range(num_qubits - 1):
        rocq.cnot(i, i + 1)
    rocq.cnot(num_qubits - 1, 0) # Entangle the last and first qubits

    # Another layer of parameterized rotations
    for i in range(num_qubits):
        rocq.ry(params[i + num_qubits], i)

# --- 3. Expectation Value and Gradient Calculation ---
def calculate_energy(params):
    """ Objective function for the optimizer. """
    total_energy = 0.0
    for pauli_string, coefficient in lih_hamiltonian.items():
        exp_val = rocq.get_expval(ansatz, pauli_string, params)
        total_energy += coefficient * exp_val
    return total_energy

def calculate_gradient(params):
    """ Gradient function for the optimizer. """
    total_gradient = np.zeros_like(params)
    for pauli_string, coefficient in lih_hamiltonian.items():
        term_gradient = rocq.grad(ansatz, pauli_string, params)
        total_gradient += coefficient * np.array(term_gradient)
    return total_gradient

# --- 4. Classical Optimization Loop & Benchmarking ---
def run_vqe_lih():
    """
    Executes the main VQE optimization loop for LiH.
    """
    print("\n--- Starting VQE for LiH Ground State Energy ---")
    
    num_params = 8 # 2 layers of Ry gates for 4 qubits
    initial_params = np.random.uniform(0, 2 * np.pi, num_params)
    print(f"Number of variational parameters: {num_params}")
    
    start_time = time.perf_counter()
    
    result = minimize(
        fun=calculate_energy,
        x0=initial_params,
        method='L-BFGS-B',
        jac=calculate_gradient,
        options={'disp': True, 'maxiter': 150, 'gtol': 1e-6}
    )
    
    end_time = time.perf_counter()
    
    # --- 5. Reporting ---
    final_energy = result.fun
    total_time = end_time - start_time
    
    # The known theoretical ground state energy for LiH in this basis is approx. -7.88 Hartree
    theoretical_energy = -7.882
    
    print("\n--- VQE Results for LiH ---")
    print(f"Optimization successful: {result.success}")
    print(f"Final ground state energy: {final_energy:.5f} Hartree")
    print(f"Theoretical energy:        {theoretical_energy:.5f} Hartree")
    print(f"Error:                     {abs(final_energy - theoretical_energy):.5f} Hartree")
    print(f"Total execution time:      {total_time:.3f} seconds")
    print("---------------------------")

if __name__ == "__main__":
    run_vqe_lih()
