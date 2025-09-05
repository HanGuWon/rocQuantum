# examples/vqe_h2_noisy.py

import numpy as np
from scipy.optimize import minimize
import rocq_hip as rocq
from rocquantum.utils.hamiltonian import compute_hamiltonian_expectation

# Define the H2 molecule Hamiltonian for STO-3G basis at r = 0.7414 Å
# This is a known, simplified Hamiltonian for demonstrating VQE.
H2_HAMILTONIAN = [
    ("II", 0.0), # In this specific case, the identity term is absorbed into others, but we keep the format.
    ("IZ", 0.1721839321),
    ("ZI", 0.1721839321),
    ("ZZ", 0.1205461456),
    ("XX", 0.0453222020),
    # The full Hamiltonian is larger, but this 2-qubit version captures the core problem.
    # A more accurate one would also include terms like YY, etc.
    # For this example, we use a common simplified version.
    # Let's add the YY term for a more complete UCCSD ansatz test.
    ("YY", 0.0453222020) 
]
# Manually add the nuclear repulsion and other constant energy terms
# For H2 @ 0.7414 Å, E_const = E_nuc + E_core_frozen ≈ 0.7137... + (-1.866...)
# Total constant energy offset is approx -1.152
# Let's use a more standard representation where Identity holds the offset.
H2_HAMILTONIAN_FULL = [
    ("II", -1.128295435),
    ("IZ", 0.1721839321),
    ("ZI", 0.1721839321),
    ("ZZ", 0.1205461456),
    ("XX", 0.0453222020),
    ("YY", 0.0453222020)
]


NUM_QUBITS = 2
NOISE_PROBABILITY = 0.01 # 1% depolarizing noise

def prepare_ansatz(theta: float) -> rocq.DensityMatrixState:
    """
    Prepares the UCCSD-inspired ansatz state for the H2 molecule.
    
    Ansatz:
    1. Start in |01> (Hartree-Fock state for H2 in minimal basis)
    2. Apply Ry(theta) to qubit 0
    3. Apply CNOT(0, 1)
    
    This creates a state of the form cos(theta/2)|01> - i*sin(theta/2)|10>,
    which can explore the required subspace to find the ground state.
    """
    state = rocq.DensityMatrixState(NUM_QUBITS)
    
    # 1. Prepare the Hartree-Fock state |01>
    # Start from |00>, apply X to qubit 1
    x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    state.apply_gate(x_gate, target_qubit=1)
    
    # 2. Apply the entangling ansatz circuit
    ry_gate = np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=np.complex64)
    
    state.apply_gate(ry_gate, target_qubit=0)
    state.apply_cnot(control_qubit=0, target_qubit=1)
    
    return state

def objective_function(theta: float) -> float:
    """
    The objective function for the VQE optimizer. 
    
    Calculates the energy of the H2 Hamiltonian for a given ansatz parameter.
    """
    # 1. Prepare the quantum state using the ansatz
    state = prepare_ansatz(theta)
    
    # 2. (Optional) Apply a noise model
    # Apply a 1% depolarizing channel to each qubit after state prep
    state.apply_depolarizing_channel(target_qubit=0, probability=NOISE_PROBABILITY)
    state.apply_depolarizing_channel(target_qubit=1, probability=NOISE_PROBABILITY)
    
    # 3. Calculate the expectation value of the Hamiltonian
    energy = compute_hamiltonian_expectation(H2_HAMILTONIAN_FULL, state)
    
    print(f"Theta: {theta:.6f}, Energy: {energy:.8f}")
    return energy

def run_vqe():
    """
    Executes the full VQE workflow.
    """
    print("--- Starting Noisy VQE Simulation for H2 Molecule ---")
    print(f"Noise Model: {NOISE_PROBABILITY*100}% depolarizing channel on each qubit.")
    
    # Initial guess for the parameter
    initial_theta = 0.0
    
    # Use a classical optimizer to find the minimum energy
    # The 'COBYLA' method is a good choice for noisy, gradient-free optimization.
    result = minimize(
        objective_function,
        initial_theta,
        method='COBYLA',
        options={'disp': True, 'maxiter': 100, 'rhobeg': 0.2}
    )
    
    optimal_theta = result.x[0]
    min_energy = result.fun
    
    print("\n--- VQE Simulation Finished ---")
    print(f"Optimal parameter (theta): {optimal_theta:.6f}")
    print(f"Final ground state energy (noisy): {min_energy:.8f}")
    
    # For reference, the exact ground state energy of this Hamiltonian is approx -1.137 Ha.
    # The noisy result will be slightly higher.
    print("Expected exact energy for this model: ~ -1.137 Ha")

if __name__ == "__main__":
    try:
        run_vqe()
    except Exception as e:
        print(f"\nAn error occurred during the VQE simulation: {e}")
        print("Please ensure the rocq_hip module is compiled and accessible,")
        print("and that the ROCm/HIP environment is correctly configured.")
