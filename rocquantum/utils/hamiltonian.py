# rocquantum/utils/hamiltonian.py

import numpy as np
import rocq_hip as rocq

# Define basis-change gates
H_GATE = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex64)

def _compute_pauli_string_expectation(state: rocq.DensityMatrixState, pauli_string: str) -> float:
    """
    Computes the expectation value for a single multi-qubit Pauli string.
    
    This function applies basis-change gates, measures the Z-product expectation,
    and then applies the inverse basis-change gates to restore the original state.
    """
    if len(pauli_string) > 64: # A reasonable limit
        raise ValueError("Pauli string is too long.")

    z_indices = []
    basis_change_qubits = {"X": [], "Y": []}

    for i, pauli_op in enumerate(pauli_string):
        if pauli_op == 'X':
            z_indices.append(i)
            basis_change_qubits["X"].append(i)
        elif pauli_op == 'Y':
            z_indices.append(i)
            basis_change_qubits["Y"].append(i)
        elif pauli_op == 'Z':
            z_indices.append(i)
        elif pauli_op != 'I':
            raise ValueError(f"Invalid Pauli operator '{pauli_op}' in string.")

    # --- Pre-measurement basis change ---
    # Apply S† for Y basis
    for qubit_idx in basis_change_qubits["Y"]:
        state.apply_gate(S_GATE, qubit_idx, adjoint=True)
    # Apply H for X and Y basis
    for qubit_idx in basis_change_qubits["X"]:
        state.apply_gate(H_GATE, qubit_idx)
    for qubit_idx in basis_change_qubits["Y"]:
        state.apply_gate(H_GATE, qubit_idx)

    # --- Measure in Z-basis ---
    if not z_indices: # The term is just Identity
        expectation_value = 1.0
    else:
        expectation_value = state._compute_z_product_expectation(z_indices)

    # --- Post-measurement basis change (inverse operations) ---
    # Apply H for X and Y basis
    for qubit_idx in basis_change_qubits["Y"]:
        state.apply_gate(H_GATE, qubit_idx)
    for qubit_idx in basis_change_qubits["X"]:
        state.apply_gate(H_GATE, qubit_idx)
    # Apply S for Y basis
    for qubit_idx in basis_change_qubits["Y"]:
        state.apply_gate(S_GATE, qubit_idx)
        
    return expectation_value

def compute_hamiltonian_expectation(
    hamiltonian: list[tuple[str, float]], 
    state: rocq.DensityMatrixState
) -> float:
    """
    Computes the total expectation value of a Hamiltonian for a given quantum state.

    The Hamiltonian is specified as a list of tuples, where each tuple contains
    a Pauli string (e.g., 'IXYZ') and its corresponding coefficient.

    Args:
        hamiltonian: The Hamiltonian definition.
        state: The quantum state (DensityMatrixState) to measure.

    Returns:
        The total energy (expectation value of the Hamiltonian).
    """
    total_energy = 0.0
    for pauli_string, coefficient in hamiltonian:
        term_expectation = _compute_pauli_string_expectation(state, pauli_string)
        total_energy += coefficient * term_expectation
    
    return total_energy
