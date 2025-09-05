# examples/bell_state_density_matrix.py

import numpy as np
# Note: The module name 'rocq_hip' is what we defined in the PYBIND11_MODULE.
# In a real build system, this would likely be part of a larger 'rocq' package.
import rocq_hip as rocq

def create_bell_state():
    """
    Demonstrates creating a Bell state |Φ+> = (|00> + |11>)/sqrt(2)
    and measuring expectation values.
    """
    print("--- Running Bell State Creation and Measurement Example ---")
    num_qubits = 2
    
    # 1. Instantiate the density matrix simulator
    try:
        state = rocq.DensityMatrixState(num_qubits)
    except RuntimeError as e:
        print(f"Error initializing simulator: {e}")
        print("Please ensure the ROCm/HIP environment is set up correctly.")
        return

    # 2. Define standard single-qubit gates as NumPy arrays
    h_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
    
    # 3. Apply gates to create the Bell state
    # Apply H gate to qubit 0 to create superposition
    state.apply_gate(h_gate, target_qubit=0)
    
    # Apply CNOT gate with qubit 0 as control and qubit 1 as target
    state.apply_cnot(control_qubit=0, target_qubit=1)
    
    print("Successfully created a Bell state.")

    # 4. Compute expectation values for the ideal (noiseless) state
    # For the |Φ+> Bell state, <Z₀> and <Z₁> should both be 0.
    # <XX> and <YY> should be 1, <ZZ> should be 1.
    # Our current API only supports single-qubit Pauli measurements.
    exp_z0 = state.compute_expectation(rocq.Pauli.Z, target_qubit=0)
    exp_z1 = state.compute_expectation(rocq.Pauli.Z, target_qubit=1)
    
    print(f"Ideal State Expectation <Z₀>: {exp_z0:.6f} (Expected: 0.0)")
    print(f"Ideal State Expectation <Z₁>: {exp_z1:.6f} (Expected: 0.0)")
    
    # 5. (Stretch Goal) Apply noise and see how expectations change
    print("\n--- Applying Noise Channel ---")
    noise_probability = 0.1 # 10% chance of a bit-flip error
    
    # Apply bit-flip noise to the first qubit
    state.apply_bit_flip_channel(target_qubit=0, probability=noise_probability)
    print(f"Applied 10% bit-flip channel to qubit 0.")
    
    # 6. Re-compute expectation values
    # After a bit-flip, the state is partially mixed. The expectation value of Z
    # will be reduced from its ideal value.
    exp_z0_noisy = state.compute_expectation(rocq.Pauli.Z, target_qubit=0)
    exp_z1_noisy = state.compute_expectation(rocq.Pauli.Z, target_qubit=1)

    print(f"Noisy State Expectation <Z₀>: {exp_z0_noisy:.6f}")
    print(f"Noisy State Expectation <Z₁>: {exp_z1_noisy:.6f}")
    print("Note: With bit-flip noise on qubit 0, <Z₀> is reduced, while <Z₁> is also affected due to entanglement.")

if __name__ == "__main__":
    create_bell_state()
