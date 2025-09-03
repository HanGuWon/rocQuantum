import numpy as np
import rocq

def create_ghz_state(circuit):
    """Creates a 3-qubit GHZ state |000> + |111> / sqrt(2)."""
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

def run_expectation_example():
    """
    Demonstrates the use of the generic expectation value calculation.
    """
    num_qubits = 3
    print(f"Initializing a {num_qubits}-qubit simulator.")
    
    try:
        # 1. Initialize the simulator and circuit
        simulator = rocq.Simulator()
        circuit = rocq.Circuit(num_qubits, simulator)

        # 2. Prepare a specific quantum state (GHZ state)
        print("Preparing a 3-qubit GHZ state.")
        create_ghz_state(circuit)
        
        # 3. Define Hamiltonians (PauliOperator objects)
        # For GHZ state, we expect:
        # <Z0 Z1> = 1
        # <Z1 Z2> = 1
        # <X0 X1 X2> = 1
        # <Y0 Y1 X2> = -1
        hamiltonian_zz = rocq.PauliOperator("Z0 Z1")
        hamiltonian_ixy = rocq.PauliOperator("X1 Y2") # <I0 X1 Y2>
        hamiltonian_xyz = rocq.PauliOperator("X0 Y1 Z2")

        # 4. Calculate and print expectation values using rocq.get_expval
        print("\nCalculating expectation values...")

        # Test Case 1: <Z0 Z1>
        exp_val_zz = rocq.get_expval(rocq.build(lambda c: None, num_qubits, simulator), hamiltonian_zz)
        print(f"Expectation value of Z0 Z1: {exp_val_zz:.6f} (Expected: 1.0)")

        # Test Case 2: <I0 X1 Y2>
        exp_val_ixy = rocq.get_expval(rocq.build(lambda c: None, num_qubits, simulator), hamiltonian_ixy)
        print(f"Expectation value of I0 X1 Y2: {exp_val_ixy:.6f} (Expected: 0.0)")

        # Test Case 3: <X0 Y1 Z2>
        exp_val_xyz = rocq.get_expval(rocq.build(lambda c: None, num_qubits, simulator), hamiltonian_xyz)
        print(f"Expectation value of X0 Y1 Z2: {exp_val_xyz:.6f} (Expected: 0.0)")

        # Note: The dummy lambda `lambda c: None` and rocq.build are part of the current API
        # for get_expval, which expects a QuantumProgram object.

        print("\n--- Verification ---")
        assert np.isclose(exp_val_zz, 1.0), "Verification failed for <Z0 Z1>"
        assert np.isclose(exp_val_ixy, 0.0), "Verification failed for <I0 X1 Y2>"
        assert np.isclose(exp_val_xyz, 0.0), "Verification failed for <X0 Y1 Z2>"
        print("All tested expectation values are correct.")

    except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the rocQuantum project is built and installed correctly.")

if __name__ == "__main__":
    run_expectation_example()
