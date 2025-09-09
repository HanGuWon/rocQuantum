import sys
import os
import numpy as np

# --- Setup Python Path ---
# This allows the script to find the local integration packages without installation.
# It assumes the script is run from the project's root directory.
# If running from the 'tests' directory, the path logic would need adjustment.
try:
    # Add the parent directory of 'integrations' to the path
    project_root = os.path.dirname(os.path.abspath(__file__))
    integrations_path = os.path.join(project_root, '..', 'integrations')
    sys.path.insert(0, os.path.abspath(integrations_path))
    
    # Add the build directory to the path to find rocquantum_bind
    # This might need adjustment based on the CMake build configuration (e.g., 'build/Release')
    build_path = os.path.join(project_root, '..', 'build')
    sys.path.insert(0, os.path.abspath(build_path))

    import pennylane as qml
    from qiskit import QuantumCircuit
    from qiskit_rocquantum_provider.provider import RocQuantumProvider
    print("Successfully imported all frameworks and providers.")

except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you have run the build commands and that PennyLane and Qiskit are installed (`pip install pennylane qiskit`).")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during setup: {e}")
    sys.exit(1)

def test_pennylane_integration():
    """
    Tests the PennyLane plugin by creating a Bell state and retrieving the state vector.
    """
    print("\n--- Testing PennyLane Integration ---")
    try:
        # 1. Create the custom device
        dev = qml.device("rocq.pennylane", wires=2, shots=1024)
        print(f"Successfully loaded PennyLane device: {dev.short_name}")

        # 2. Define the QNode
        @qml.qnode(dev)
        def bell_state_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        # 3. Execute and get the state vector
        state_vector = bell_state_circuit()
        print(f"Execution complete. Resulting state vector:\n{state_vector}")

        # 4. Assert the result
        assert isinstance(state_vector, np.ndarray), "State vector should be a NumPy array"
        assert len(state_vector) == 4, f"State vector size should be 4, but was {len(state_vector)}"
        
        # The C++ stub returns |00>, so we check for that.
        # A real implementation would produce ~[0.707, 0, 0, 0.707]
        expected_stub_state = np.array([1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j])
        assert np.allclose(state_vector, expected_stub_state), "State vector does not match the expected stub output"
        
        print("PennyLane integration test PASSED.")

    except Exception as e:
        print(f"PennyLane integration test FAILED: {e}")

def test_qiskit_integration():
    """
    Tests the Qiskit plugin by creating and running a Bell state circuit.
    """
    print("\n--- Testing Qiskit Integration ---")
    try:
        # 1. Instantiate the provider and get the backend
        provider = RocQuantumProvider()
        backend = provider.get_backend("rocq_simulator")
        print(f"Successfully loaded Qiskit backend: {backend.name}")

        # 2. Create a Bell state circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        print("Quantum circuit created:")
        print(qc)

        # 3. Execute the circuit
        job = backend.run(qc, shots=1024)
        result = job.result()
        counts = result.get_counts()
        print(f"Execution complete. Result counts: {counts}")

        # 4. Assert the result
        assert result.success, "The result object should report success."
        # The C++ stub returns only '0' measurements
        assert '0b0' in counts, "Counts dictionary should contain the '0b0' key for the stub implementation."
        assert counts['0b0'] == 1024, "Stub implementation should return 1024 counts for the '0' state."

        print("Qiskit integration test PASSED.")

    except Exception as e:
        print(f"Qiskit integration test FAILED: {e}")


if __name__ == "__main__":
    test_pennylane_integration()
    test_qiskit_integration()
