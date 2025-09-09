import sys
import os
import numpy as np

# --- Setup Python Path ---
project_root = os.path.dirname(os.path.abspath(__file__))
integrations_path = os.path.join(project_root, '..', 'integrations')
build_path = os.path.join(project_root, '..', 'build')
sys.path.insert(0, os.path.abspath(integrations_path))
sys.path.insert(0, os.path.abspath(build_path))

try:
    import pennylane as qml
    from qiskit import QuantumCircuit, transpile
    from qiskit_rocquantum_provider.provider import RocQuantumProvider
    print("Successfully imported all frameworks and providers.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you have run the build commands and that PennyLane and Qiskit are installed.")
    sys.exit(1)

def test_pennylane_integration():
    print("\n--- Testing PennyLane Integration (Bell State) ---")
    try:
        dev = qml.device("rocq.pennylane", wires=2, shots=1024)
        
        @qml.qnode(dev)
        def bell_state_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        state_vector = bell_state_circuit()
        print(f"Execution complete. Resulting state vector:\n{state_vector}")

        expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        assert np.allclose(state_vector, expected_state), "State vector does not match expected Bell state."
        
        print("PennyLane integration test PASSED.")
    except Exception as e:
        print(f"PennyLane integration test FAILED: {e}")
        raise e

def test_qiskit_integration():
    print("\n--- Testing Qiskit Integration (Bell State) ---")
    try:
        provider = RocQuantumProvider()
        backend = provider.get_backend("rocq_simulator")
        
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        # Transpile for the backend
        t_qc = transpile(qc, backend)
        
        shots = 2048
        job = backend.run(t_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        print(f"Execution complete. Result counts: {counts}")

        assert result.success, "The result object should report success."
        # Qiskit counts keys are hex strings of the integer result
        # e.g., 0 -> '0x0', 3 -> '0x3'
        # Note: The binder might return '0b0' format, need to be robust. Let's check for int keys from binder.
        # Let's assume the binder gives integer keys for now.
        # The provider should format them. Let's assume the provider returns string keys like '00' and '11'.
        # The current backend.py returns '0b...' strings.
        
        # Let's check for the presence of the two expected outcomes
        assert '0b0' in counts or '0' in counts
        assert '0b11' in counts or '3' in counts

        # Verify the distribution is roughly 50/50
        count_00 = counts.get('0b0', counts.get('0', 0))
        count_11 = counts.get('0b11', counts.get('3', 0))
        
        assert abs(count_00 - shots/2) < shots/10, "Measurement of '00' is not within 10% tolerance."
        assert abs(count_11 - shots/2) < shots/10, "Measurement of '11' is not within 10% tolerance."

        print("Qiskit integration test PASSED.")
    except Exception as e:
        print(f"Qiskit integration test FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_pennylane_integration()
    test_qiskit_integration()