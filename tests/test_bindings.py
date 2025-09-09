import numpy as np
import os
import sys

# Add build directory to path to find the compiled module
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, build_path)

try:
    import rocquantum_bind
except ImportError:
    print(f"Error: Could not import rocquantum_bind from '{build_path}'.")
    print("Please ensure the module is compiled.")
    sys.exit(1)

def test_bell_state_simulation():
    """
    Tests the C++ simulator by creating a Bell state and checking the final state vector.
    """
    num_qubits = 2
    expected_sv_size = 2**num_qubits

    print(f"--- Testing Bell State Simulation ({num_qubits} qubits) ---")

    # 1. Initialization
    sim = rocquantum_bind.QuantumSimulator(num_qubits=num_qubits)
    
    # 2. Create Bell State: H on qubit 0, CNOT on (0, 1)
    print("\n1. Applying Hadamard to qubit 0...")
    sim.apply_gate("H", [0], [])
    
    print("2. Applying CNOT to (control=0, target=1)...")
    sim.apply_gate("CNOT", [0, 1], [])

    # 3. Get and verify the state vector
    print("\n3. Verifying final state vector...")
    statevector = sim.get_statevector()
    
    assert len(statevector) == expected_sv_size, f"Statevector size should be {expected_sv_size}"
    
    # Expected Bell state: 1/sqrt(2) * (|00> + |11>)
    expected_state = np.zeros(expected_sv_size, dtype=np.complex128)
    expected_state[0] = 1 / np.sqrt(2)
    expected_state[3] = 1 / np.sqrt(2)
    
    print(f"   Expected state: {expected_state}")
    print(f"   Actual state:   {statevector}")
    
    assert np.allclose(statevector, expected_state), "Final state vector does not match the expected Bell state."
    print("   State vector PASSED.")

    # 4. Test measurement
    print("\n4. Testing measurement...")
    shots = 2000
    measure_qubits = [0, 1]
    results = sim.measure(measure_qubits, shots)
    
    assert len(results) == shots, f"Expected {shots} measurement shots, got {len(results)}"
    
    counts = {0: 0, 3: 0}
    for r in results:
        if r in counts:
            counts[r] += 1
    
    print(f"   Measurement counts for states 0 (|00>) and 3 (|11>): {counts}")
    # Check if results are close to 50/50 split
    assert abs(counts[0] - shots/2) < shots/10, "Measurement of |00> is not within 10% tolerance."
    assert abs(counts[3] - shots/2) < shots/10, "Measurement of |11> is not within 10% tolerance."
    print("   Measurement PASSED.")

    print("\n--- Core simulation tests completed successfully! ---")

if __name__ == "__main__":
    test_bell_state_simulation()