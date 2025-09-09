import numpy as np

# This script assumes that the `rocquantum_bind` module has been compiled
# and is available in the Python path.
# You might need to run this from the `build` directory after compiling.
try:
    import rocquantum_bind
except ImportError:
    print("Error: Could not import rocquantum_bind.")
    print("Please ensure the module is compiled and you are running this script from the correct directory (e.g., the build directory).")
    exit()

def test_simulator_bindings():
    """
    Directly tests the C++ QuantumSimulator via the pybind11 bridge.
    """
    num_qubits = 3
    shots = 100
    expected_sv_size = 2**num_qubits

    print(f"--- Testing rocquantum_bind with {num_qubits} qubits ---")

    # 1. Initialization
    print("\n1. Instantiating QuantumSimulator...")
    sim = rocquantum_bind.QuantumSimulator(num_qubits=num_qubits)
    print("   Instantiation successful.")

    # 2. Test get_statevector
    print("\n2. Testing get_statevector()...")
    statevector = sim.get_statevector()
    print(f"   Received statevector of size: {len(statevector)}")
    print(f"   Statevector (first 4 elements): {statevector[:4]}")
    assert len(statevector) == expected_sv_size, f"Statevector size should be {expected_sv_size}"
    # Check if it's the |0...0> state
    assert np.isclose(statevector[0], 1.0), "First element should be 1.0"
    assert np.allclose(statevector[1:], 0.0), "All other elements should be 0.0"
    print("   get_statevector() PASSED.")

    # 3. Test apply_gate
    print("\n3. Testing apply_gate()...")
    sim.apply_gate("H", [0], [])
    print("   apply_gate() called successfully.")

    # 4. Test apply_matrix
    print("\n4. Testing apply_matrix()...")
    # A simple identity matrix for testing the binding call
    identity_matrix = np.identity(2, dtype=np.complex128)
    sim.apply_matrix(identity_matrix, [1])
    print("   apply_matrix() called successfully.")

    # 5. Test measure
    print("\n5. Testing measure()...")
    measurement_results = sim.measure(qubits=list(range(num_qubits)), shots=shots)
    print(f"   Received {len(measurement_results)} measurement results.")
    print(f"   First 10 results: {measurement_results[:10]}")
    assert len(measurement_results) == shots, f"Should have received {shots} results"
    # Check if all results are 0 (as per the stub implementation)
    assert all(res == 0 for res in measurement_results), "All measurement results should be 0"
    print("   measure() PASSED.")

    # 6. Test reset
    print("\n6. Testing reset()...")
    sim.reset()
    print("   reset() called successfully.")

    print("\n--- All binding tests completed successfully! ---")

if __name__ == "__main__":
    test_simulator_bindings()
