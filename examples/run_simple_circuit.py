import sys
import os
import numpy as np

# Add the python module to the path if it's not installed
# This assumes the script is run from the root of the repository
# and the compiled module is in build/python/rocq or similar
module_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'build/python/rocq'),
    os.path.join(os.path.dirname(__file__), '..', 'python'), # If __init__.py is in python/rocq
]
# Adjust based on your actual build output directory structure for the .so file
# For development, people often build in a 'build' dir and run examples from root.
# This path adjustment tries to find the module if it's in a typical build layout.

# Try to find the local build if not installed
# This is a common pattern for running examples before installation.
# It assumes a certain build directory structure.
# If you install the package, you can just do `import rocq`
try:
    import rocq
except ImportError:
    for path_to_try in module_paths:
        if os.path.exists(os.path.join(path_to_try, 'rocq')): # Check if rocq package dir exists
             print(f"Adding {path_to_try} to sys.path")
             sys.path.insert(0, os.path.abspath(path_to_try))
             break
    try:
        import rocq
    except ImportError as e:
        print("Failed to import rocq. Ensure the module is built and in PYTHONPATH,")
        print("or run this script from the repository root after building.")
        print(f"Original error: {e}")
        sys.exit(1)


def main():
    print("Starting rocQuantum simple circuit example...")

    try:
        # 1. Create a Simulator instance
        print("Initializing simulator...")
        sim = rocq.Simulator()
        print("Simulator initialized.")

        # 2. Create a Circuit instance
        num_qubits = 3
        print(f"Creating a circuit with {num_qubits} qubits...")
        circuit = rocq.Circuit(num_qubits=num_qubits, simulator=sim)
        print("Circuit created.")

        # 3. Apply some gates
        print("Applying gates...")
        circuit.h(0)
        print("- Applied H to qubit 0")
        circuit.cx(0, 1)
        print("- Applied CX to control=0, target=1 (CNOT)")
        circuit.rz(np.pi / 4, 2) # Apply Rz(pi/4) to qubit 2
        print(f"- Applied Rz(pi/4) to qubit 2")

        # Example of applying a generic 2-qubit unitary (SWAP matrix)
        # SWAP = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        swap_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex64)
        
        print("Applying generic SWAP matrix to qubits 1 and 2...")
        # The apply_unitary expects a DeviceBuffer.
        # The Circuit class's apply_unitary will handle creating it.
        circuit.apply_unitary(qubit_indices=[1, 2], matrix=swap_matrix)
        print("- Generic SWAP matrix applied.")


        # 4. Perform measurements
        print("Performing measurements...")
        
        # Measure qubit 0
        q0_outcome, q0_prob = circuit.measure(0)
        print(f"- Measured qubit 0: outcome = {q0_outcome}, probability = {q0_prob:.4f}")
        
        # Note: State is collapsed after measurement. 
        # If you want to measure another qubit on the *original* state, 
        # you'd need to recreate the circuit or manage state copies.
        # For this example, we continue with the collapsed state.

        # Measure qubit 1 (on the state collapsed by measuring qubit 0)
        # To get meaningful probabilities for qubit 1 independent of qubit 0's collapse,
        # one would typically run the circuit again or use a non-destructive measurement method
        # if the backend supported it (e.g. getting full probability vector).
        # The current `measure` is destructive.
        
        # Let's re-initialize and apply a simple state for a clearer second measurement
        print("Re-initializing circuit for another measurement example...")
        circuit2 = rocq.Circuit(num_qubits=2, simulator=sim)
        circuit2.h(0)
        # Now measure qubit 0 of circuit2
        q0_c2_outcome, q0_c2_prob = circuit2.measure(0)
        print(f"- Circuit 2: Measured qubit 0: outcome = {q0_c2_outcome}, probability = {q0_c2_prob:.4f}")
        # Measure qubit 1 of circuit2 (state is now collapsed based on q0_c2_outcome)
        q1_c2_outcome, q1_c2_prob = circuit2.measure(1)
        print(f"- Circuit 2: Measured qubit 1 (after q0 collapse): outcome = {q1_c2_outcome}, probability = {q1_c2_prob:.4f}")


        print("\nTo get full state probabilities (not implemented in this example directly via a single call):")
        print("One would typically use a dedicated function in hipStateVec to get the full state vector")
        print("or probabilities, or sample multiple shots from the final state without intermediate collapse.")
        print("The current 'measure' method is destructive.")


    except RuntimeError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\nSimple circuit example finished.")

if __name__ == "__main__":
    main()
