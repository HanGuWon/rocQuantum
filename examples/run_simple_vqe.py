import numpy as np
from scipy.optimize import minimize
import rocq # Assuming rocq is installed or in PYTHONPATH

# 1. Define the quantum ansatz using @rocq.kernel
@rocq.kernel
def vqe_ansatz_simple(circuit: rocq.Circuit, theta: float, phi: float):
    """
    A simple VQE ansatz.
    Args:
        circuit: The rocq.Circuit object to apply gates to.
        theta: Parameter for an Rx gate.
        phi: Parameter for a CNOT then Rz (example).
    """
    # Example: 2-qubit ansatz
    # Apply Rx(theta) to qubit 0
    circuit.rx(theta, 0)
    # Apply CNOT between qubit 0 (control) and qubit 1 (target)
    if circuit.num_qubits > 1: # Ensure there's a target for CNOT
        circuit.cx(0, 1)
        circuit.rz(phi, 1) # Example Rz on target
    else: # Fallback for 1 qubit case
        circuit.rz(phi, 0)


# 2. Define the Hamiltonian
# H = 0.5*X0 + 0.75*Z1
hamiltonian = rocq.PauliOperator({"X0": 0.5, "Z1": 0.75})
# Example of a more complex Hamiltonian (though products are not yet supported by get_expval):
# hamiltonian = rocq.PauliOperator({"X0 Z1": 0.5, "Z0 X1": 0.5, "Y0 Y1": -0.2})

# 3. Define the number of qubits
NUM_QUBITS = 2

# Create a global simulator instance to avoid re-creation in objective_function
# This is important as rocsvCreate/Destroy can be relatively expensive.
# Ensure the simulator is properly managed if the script has a longer lifetime.
try:
    global_simulator = rocq.Simulator()
except Exception as e:
    print(f"Failed to initialize global ROCQuantum Simulator: {e}")
    print("Exiting VQE example.")
    exit()

# 4. Define the objective function for the optimizer
def objective_function(params):
    """
    Calculates the energy (expectation value of the Hamiltonian) for given parameters.
    """
    theta = params[0]
    phi = params[1] if len(params) > 1 else 0.0 # Handle single param case if ansatz changes

    print(f"  VQE Iteration: theta={theta:.4f}, phi={phi:.4f}", end="")

    try:
        # Build the circuit with current parameters
        # The circuit is rebuilt in each call for this simple VQE.
        # A more advanced version would update parameters in an existing program/circuit representation.
        circuit = rocq.build(vqe_ansatz_simple, NUM_QUBITS, global_simulator, theta, phi)

        # Calculate expectation value
        energy = rocq.get_expval(circuit, hamiltonian)
        print(f" -> Energy: {energy:.6f}")
        return energy
    except Exception as e:
        print(f"\nError during objective function evaluation: {e}")
        # Return a high value to steer optimizer away from problematic parameters
        return 1e6


def run_vqe():
    print("Starting Simple VQE Example...")
    print(f"Target Hamiltonian: {hamiltonian}")
    print(f"Number of Qubits: {NUM_QUBITS}")

    # Initial parameters for the ansatz
    # For vqe_ansatz_simple with Rx(theta) and Rz(phi)
    initial_params = np.array([0.5, 0.25]) # theta, phi

    print(f"Initial parameters: {initial_params}")

    # Use SciPy's minimize function
    # 'COBYLA' is a common choice for VQE as it doesn't require gradients.
    # Note: For real VQE, noise and shot noise would be factors. This is ideal simulation.
    try:
        result = minimize(objective_function,
                          initial_params,
                          method='COBYLA',
                          options={'maxiter': 50, 'disp': False}) # disp=True for more optimizer output

        print("\nOptimization Complete.")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Optimal parameters: {result.x}")
        print(f"Minimum energy found: {result.fun:.6f}")

    except NotImplementedError as nie:
        print(f"\nOptimization stopped due to a missing feature: {nie}")
        print("This likely means a backend function for a specific Pauli term in get_expval is not yet implemented.")
    except RuntimeError as rte:
        print(f"\nOptimization stopped due to a runtime error: {rte}")
        print("This could be an issue with the backend simulation or bindings.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during optimization: {e}")

    # The global_simulator's handle will be released when global_simulator object is garbage collected
    # or if we explicitly call global_simulator.release() if that method were added.
    print("\nVQE Example Finished.")


if __name__ == "__main__":
    if NUM_QUBITS > 10 and global_simulator.handle.get_num_gpus() < 2 : # Arbitrary threshold
        print(f"Warning: Simulating {NUM_QUBITS} qubits on a single GPU might be slow or run out of memory.")

    # A quick check that the backend is available for Z expectation
    # This is a bit of a hack. Ideally, the API would allow capability checks.
    try:
        print("Checking for Z expectation value support...")
        temp_circuit = rocq.Circuit(1, global_simulator)
        temp_hamiltonian = rocq.PauliOperator("Z0")
        rocq.get_expval(temp_circuit, temp_hamiltonian)
        print("Z expectation value support seems present.")
        del temp_circuit # Explicitly delete to free resources sooner
        run_vqe()
    except NotImplementedError as e:
        print(f"VQE cannot run: {e}")
    except RuntimeError as e:
         print(f"VQE setup failed with runtime error: {e}")
    finally:
        # Explicitly delete the global simulator to ensure its __del__ (and thus RocsvHandleWrapper's __del__)
        # is called before the script potentially exits and Python's garbage collection order becomes less predictable.
        # This is good practice for resource cleanup in scripts.
        if 'global_simulator' in globals():
            del global_simulator
            # print("Global simulator explicitly deleted.") # For debug

```
