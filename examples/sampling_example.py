import numpy as np
import rocq
from collections import Counter

def create_bell_state(circuit):
    """Creates a 2-qubit Bell state |00> + |11> / sqrt(2)."""
    circuit.h(0)
    circuit.cx(0, 1)

def run_sampling_example():
    """
    Demonstrates the use of the state vector sampling feature.
    """
    num_qubits = 2
    num_shots = 1000
    print(f"Initializing a {num_qubits}-qubit simulator.")

    try:
        # 1. Initialize the simulator and circuit
        simulator = rocq.Simulator()
        circuit = rocq.Circuit(num_qubits, simulator)

        # 2. Prepare a Bell state
        print("Preparing a 2-qubit Bell state.")
        create_bell_state(circuit)

        # 3. Sample from the state vector
        measured_qubits = [0, 1]
        print(f"\nSampling {num_shots} shots from qubits {measured_qubits}...")
        
        # The sample method returns an array of integer bitstrings
        # For measured_qubits=[0, 1], bit 0 is from qubit 0, bit 1 from qubit 1
        # e.g., outcome 3 (binary 11) means qubit 0 was 1, qubit 1 was 1.
        results = circuit.sample(measured_qubits, num_shots)

        # 4. Analyze and print the results
        counts = Counter(results)
        print("\n--- Measurement Counts ---")
        print(f"Outcome '00' (int 0): {counts.get(0, 0)} times")
        print(f"Outcome '01' (int 1): {counts.get(1, 0)} times")
        print(f"Outcome '10' (int 2): {counts.get(2, 0)} times")
        print(f"Outcome '11' (int 3): {counts.get(3, 0)} times")

        # 5. Verification
        print("\n--- Verification ---")
        # For a perfect Bell state, we should only get outcomes 0 (00) and 3 (11)
        has_only_bell_outcomes = (counts.get(1, 0) == 0 and counts.get(2, 0) == 0)
        if has_only_bell_outcomes:
            print("Verification PASSED: Only correlated outcomes (00 and 11) were observed.")
        else:
            print("Verification FAILED: Uncorrelated outcomes (01 or 10) were observed.")
        
        prob_00 = counts.get(0, 0) / num_shots
        prob_11 = counts.get(3, 0) / num_shots
        print(f"Observed P(00) = {prob_00:.4f} (Expected: ~0.5)")
        print(f"Observed P(11) = {prob_11:.4f} (Expected: ~0.5)")
        assert has_only_bell_outcomes
        assert np.isclose(prob_00, 0.5, atol=0.05), "Probability of 00 is not close to 0.5"
        assert np.isclose(prob_11, 0.5, atol=0.05), "Probability of 11 is not close to 0.5"

    except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the rocQuantum project is built and installed correctly.")

if __name__ == "__main__":
    run_sampling_example()
