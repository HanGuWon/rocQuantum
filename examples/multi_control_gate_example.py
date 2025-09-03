import numpy as np
import rocq

def run_multi_control_gate_example():
    """
    Demonstrates the use of the generic multi-controlled unitary gate application.
    We will implement a Toffoli (CCX) gate and verify its behavior.
    """
    num_qubits = 3
    print(f"Initializing a {num_qubits}-qubit simulator.")

    # Pauli-X matrix
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)

    # Test cases: initial state -> expected final state (integer representation)
    test_cases = {
        "|011>": {"initial_state_idx": 3, "expected_final_idx": 3}, # 011 -> 011 (control 0 is 0)
        "|101>": {"initial_state_idx": 5, "expected_final_idx": 5}, # 101 -> 101 (control 1 is 0)
        "|110>": {"initial_state_idx": 6, "expected_final_idx": 7}, # 110 -> 111 (controls are 1, target flips)
        "|111>": {"initial_state_idx": 7, "expected_final_idx": 6}  # 111 -> 110 (controls are 1, target flips)
    }

    try:
        for name, case in test_cases.items():
            print(f"\n--- Testing case: {name} ---")
            simulator = rocq.Simulator()
            circuit = rocq.Circuit(num_qubits, simulator)

            # Prepare the initial state by flipping the appropriate bits from |000>
            initial_idx = case["initial_state_idx"]
            for i in range(num_qubits):
                if (initial_idx >> i) & 1:
                    circuit.x(i)
            
            # Define the CCX gate: controls on qubits 0 and 1, target on 2
            control_qubits = [0, 1]
            target_qubits = [2]
            
            print(f"Applying controlled-X with controls={control_qubits} and target={target_qubits}")
            circuit.apply_controlled_unitary(control_qubits, target_qubits, pauli_x)

            # To verify the result, we measure all qubits.
            # For a deterministic outcome, we can just measure.
            # The state should be collapsed to a single computational basis state.
            
            # We find which basis state has probability 1.0
            final_state_idx = -1
            for i in range(1 << num_qubits):
                # This is not efficient, but good for verification.
                # A get_state_vector() method would be ideal here.
                # We can create a temporary copy to measure without destroying the state for all checks.
                temp_sim = rocq.Simulator()
                temp_circuit = rocq.Circuit(num_qubits, temp_sim)
                
                # Re-create the state in the temp circuit
                if (initial_idx >> 0) & 1: temp_circuit.x(0)
                if (initial_idx >> 1) & 1: temp_circuit.x(1)
                if (initial_idx >> 2) & 1: temp_circuit.x(2)
                temp_circuit.apply_controlled_unitary(control_qubits, target_qubits, pauli_x)

                # Measure qubit 0
                outcome, prob = temp_circuit.measure(0)
                if np.isclose(prob, 1.0):
                    # If we are certain about the outcome of qubit 0, we check the rest.
                    # This is getting complicated. A simpler way is to sample.
                    pass # Pass for now, will use sampling below.

            # A better way to verify is to sample and see if we get one outcome 100% of the time.
            results = circuit.sample([0, 1, 2], 100)
            counts = Counter(results)
            final_state_idx = list(counts.keys())[0]
            num_outcomes = len(counts)

            print(f"Sampling result: Got outcome {final_state_idx} with 100% probability.")
            expected_idx = case["expected_final_idx"]
            print(f"Expected final state index: {expected_idx}")

            assert num_outcomes == 1, f"Verification FAILED: Expected a single outcome, but got {num_outcomes}"
            assert final_state_idx == expected_idx, f"Verification FAILED: Expected state {expected_idx}, but got {final_state_idx}"
            print("Verification PASSED.")

    except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the rocQuantum project is built and installed correctly.")

if __name__ == "__main__":
    run_multi_control_gate_example()
