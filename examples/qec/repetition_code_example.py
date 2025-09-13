# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
repetition_code_example.py

This script demonstrates the usage of the rocQuantum QEC framework by running
a full error correction cycle for the 3-qubit bit-flip repetition code.
"""

# --- rocQuantum Imports ---
import rocquantum.python.rocq as roc_q
from rocquantum.qec.framework import QEC_Experiment
from rocquantum.qec.codes.repetition_code import ThreeQubitRepetitionCode
from rocquantum.qec.decoders.repetition_decoder import RepetitionCodeDecoder


def run_repetition_code_example():
    """
    Executes the main QEC demonstration.
    """
    # 1. Setup
    print("--- Setting up QEC Experiment ---")
    try:
        sim = roc_q.Simulator()
        print("Simulator initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize simulator: {e}")
        return

    code = ThreeQubitRepetitionCode()
    decoder = RepetitionCodeDecoder()
    experiment = QEC_Experiment(simulator=sim)

    NUM_QUBITS = 5  # 3 data + 2 ancilla
    ANCILLA_INDICES = [3, 4]

    # 2. Define Initial State (Logical |+>)
    # The logical |+> state is an equal superposition of |000> and |111>.
    @roc_q.kernel
    def logical_plus_state_kernel(q):
        q.h(0)
        q.cx(0, 1)
        q.cx(0, 2)

    # 3. Inject an Error
    # We create a new kernel that prepares the logical state and then
    # introduces a manual bit-flip error on data qubit 1.
    print("\n--- Test Case: Injecting X error on data qubit 1 ---")
    @roc_q.kernel
    def error_state_kernel(q):
        # First, prepare the ideal state
        logical_plus_state_kernel(q)
        # Then, inject the error
        print("Injecting a manual X error on qubit 1...")
        q.x(1)

    # 4. Run Experiment
    # We run the experiment using the kernel that contains the injected error.
    print("\n--- Running QEC Round ---")
    results = experiment.run_single_round(
        code=code,
        decoder=decoder,
        initial_state_kernel=error_state_kernel,
        num_qubits=NUM_QUBITS,
        ancilla_qubit_indices=ANCILLA_INDICES
    )

    # 5. Verify Results
    print("\n--- Verification ---")
    print(f"Measured Syndrome: {results['syndrome']}")
    print(f"Decoded Correction: {results['correction_applied']}")

    expected_syndrome = [1, 1]
    expected_correction_str = str(roc_q.PauliOperator({"X1": 1.0}))

    assert results['syndrome'] == expected_syndrome, \
        f"Syndrome check FAILED! Expected {expected_syndrome}, got {results['syndrome']}"
    print("Syndrome check PASSED!")

    assert results['correction_applied'] == expected_correction_str, \
        f"Correction check FAILED! Expected {expected_correction_str}, got {results['correction_applied']}"
    print("Correction check PASSED!")

    print("\nQEC MVP example completed successfully.")


if __name__ == '__main__':
    run_repetition_code_example()