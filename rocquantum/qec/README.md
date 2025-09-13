# rocQuantum QEC Framework

## Architectural Overview

This document details the Quantum Error Correction (QEC) framework within rocQuantum. The framework is designed to provide a flexible and extensible platform for simulating QEC codes.

### The "Circuit Fragmentation" Strategy

A core design principle of this framework is the **"Circuit Fragmentation"** strategy. Due to the current rocQuantum API's lack of native support for mid-circuit measurement with classical feedback (i.e., conditional operations based on measurement outcomes), a single simulation run cannot perform a full QEC cycle.

To overcome this, our framework treats one round of error correction as a sequence of separate, independent circuit simulations orchestrated by classical Python code. The workflow is as follows:

1.  A circuit fragment is generated for each stabilizer measurement. Each fragment is a complete simulation that prepares an initial state, applies the stabilizer measurement gates, and measures a single ancilla qubit.
2.  The `QEC_Experiment` class executes each of these fragments sequentially, collecting the classical measurement outcomes.
3.  The collected outcomes form a classical "syndrome" string.
4.  This syndrome is passed to a `Decoder`, which uses classical logic to determine the most likely error and the required correction.

This approach, while less performant than a native implementation, provides a powerful and fully general way to simulate any QEC code on the existing rocQuantum backend.

### Core Classes

-   **`QuantumErrorCode`**: An abstract base class that defines the structure of a specific QEC code. Its primary responsibility is to generate the list of circuit fragments needed for the stabilizer measurements.
-   **`Decoder`**: An abstract base class for a classical decoder. Its role is to take a syndrome (e.g., `[1, 0, 1]`) and return the corresponding correction operator (e.g., `PauliOperator({"X1": 1.0})`).
-   **`QEC_Experiment`**: The main orchestrator class. It manages the end-to-end workflow, executing the circuit fragments, collecting the syndrome, invoking the decoder, and reporting the results.

## Basic Usage

The following example demonstrates a full workflow for detecting and identifying an error in a 3-qubit repetition code.

```python
# --- 1. Setup ---
import rocquantum.python.rocq as roc_q
from rocquantum.qec.framework import QEC_Experiment
from rocquantum.qec.codes.repetition_code import ThreeQubitRepetitionCode
from rocquantum.qec.decoders.repetition_decoder import RepetitionCodeDecoder

# Initialize the simulator and the QEC components
sim = roc_q.Simulator()
code = ThreeQubitRepetitionCode()
decoder = RepetitionCodeDecoder()
experiment = QEC_Experiment(simulator=sim)

NUM_QUBITS = 5 # 3 data + 2 ancilla
ANCILLA_INDICES = [3, 4]

# --- 2. Define Initial State with an Injected Error ---
# We define a kernel that prepares a logical |+> state and then
# manually injects an X error on data qubit 1 for testing.
@roc_q.kernel
def error_state_kernel(q):
    # Prepare logical |+> state: H(q0), CX(q0,q1), CX(q0,q2)
    q.h(0)
    q.cx(0, 1)
    q.cx(0, 2)
    # Inject error
    q.x(1)

# --- 3. Run the Experiment ---
# The orchestrator runs the fragments, gets the syndrome, and decodes it.
results = experiment.run_single_round(
    code=code,
    decoder=decoder,
    initial_state_kernel=error_state_kernel,
    num_qubits=NUM_QUBITS,
    ancilla_qubit_indices=ANCILLA_INDICES
)

# --- 4. Verify Results ---
print(f"Measured Syndrome: {results['syndrome']}")
print(f"Decoded Correction: {results['correction_applied']}")

# For an X error on qubit 1, the expected syndrome is [1, 1]
assert results['syndrome'] == [1, 1]
```

## Extensibility Guide

### How to Add a New Error-Correcting Code

To add a new code (e.g., the 5-qubit code), you must implement two new classes: one for the code definition and one for its corresponding decoder.

**Step 1: Implement the `QuantumErrorCode`**

Create a new class that inherits from `QuantumErrorCode` and implement its abstract methods.

```python
# in file: rocquantum/qec/codes/five_qubit_code.py
from rocquantum.qec.framework import QuantumErrorCode
# ... other imports

class FiveQubitCode(QuantumErrorCode):
    def generate_stabilizer_circuits(self, initial_state_kernel, num_qubits, simulator):
        # Implement the logic to generate the 4 stabilizer measurement
        # circuit fragments for the 5-qubit code.
        # Return a list of 4 roc_q.QuantumProgram objects.
        stabilizer_programs = []
        # ... logic to build and append each program
        return stabilizer_programs

    def define_logical_operators(self):
        # Return the logical X and Z operators for the 5-qubit code.
        return {
            "logical_X": PauliOperator({"X0 X1 X2 X3 X4": 1.0}), # Example
            "logical_Z": PauliOperator({"Z0 Z1 Z2 Z3 Z4": 1.0})  # Example
        }
```

**Step 2: Implement the `Decoder`**

Create a corresponding decoder class that inherits from `Decoder`.

```python
# in file: rocquantum/qec/decoders/five_qubit_decoder.py
from rocquantum.qec.framework import Decoder
# ... other imports

class FiveQubitDecoder(Decoder):
    def decode(self, syndrome: List[int]) -> PauliOperator:
        # The syndrome will be a list of 4 bits.
        # Implement the logic to map each of the 16 possible syndromes
        # to the appropriate 1-qubit Pauli correction (or Identity).
        if syndrome == [0, 0, 0, 0]:
            return PauliOperator() # No error
        elif syndrome == [1, 0, 0, 1]:
            return PauliOperator({"X0": 1.0}) # Example for X error on qubit 0
        # ... other 14 cases
```