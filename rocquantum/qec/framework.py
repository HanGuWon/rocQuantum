# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Core abstract classes for the rocQuantum Quantum Error Correction (QEC) framework.

This module defines the blueprints for specifying error-correcting codes,
decoders, and the experimental orchestrator. The design is fundamentally based
on a "Circuit Fragmentation" strategy.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Any

# --- rocQuantum Imports ---
import rocquantum.python.rocq as roc_q
from rocquantum.python.rocq import PauliOperator, QuantumProgram

# --- Type Hinting Definitions ---
AnsatzKernel = Callable[..., None]


class QuantumErrorCode(ABC):
    """Abstract base class for defining a quantum error-correcting code."""
    @abstractmethod
    def generate_stabilizer_circuits(
        self,
        initial_state_kernel: AnsatzKernel,
        num_qubits: int,
        simulator: roc_q.Simulator
    ) -> List[QuantumProgram]:
        """Generates the sequence of circuits for measuring each stabilizer."""
        pass

    @abstractmethod
    def define_logical_operators(self) -> Dict[str, PauliOperator]:
        """Defines the logical operators for the code."""
        pass

class Decoder(ABC):
    """Abstract base class for a QEC decoder."""
    @abstractmethod
    def decode(self, syndrome: List[int]) -> PauliOperator:
        """Processes a syndrome to determine the required correction."""
        pass

class QEC_Experiment:
    """Orchestrates a QEC experiment using a "Circuit Fragmentation" strategy."""
    def __init__(self, simulator: roc_q.Simulator):
        if not isinstance(simulator, roc_q.Simulator):
            raise TypeError("A valid roc_q.Simulator instance is required.")
        self.simulator = simulator

    def run_single_round(
        self,
        code: QuantumErrorCode,
        decoder: Decoder,
        initial_state_kernel: AnsatzKernel,
        num_qubits: int,
        ancilla_qubit_indices: List[int]
    ) -> Dict[str, Any]:
        """Executes a single, complete round of quantum error correction."""
        print("Step 1: Generating stabilizer measurement circuit fragments...")
        stabilizer_circuits = code.generate_stabilizer_circuits(
            initial_state_kernel, num_qubits, self.simulator
        )

        print("Step 2: Measuring syndrome by executing each fragment...")
        syndrome = []
        for i, stab_program in enumerate(stabilizer_circuits):
            ancilla_idx = ancilla_qubit_indices[i]
            outcome, _ = stab_program.circuit_ref.measure(ancilla_idx)
            print(f"  - Measured stabilizer {i} on ancilla q[{ancilla_idx}]: outcome = {outcome}")
            syndrome.append(outcome)

        print(f"\nStep 3: Decoding syndrome {syndrome}...")
        correction_operator = decoder.decode(syndrome)
        print(f"  - Decoder determined correction: {correction_operator}")

        # Note: Final state calculation can be added here if needed.
        # For now, the primary goal is verifying the syndrome and correction.

        return {
            "syndrome": syndrome,
            "correction_applied": str(correction_operator),
            "logical_operators": code.define_logical_operators(),
        }
