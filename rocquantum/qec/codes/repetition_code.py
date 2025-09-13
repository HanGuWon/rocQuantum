# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Concrete implementation of the 3-qubit bit-flip repetition code.
"""

from typing import List, Dict, Callable

import rocquantum.python.rocq as roc_q
from rocquantum.python.rocq import PauliOperator, QuantumProgram
from rocquantum.qec.framework import QuantumErrorCode

# Type Hinting
AnsatzKernel = Callable[..., None]

class ThreeQubitRepetitionCode(QuantumErrorCode):
    """
    Implements the 3-qubit bit-flip repetition code.

    This code uses 3 data qubits and 2 ancilla qubits.
    - Data Qubits: 0, 1, 2
    - Ancilla Qubits: 3, 4 (by convention in the example)
    """
    def generate_stabilizer_circuits(self,
                                     initial_state_kernel: AnsatzKernel,
                                     num_qubits: int,
                                     simulator: roc_q.Simulator) -> List[QuantumProgram]:
        """Generates one circuit for the Z0Z1 stabilizer and one for Z1Z2."""
        programs = []

        # --- Circuit 1: Measure Z0Z1 stabilizer on the first ancilla ---
        @roc_q.kernel
        def z0z1_stabilizer_kernel(q):
            initial_state_kernel(q)
            # Measurement circuit for Z0Z1 (ancilla at qubit 3)
            q.h(3)
            q.cx(0, 3)
            q.cx(1, 3)
            q.h(3)

        prog1 = roc_q.build(z0z1_stabilizer_kernel, num_qubits, simulator)
        programs.append(prog1)

        # --- Circuit 2: Measure Z1Z2 stabilizer on the second ancilla ---
        @roc_q.kernel
        def z1z2_stabilizer_kernel(q):
            initial_state_kernel(q)
            # Measurement circuit for Z1Z2 (ancilla at qubit 4)
            q.h(4)
            q.cx(1, 4)
            q.cx(2, 4)
            q.h(4)

        prog2 = roc_q.build(z1z2_stabilizer_kernel, num_qubits, simulator)
        programs.append(prog2)

        return programs

    def define_logical_operators(self) -> Dict[str, PauliOperator]:
        """Returns the logical Z and logical X operators."""
        return {
            "logical_Z": PauliOperator({"Z0": 1.0}),
            "logical_X": PauliOperator({"X0 X1 X2": 1.0})
        }