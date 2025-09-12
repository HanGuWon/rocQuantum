# rocquantum/circuit.py

"""
This module provides a high-level, Pythonic interface for building quantum
circuits, abstracting away the need for users to write raw OpenQASM strings.
"""

from typing import List, Tuple, Union, Any

class QuantumCircuit:
    """
    A class for programmatically constructing a quantum circuit.

    This class provides an intuitive API for adding quantum gates and
    measurements. It can then compile this sequence of operations into a
    valid OpenQASM 3.0 string representation.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes a new quantum circuit.

        Args:
            num_qubits (int): The number of qubits in the circuit.

        Raises:
            ValueError: If the number of qubits is not a positive integer.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        self.num_qubits = num_qubits
        self._operations: List[Tuple[str, Any]] = []
        self._measured = False

    def _validate_qubit_index(self, *indices):
        """Checks if qubit indices are valid for this circuit."""
        for index in indices:
            if not (0 <= index < self.num_qubits):
                raise ValueError(
                    f"Qubit index {index} is out of bounds for a circuit with "
                    f"{self.num_qubits} qubits."
                )

    def h(self, qubit_index: int):
        """Applies a Hadamard gate to a specific qubit."""
        self._validate_qubit_index(qubit_index)
        self._operations.append(('h', qubit_index))

    def x(self, qubit_index: int):
        """Applies a Pauli-X (NOT) gate to a specific qubit."""
        self._validate_qubit_index(qubit_index)
        self._operations.append(('x', qubit_index))

    def cx(self, control_index: int, target_index: int):
        """Applies a Controlled-X (CNOT) gate."""
        if control_index == target_index:
            raise ValueError("Control and target qubits cannot be the same.")
        self._validate_qubit_index(control_index, target_index)
        self._operations.append(('cx', (control_index, target_index)))

    def measure_all(self):
        """Adds a measurement operation for all qubits."""
        if self._measured:
            raise ValueError("Measurement has already been added to this circuit.")
        self._operations.append(('measure_all', None))
        self._measured = True

    def to_qasm(self) -> str:
        """
        Generates a valid OpenQASM 3.0 string from the circuit's operations.

        Returns:
            str: The OpenQASM 3.0 representation of the circuit.
        """
        if not self._measured:
            # Automatically add measurement if the user hasn't explicitly.
            self.measure_all()

        qasm_lines = [
            "OPENQASM 3.0;",
            f"qubit[{self.num_qubits}] q;",
            f"bit[{self.num_qubits}] c;",
        ]

        for op_name, op_args in self._operations:
            if op_name == 'h':
                qasm_lines.append(f"h q[{op_args}];")
            elif op_name == 'x':
                qasm_lines.append(f"x q[{op_args}];")
            elif op_name == 'cx':
                control, target = op_args
                qasm_lines.append(f"cx q[{control}], q[{target}];")
            elif op_name == 'measure_all':
                qasm_lines.append(f"c = measure q;")

        return "\n".join(qasm_lines)
