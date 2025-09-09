import numpy as np
import uuid
from collections import Counter

from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit.result import Result, ExperimentResult, ExperimentResultData
from qiskit.circuit import Measure
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate, HGate, UnitaryGate

from rocquantum_bind import QuantumSimulator

class RocQuantumBackend(BackendV2):
    """
    rocQuantum Qiskit Backend.

    A Qiskit backend that interfaces with the rocQuantum C++/HIP simulator.
    """
    def __init__(self, provider, **kwargs):
        super().__init__(provider=provider, name="rocq_simulator", **kwargs)
        
        # The simulator is instantiated once and maintained for the lifetime of the backend
        # We assume a fixed, maximum number of qubits or re-instantiate if needed.
        # For simplicity, let's assume it's configured on first use.
        self._simulator = None
        self._num_qubits = 0

        # Define the target with supported gates
        self.target = Target()
        self.target.add_instruction(RXGate, name="rx")
        self.target.add_instruction(RYGate, name="ry")
        self.target.add_instruction(RZGate, name="rz")
        self.target.add_instruction(CXGate, name="cx") # CNOT
        self.target.add_instruction(HGate, name="h")
        self.target.add_instruction(UnitaryGate, name="unitary")
        self.target.add_instruction(Measure, name="measure")

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def _ensure_simulator(self, num_qubits):
        """Create or reset the simulator if the qubit count changes."""
        if self._simulator is None or self._num_qubits != num_qubits:
            self._simulator = QuantumSimulator(num_qubits)
            self._num_qubits = num_qubits
        else:
            self._simulator.reset()

    def run(self, run_input, **options):
        if not isinstance(run_input, list):
            run_input = [run_input]

        job_id = str(uuid.uuid4())
        shots = options.get("shots", self.options.shots)
        results = []

        for circuit in run_input:
            self._ensure_simulator(circuit.num_qubits)
            
            # Simplified translation loop
            measured_bits = {} # Map classical bit index to qubit index
            for instruction in circuit.data:
                op = instruction.operation
                q_indices = [circuit.find_bit(q).index for q in instruction.qubits]
                
                if op.name in ('rx', 'ry', 'rz'):
                    self._simulator.apply_gate(op.name.upper(), q_indices, op.params)
                elif op.name in ('cx', 'h'):
                    self._simulator.apply_gate(op.name.upper(), q_indices, [])
                elif op.name == 'unitary':
                    self._simulator.apply_matrix(op.to_matrix(), q_indices)
                elif op.name == 'measure':
                    # Store which classical bit this measurement corresponds to
                    c_index = circuit.find_bit(instruction.clbits[0]).index
                    measured_bits[c_index] = q_indices[0]

            # Perform measurement on the required qubits
            qubits_to_measure = list(measured_bits.values())
            if not qubits_to_measure: # If no measure op, measure all
                qubits_to_measure = list(range(circuit.num_qubits))

            raw_samples = self._simulator.measure(qubits_to_measure, shots)
            
            # Format results into Qiskit's desired structure
            counts = Counter(raw_samples)
            # Format keys as binary strings '0b...'
            formatted_counts = {bin(k): v for k, v in counts.items()}

            exp_data = ExperimentResultData(
                counts=formatted_counts,
                memory=[bin(s) for s in raw_samples] # Memory stores each shot result
            )
            
            exp_result = ExperimentResult(
                shots=shots,
                success=True,
                data=exp_data,
                header=circuit.header
            )
            results.append(exp_result)

        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            job_id=job_id,
            qobj_id=None,
            success=True,
            results=results
        )