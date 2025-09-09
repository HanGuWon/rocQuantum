# qiskit-rocquantum-provider/qiskit_rocquantum_provider/backend.py
import uuid
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit.result import Result
from qiskit.circuit import Measure
from qiskit.circuit.library import (
    XGate, YGate, ZGate, HGate, SGate, TGate, CNOTGate, CZGate,
    RXGate, RYGate, RZGate, UnitaryGate, SaveStatevector
)
from qiskit.providers.job import JobV1, JobStatus

try:
    import rocquantum_bind
except ImportError:
    raise ImportError("The 'rocquantum_bind' module is not installed.")

QISKIT_TO_ROCQ_GATES = {
    "x": "X", "y": "Y", "z": "Z", "h": "H",
    "s": "S", "t": "T", "cx": "CNOT", "cz": "CZ",
}

class RocQuantumJob(JobV1):
    def __init__(self, backend, job_id, fn, *args):
        super().__init__(backend, job_id)
        self._future = ThreadPoolExecutor(max_workers=1).submit(fn, *args)
    def result(self): return self._future.result()
    def status(self): return JobStatus.DONE if self._future.done() else JobStatus.RUNNING
    def submit(self): raise NotImplementedError

class RocQuantumBackend(BackendV2):
    def __init__(self, **kwargs):
        super().__init__(name="rocquantum_simulator", **kwargs)
        self._target = self._create_target()

    @property
    def target(self): return self._target
    @property
    def max_circuits(self): return 1
    @classmethod
    def _default_options(cls): return Options(shots=1024)

    def _create_target(self):
        target = Target(description="rocQuantum Simulator Target")
        qubit_range = range(30)
        # Add gates
        for i in qubit_range:
            for gate in [XGate, YGate, ZGate, HGate, SGate, TGate, RXGate, RYGate, RZGate, UnitaryGate]:
                target.add_instruction(gate, {(i,): None})
            for j in qubit_range:
                if i != j:
                    target.add_instruction(CNOTGate, {(i, j): None})
                    target.add_instruction(CZGate, {(i, j): None})
        target.add_instruction(SaveStatevector)
        target.add_instruction(Measure)
        return target

    def _run_circuit(self, circuit, shots):
        sim = rocquantum_bind.QSim(num_qubits=circuit.num_qubits)
        saves_statevector = False
        for instruction in circuit.data:
            op, qargs = instruction.operation, instruction.qubits
            q_indices = [circuit.find_bit(q).index for q in qargs]
            if op.name in QISKIT_TO_ROCQ_GATES:
                sim.ApplyGate(QISKIT_TO_ROCQ_GATES[op.name], *q_indices)
            elif op.name in ["rx", "ry", "rz", "unitary"]:
                sim.ApplyGate(op.to_matrix(), q_indices[0])
            elif isinstance(op, SaveStatevector):
                saves_statevector = True
            elif not isinstance(op, Measure):
                raise NotImplementedError(f"Unsupported op: {op.name}")
        sim.Execute()
        statevector = sim.GetStateVector()
        data = {}
        if saves_statevector: data["statevector"] = statevector
        if circuit.num_clbits > 0:
            probs = np.abs(statevector)**2
            outcomes = np.random.choice(len(probs), size=shots, p=probs)
            counts = Counter(outcomes)
            data["counts"] = {f"{k:0{circuit.num_qubits}b}": v for k, v in counts.items()}
        return {"success": True, "data": data, "metadata": {}}

    def run(self, run_input, **options):
        circuit = run_input
        job_id = str(uuid.uuid4())
        shots = options.get("shots", self.options.shots)
        job_fn = lambda: Result.from_dict({
            "backend_name": self.name, "backend_version": "0.1.0",
            "job_id": job_id, "qobj_id": circuit.name, "success": True,
            "results": [self._run_circuit(circuit, shots)],
        })
        return RocQuantumJob(self, job_id, job_fn)
