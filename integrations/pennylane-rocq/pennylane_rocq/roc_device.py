import numpy as np
from pennylane import Device
from rocquantum_bind import QuantumSimulator

class RocqDevice(Device):
    """
    rocQuantum PennyLane Device.

    A PennyLane device that interfaces with the rocQuantum C++/HIP simulator.
    """
    name = "rocQuantum PennyLane Device"
    short_name = "rocq.pennylane"
    author = "Gemini"
    version = "0.1.0"

    # Define the supported operations and observables
    operations = {
        "Identity", "Hadamard", "PauliX", "PauliY", "PauliZ",
        "CNOT", "RX", "RY", "RZ", "QubitUnitary"
    }
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"}

    def __init__(self, wires, shots=1024):
        super().__init__(wires=wires, shots=shots)
        # The simulator is instantiated once and maintained for the lifetime of the device
        self._simulator = QuantumSimulator(len(self.wires))
        self._is_first_run = True

    def reset(self):
        """
        Reset the underlying simulator to the |0...0> state.
        """
        if not self._is_first_run:
            self._simulator.reset()

    def apply(self, operations, **kwargs):
        """
        Apply a sequence of operations to the simulator.
        """
        # Reset the simulator for the new circuit execution
        self.reset()
        self._is_first_run = False

        for op in operations:
            gate_name = op.name
            targets = op.wires.tolist()
            params = op.parameters

            if gate_name == "QubitUnitary":
                matrix = params[0]
                self._simulator.apply_matrix(matrix, targets)
            else:
                # Map PennyLane names to potential C++ names if they differ
                # For now, assume they are the same (e.g., "RX", "CNOT")
                self._simulator.apply_gate(gate_name, targets, params)

    def generate_samples(self):
        """
        Generate samples from the simulator after the circuit has been executed.
        """
        # This default implementation assumes measurement of all wires.
        # For more complex measurement logic, this method needs to be expanded.
        all_wires = list(range(len(self.wires)))
        samples = self._simulator.measure(all_wires, self.shots)
        return np.array(samples).reshape(self.shots, len(all_wires))

    @property
    def state(self):
        """
        Return the state vector from the simulator.
        """
        return self._simulator.get_statevector()
