import numpy as np
from . import _rocq_hip_backend as backend # Assuming the compiled module is named this

class Simulator:
    """
    Manages the hipStateVec simulation backend handle.
    A Simulator instance is required to create and run circuits.
    """
    def __init__(self):
        try:
            self._handle_wrapper = backend.RocsvHandle()
            self._active_circuits = 0 # Basic tracking of active circuits using this sim
        except RuntimeError as e:
            print(f"Failed to initialize rocQuantum Simulator: {e}")
            print("Please ensure the ROCm environment is set up correctly and the rocQuantum backend libraries are found.")
            raise

    @property
    def handle(self):
        if self._handle_wrapper is None:
            raise RuntimeError("Simulator handle is not initialized or has been released.")
        return self._handle_wrapper # Corrected: self._handle_wrapper

    # In a more complex scenario, if the handle could be explicitly released:
    # def release(self):
    #     if self._handle_wrapper:
    #         # The C++ RocsvHandleWrapper destructor handles rocsvDestroy
    #         self._handle_wrapper = None 
    #         print("rocQuantum Simulator handle released.")

    def __del__(self):
        # The C++ RocsvHandleWrapper's destructor will be called automatically
        # when the _handle_wrapper object is garbage collected if not explicitly released.
        # print("Simulator instance being deleted.") # For debug
        pass

    def create_device_matrix(self, numpy_matrix: np.ndarray) -> backend.DeviceBuffer:
        """
        Creates a device buffer and copies a NumPy matrix to it.
        The matrix should be of type np.complex64.
        """
        if not isinstance(numpy_matrix, np.ndarray):
            raise TypeError("Input matrix must be a NumPy array.")
        
        # Ensure the numpy array is of type rocComplex (complex64) and C-contiguous
        # Pybind11's array_t<rocComplex, py::array::c_style | py::array::forcecast> handles casting.
        # However, it's good practice to ensure dtype.
        if numpy_matrix.dtype != np.complex64:
            # print("Warning: Input matrix dtype is not np.complex64. Attempting to cast.") # Optional warning
            numpy_matrix = numpy_matrix.astype(np.complex64, order='C') # order='C' for c_style

        if not numpy_matrix.flags['C_CONTIGUOUS']:
            # print("Warning: Input matrix is not C-contiguous. Making a contiguous copy.") # Optional warning
            numpy_matrix = np.ascontiguousarray(numpy_matrix, dtype=np.complex64)

        return backend.create_device_matrix_from_numpy(numpy_matrix)


class Circuit:
    """
    Represents a quantum circuit and provides methods to build and simulate it
    using the hipStateVec backend.
    """
    def __init__(self, num_qubits: int, simulator: Simulator):
        if not isinstance(simulator, Simulator):
            raise TypeError("A valid Simulator instance is required.")
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")

        self.num_qubits = num_qubits
        self.simulator = simulator
        self._sim_handle = simulator._handle_wrapper # Get the C++ handle wrapper

        try:
            # allocate_state_internal returns an owning DeviceBuffer
            self._d_state_buffer = backend.allocate_state_internal(self._sim_handle, self.num_qubits)
            status = backend.initialize_state(self._sim_handle, self._d_state_buffer, self.num_qubits)
            if status != backend.rocqStatus.SUCCESS:
                raise RuntimeError(f"Failed to initialize state: {status}")
            self.simulator._active_circuits +=1
        except RuntimeError as e:
            # If d_state_buffer was allocated but initialize failed, it will be auto-freed by DeviceBuffer destructor
            # if self._d_state_buffer was assigned.
            print(f"Error during Circuit initialization: {e}")
            raise
            
    def __del__(self):
        # The self._d_state_buffer (DeviceBuffer) will automatically call hipFree 
        # in its C++ destructor when this Circuit object is garbage collected.
        if hasattr(self, 'simulator') and self.simulator is not None and hasattr(self.simulator, '_active_circuits'):
             self.simulator._active_circuits -=1
        # print(f"Circuit for {self.num_qubits} qubits being deleted.") # For debug

    def _validate_qubit_index(self, qubit_index, name="target qubit"):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            raise ValueError(
                f"{name} index {qubit_index} is out of range for {self.num_qubits} qubits."
            )

    def _validate_control_target(self, control_qubit, target_qubit):
        self._validate_qubit_index(control_qubit, "control qubit")
        self._validate_qubit_index(target_qubit, "target qubit")
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same.")

    # --- Single-Qubit Gates ---
    def x(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_x(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply X failed: {status}")

    def y(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_y(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Y failed: {status}")

    def z(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_z(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Z failed: {status}")

    def h(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_h(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply H failed: {status}")

    def s(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_s(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply S failed: {status}")

    def t(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_t(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply T failed: {status}")

    def rx(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_rx(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit, angle)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Rx failed: {status}")

    def ry(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_ry(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit, angle)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Ry failed: {status}")

    def rz(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        status = backend.apply_rz(self._sim_handle, self._d_state_buffer, self.num_qubits, target_qubit, angle)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Rz failed: {status}")

    # --- Two-Qubit Gates ---
    def cx(self, control_qubit: int, target_qubit: int): # CNOT
        self._validate_control_target(control_qubit, target_qubit)
        status = backend.apply_cnot(self._sim_handle, self._d_state_buffer, self.num_qubits, control_qubit, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply CNOT failed: {status}")

    def cz(self, qubit1: int, qubit2: int):
        self._validate_control_target(qubit1, qubit2) # Same validation applies
        status = backend.apply_cz(self._sim_handle, self._d_state_buffer, self.num_qubits, qubit1, qubit2)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply CZ failed: {status}")

    def swap(self, qubit1: int, qubit2: int):
        self._validate_control_target(qubit1, qubit2) # Same validation applies
        status = backend.apply_swap(self._sim_handle, self._d_state_buffer, self.num_qubits, qubit1, qubit2)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply SWAP failed: {status}")

    # --- Generic Unitary ---
    def apply_unitary(self, qubit_indices: list[int], matrix: np.ndarray):
        """
        Applies a generic unitary matrix to the specified qubits.
        
        Args:
            qubit_indices (list[int]): A list of target qubit indices.
            matrix (np.ndarray): A NumPy array representing the unitary matrix.
                                 Shape must be (2^m, 2^m) where m = len(qubit_indices).
                                 dtype should be np.complex64.
        """
        num_target_qubits = len(qubit_indices)
        if num_target_qubits == 0:
            raise ValueError("qubit_indices cannot be empty for apply_unitary.")
        for idx in qubit_indices:
            self._validate_qubit_index(idx, f"qubit_indices element {idx}")
        # Check for duplicate qubit indices
        if len(set(qubit_indices)) != num_target_qubits:
            raise ValueError("Duplicate qubit indices are not allowed.")

        expected_dim = 1 << num_target_qubits
        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Matrix shape {matrix.shape} is not valid for {num_target_qubits} qubits. "
                f"Expected ({expected_dim}, {expected_dim})."
            )

        # Python side creates a device matrix using the Simulator's helper
        device_matrix_buffer = self.simulator.create_device_matrix(matrix)
        
        try:
            status = backend.apply_matrix(
                self._sim_handle, 
                self._d_state_buffer, 
                self.num_qubits,
                qubit_indices,          # Pybind11 should handle list[int] to std::vector<unsigned>
                device_matrix_buffer,   # Pass the DeviceBuffer object
                expected_dim
            )
            if status != backend.rocqStatus.SUCCESS:
                raise RuntimeError(f"Apply Matrix failed: {status}")
        finally:
            # device_matrix_buffer will be auto-freed by its C++ destructor
            # when it goes out of scope in Python if it was created here.
            # No explicit free needed if using the RAII DeviceBuffer from bindings.
            pass


    # --- Measurement ---
    def measure(self, qubit_to_measure: int) -> tuple[int, float]:
        """
        Measures a single qubit in the computational basis.
        This operation collapses the state vector.

        Args:
            qubit_to_measure (int): The index of the qubit to measure.

        Returns:
            tuple[int, float]: A tuple containing the measurement outcome (0 or 1)
                               and the probability of that outcome.
        """
        self._validate_qubit_index(qubit_to_measure)
        try:
            outcome, probability = backend.measure(
                self._sim_handle, self._d_state_buffer, self.num_qubits, qubit_to_measure
            )
            return outcome, probability
        except RuntimeError as e:
            raise RuntimeError(f"Measure failed: {e}")

    # def get_probabilities(self): # Future enhancement
    #     # This would ideally involve a hipStateVec function to copy probabilities or full state vector
    #     # For now, users can call measure repeatedly (on copies or by re-preparing state)
    #     # or use a future sampling function.
    #     raise NotImplementedError("get_probabilities is not yet implemented.")

    # def run(self, shots=1): # Future enhancement for shot-based simulation
    #     # For now, operations are applied eagerly.
    #     # This could collect measurement statistics over multiple shots.
    #     if shots == 1: # For compatibility with a potential future API
    #        print("Warning: 'run' method called with shots=1. Operations are applied eagerly. Consider using 'measure'.")
    #        # Or, if a final measurement is implied:
    #        # return self.measure_all() # if such a function exists
    #     raise NotImplementedError("'run' method with shots is not fully implemented for this eager execution model.")

```
