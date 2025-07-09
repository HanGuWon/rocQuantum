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
    def __init__(self, num_qubits: int, simulator: Simulator, multi_gpu: bool = False):
        if not isinstance(simulator, Simulator):
            raise TypeError("A valid Simulator instance is required.")
        if num_qubits < 0: # Allow 0 qubits for 1-element state vector
            raise ValueError("Number of qubits must be non-negative.")

        self.num_qubits = num_qubits
        self.simulator = simulator
        self._sim_handle = simulator._handle_wrapper # Get the C++ handle wrapper
        self.is_multi_gpu = multi_gpu
        self._d_state_buffer = None # Will be None for multi-GPU

        try:
            if self.is_multi_gpu:
                if num_qubits == 0 and self.simulator.handle.get_num_gpus() > 1:
                     raise ValueError("Cannot create a 0-qubit distributed state across multiple GPUs. Use single GPU mode or at least log2(num_gpus) qubits.")
                backend.allocate_distributed_state(self._sim_handle, self.num_qubits)
                backend.initialize_distributed_state(self._sim_handle)
                # For multi-GPU, _d_state_buffer remains None as state is managed by handle
            else:
                # allocate_state_internal returns an owning DeviceBuffer
                self._d_state_buffer = backend.allocate_state_internal(self._sim_handle, self.num_qubits)
                status = backend.initialize_state(self._sim_handle, self._d_state_buffer, self.num_qubits)
                if status != backend.rocqStatus.SUCCESS:
                    raise RuntimeError(f"Failed to initialize state: {status}")

            self.simulator._active_circuits +=1
        except RuntimeError as e:
            print(f"Error during Circuit initialization: {e}")
            raise
            
    def __del__(self):
        # self._d_state_buffer (if not None) will be auto-freed by its C++ destructor
        if hasattr(self, 'simulator') and self.simulator is not None and hasattr(self.simulator, '_active_circuits'):
             if self.simulator._active_circuits > 0 : # Ensure it doesn't go negative
                self.simulator._active_circuits -=1
        # print(f"Circuit for {self.num_qubits} qubits (multi_gpu={self.is_multi_gpu}) being deleted.") # For debug

    def _get_d_state_for_backend(self) -> backend.DeviceBuffer:
        """Returns the appropriate DeviceBuffer for backend calls."""
        if self.is_multi_gpu:
            # For multi-GPU calls, the C-API functions often have a legacy d_state parameter.
            # The actual distributed state is within the handle. We pass a default (null) DeviceBuffer.
            # The C++ side should primarily use handle->d_local_state_slices in multi-GPU mode.
            return backend.DeviceBuffer()
        return self._d_state_buffer

    def _validate_qubit_index(self, qubit_index, name="target qubit"):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            # Allow target qubit 0 for a 0-qubit system (which has 1 state).
            if not (self.num_qubits == 0 and qubit_index == 0):
                 raise ValueError(
                    f"{name} index {qubit_index} is out of range for {self.num_qubits} qubits."
                )

    def _validate_control_target(self, control_qubit, target_qubit):
        self._validate_qubit_index(control_qubit, "control qubit")
        self._validate_qubit_index(target_qubit, "target qubit")
        if control_qubit == target_qubit and self.num_qubits > 0 : # For 0 qubits, c=t=0 is only option
            raise ValueError("Control and target qubits cannot be the same.")

    # --- Single-Qubit Gates ---
    def x(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_x(self._sim_handle, d_state_arg, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply X failed: {status}")

    def y(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_y(self._sim_handle, d_state_arg, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Y failed: {status}")

    def z(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_z(self._sim_handle, d_state_arg, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Z failed: {status}")

    def h(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_h(self._sim_handle, d_state_arg, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply H failed: {status}")

    def s(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_s(self._sim_handle, d_state_arg, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply S failed: {status}")

    def t(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_t(self._sim_handle, d_state_arg, self.num_qubits, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply T failed: {status}")

    def rx(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_rx(self._sim_handle, d_state_arg, self.num_qubits, target_qubit, angle)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Rx failed: {status}")

    def ry(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_ry(self._sim_handle, d_state_arg, self.num_qubits, target_qubit, angle)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Ry failed: {status}")

    def rz(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_rz(self._sim_handle, d_state_arg, self.num_qubits, target_qubit, angle)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply Rz failed: {status}")

    # --- Two-Qubit Gates ---
    def cx(self, control_qubit: int, target_qubit: int): # CNOT
        self._validate_control_target(control_qubit, target_qubit)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_cnot(self._sim_handle, d_state_arg, self.num_qubits, control_qubit, target_qubit)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply CNOT failed: {status}")

    def cz(self, qubit1: int, qubit2: int):
        self._validate_control_target(qubit1, qubit2)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_cz(self._sim_handle, d_state_arg, self.num_qubits, qubit1, qubit2)
        if status != backend.rocqStatus.SUCCESS: raise RuntimeError(f"Apply CZ failed: {status}")

    def swap(self, qubit1: int, qubit2: int):
        self._validate_control_target(qubit1, qubit2)
        d_state_arg = self._get_d_state_for_backend()
        status = backend.apply_swap(self._sim_handle, d_state_arg, self.num_qubits, qubit1, qubit2)
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
        d_state_arg = self._get_d_state_for_backend()
        
        try:
            status = backend.apply_matrix(
                self._sim_handle, 
                d_state_arg,
                self.num_qubits,
                qubit_indices,          # Pybind11 should handle list[int] to std::vector<unsigned>
                device_matrix_buffer,   # Pass the DeviceBuffer object
                expected_dim
            )
            if status != backend.rocqStatus.SUCCESS:
                raise RuntimeError(f"Apply Matrix failed: {status}")
        finally:
            # device_matrix_buffer is an RAII object and will be freed when it goes out of scope.
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
        d_state_arg = self._get_d_state_for_backend()
        try:
            # The numQubits passed to backend.measure should be the total number of qubits in the system
            # which is self.num_qubits. The C++ backend will use this along with the handle
            # to determine local qubit counts if it's a multi-GPU measurement.
            outcome, probability = backend.measure(
                self._sim_handle, d_state_arg, self.num_qubits, qubit_to_measure
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
    #     raise NotImplementedError("'run' method with shots is not fully implemented for this eager execution model.")


class PauliOperator:
    """
    Represents a Pauli operator as a sum of Pauli strings with coefficients.
    Example: 0.5 * Z0 Z1 + 0.25 * X0 I1
    """
    def __init__(self, terms: dict[str, float] | str = None):
        """
        Args:
            terms: A dictionary where keys are Pauli strings (e.g., "Z0 Z1", "X0")
                   and values are their coefficients.
                   Or a single string like "Z0 Z1" (coefficient 1.0).
        """
        self.terms: list[tuple[list[tuple[str, int]], float]] = [] # Canonical: [([(P,qidx),...], coeff), ...]

        if terms is None:
            return
        if isinstance(terms, str):
            self._add_pauli_string(terms, 1.0)
        elif isinstance(terms, dict):
            for pauli_str, coeff in terms.items():
                self._add_pauli_string(pauli_str, coeff)
        else:
            raise TypeError("PauliOperator terms must be a dict or a single Pauli string.")

    def _add_pauli_string(self, pauli_str: str, coeff: float):
        """Parses a single Pauli string like "X0 Y1 Z2" and adds it."""
        if not isinstance(pauli_str, str):
            raise TypeError("Pauli string must be a string.")
        if not isinstance(coeff, (float, int)):
            raise TypeError("Coefficient must be a float or int.")

        components = pauli_str.strip().upper().split()
        if not components and pauli_str: # Non-empty string but no components after split (e.g. "I")
             if pauli_str.strip().upper() == "I": # Global Identity
                self.terms.append(([], float(coeff))) # Empty list of ops for Identity
                return
             else:
                raise ValueError(f"Invalid Pauli string component: {pauli_str}")

        parsed_ops = []
        for comp in components:
            if not comp: continue # Skip empty parts if there were multiple spaces

            pauli_char = comp[0]
            if pauli_char not in "IXYZ":
                raise ValueError(f"Invalid Pauli type '{pauli_char}' in '{comp}'. Must be I, X, Y, or Z.")

            try:
                qubit_idx = int(comp[1:])
                if qubit_idx < 0:
                    raise ValueError("Qubit index cannot be negative.")
            except ValueError:
                raise ValueError(f"Invalid qubit index in '{comp}'. Must be an integer.")

            # Don't add Identity if it's specified per qubit e.g. "I0 X1" -> just "X1"
            if pauli_char != 'I':
                parsed_ops.append((pauli_char, qubit_idx))

        # Sort by qubit index for a canonical representation, though not strictly necessary for correctness here
        # parsed_ops.sort(key=lambda x: x[1])
        self.terms.append((parsed_ops, float(coeff)))

    def __repr__(self):
        if not self.terms:
            return "PauliOperator(Empty)"
        term_strs = []
        for ops, coeff in self.terms:
            if not ops: # Identity term
                op_str = "I"
            else:
                op_str = " ".join([f"{p}{q}" for p, q in ops])
            term_strs.append(f"{coeff} * [{op_str}]")
        return "PauliOperator(\n  " + "\n+ ".join(term_strs) + "\n)"

    def __add__(self, other):
        if not isinstance(other, PauliOperator):
            return NotImplemented
        new_op = PauliOperator()
        new_op.terms = self.terms + other.terms # Simple concatenation, could simplify later
        return new_op

    def __mul__(self, scalar: float):
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        new_op = PauliOperator()
        new_op.terms = [(ops, coeff * float(scalar)) for ops, coeff in self.terms]
        return new_op

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)


import ast
import inspect # To get source code for AST parsing

# Global/module level functions for rocHybrid v0.1
# These will eventually interact with a more sophisticated Program object and compiler.

class QuantumProgram:
    """
    Represents a "compiled" quantum program, initially just storing
    a string representation of conceptual MLIR.
    """
    def __init__(self, name: str, num_qubits: int, mlir_string: str):
        self.name = name
        self.num_qubits = num_qubits
        self.mlir_string = mlir_string
        self.circuit_ref = None # Will hold the executable Circuit for v0.1

    def __repr__(self):
        return f"<QuantumProgram name='{self.name}' num_qubits={self.num_qubits}>\nMLIR:\n{self.mlir_string}"


def kernel(func):
    """
    Decorator for quantum kernels.
    For v0.1, this will attach AST parsing capabilities to the function.
    """
    def generate_mlir_for_call(kernel_args, kernel_kwargs):
        """
        Parses the AST of the decorated function `func` and generates
        a conceptual MLIR string based on simple gate calls.
        `kernel_args` are the arguments passed to the kernel at build time.
        """
        mlir_lines = []
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)

            # Assuming the first argument to the kernel is the circuit/qubit register placeholder
            # And subsequent arguments are parameters.
            # For simplicity, map kernel parameters to MLIR function arguments.

            func_def = None
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break

            if not func_def:
                mlir_lines.append("// Could not find FunctionDef in AST")
                return "\n".join(mlir_lines)

            # Create a simple mapping from kernel Python arg names to MLIR arg names for the conceptual MLIR
            # The first Python arg is assumed to be the 'circuit' or 'qreg'
            param_names = [arg.arg for arg in func_def.args.args[1:]] # Skip circuit/qreg

            # Conceptual MLIR function signature
            # Example: func @kernel_name(%qreg : !quantum.qreg<num_qubits>, %theta : f64) { ... }
            # For now, we'll simplify and assume qubits are implicitly managed.
            # And parameters are passed to gates directly.

            mlir_lines.append(f"func.func @{func.__name__}({', '.join([f'%arg{i}: !quantum.qubit' for i in range(kernel_args[0])])}) {{") # kernel_args[0] is num_qubits

            # Very basic AST traversal for Call nodes
            for node in ast.walk(func_def):
                if isinstance(node, ast.Call):
                    # Check if it's a method call on the first argument (e.g., circuit.h())
                    if isinstance(node.func, ast.Attribute) and \
                       isinstance(node.func.value, ast.Name) and \
                       node.func.value.id == func_def.args.args[0].arg: # Check if it's like 'circuit.gate()'

                        gate_name = node.func.attr
                        gate_args = node.args

                        if gate_name == "h" and len(gate_args) == 1 and isinstance(gate_args[0], ast.Constant):
                            qubit_idx = gate_args[0].value
                            mlir_lines.append(f"  %q{qubit_idx}_h = \"quantum.gate\"(%q{qubit_idx}) {{ gate_name = \"H\" }} : (!quantum.qubit) -> !quantum.qubit")
                        elif gate_name == "cx" and len(gate_args) == 2 and \
                             isinstance(gate_args[0], ast.Constant) and isinstance(gate_args[1], ast.Constant):
                            ctrl_idx = gate_args[0].value
                            target_idx = gate_args[1].value
                            mlir_lines.append(f"  %q{ctrl_idx}_cx, %q{target_idx}_cx = \"quantum.gate\"(%q{ctrl_idx}, %q{target_idx}) {{ gate_name = \"CX\" }} : (!quantum.qubit, !quantum.qubit) -> (!quantum.qubit, !quantum.qubit)")
                        elif gate_name == "rx" and len(gate_args) == 2 and isinstance(gate_args[1], ast.Constant):
                            # Parameter for Rx needs to be identified from kernel_args
                            # This is a simplification: assumes param name in kernel matches call
                            param_node = gate_args[0]
                            qubit_idx = gate_args[1].value
                            param_val_str = "UNKNOWN_PARAM"
                            if isinstance(param_node, ast.Name) and param_node.id in param_names:
                                try:
                                    # Get actual value passed to kernel for this param
                                    param_idx_in_kernel_args = param_names.index(param_node.id)
                                    actual_param_value = kernel_args[1:][param_idx_in_kernel_args] # kernel_args[0] is num_qubits
                                    param_val_str = str(actual_param_value)
                                except (IndexError, ValueError):
                                    pass # Keep UNKNOWN_PARAM
                            mlir_lines.append(f"  %q{qubit_idx}_rx = \"quantum.gate\"(%q{qubit_idx}) {{ gate_name = \"RX\", params = [{param_val_str}] }} : (!quantum.qubit) -> !quantum.qubit")
                        # Add more gate translations here...
                        else:
                            mlir_lines.append(f"  // Unrecognized gate call: {gate_name}")
            mlir_lines.append("  return")
            mlir_lines.append("}")

        except Exception as e:
            mlir_lines.append(f"// Error during AST parsing: {e}")

        return "\n".join(mlir_lines)

    # Attach the MLIR generation function to the decorated kernel function
    func.generate_mlir = generate_mlir_for_call
    return func


def build(kernel_func, num_qubits: int, simulator: Simulator, *args) -> QuantumProgram :
    """
    "Builds" a quantum program from a kernel function and its arguments.
    For v0.1 MLIR scaffolding, this means:
    1. Triggering the AST-based MLIR string generation.
    2. Printing the MLIR string.
    3. Returning a QuantumProgram object that might store this string.
    The actual circuit execution is deferred/changed.
    """
    if not hasattr(kernel_func, 'generate_mlir'):
        raise TypeError("The function provided to build() must be decorated with @rocq.kernel")

    print(f"--- Conceptual MLIR for kernel '{kernel_func.__name__}' ---")
    # Pass (num_qubits, *args) to generate_mlir because the kernel's first arg (circuit/qreg) is implicit in MLIR context
    mlir_str = kernel_func.generate_mlir((num_qubits,) + args, {})
    print(mlir_str)
    print("----------------------------------------------------")

    program = QuantumProgram(kernel_func.__name__, num_qubits, mlir_str)

    # For VQE v0.1 to still work somewhat, we can execute the original kernel logic
    # This part makes `build` dual-purpose for now: generate MLIR and execute for v0.1 get_expval.
    # This is a temporary bridge.
    if simulator:
        if not isinstance(simulator, Simulator):
             raise TypeError("A valid rocQ Simulator object is required if execution is expected.")
        qcircuit = Circuit(num_qubits, simulator)
        original_kernel_args = [qcircuit] + list(args)
        func_to_call = kernel_func.__wrapped__ if hasattr(kernel_func, '__wrapped__') else kernel_func
        func_to_call(*original_kernel_args)
        program.circuit_ref = qcircuit # Store the executed circuit for get_expval

    return program


def get_expval(program: QuantumProgram, hamiltonian: PauliOperator) -> float:
    """
    Calculates the expectation value of a Hamiltonian with respect to the state
    prepared by the circuit.
    <psi|H|psi>
    For v0.1, this will be simplified.
    """
    # For v0.1, program.circuit_ref holds the executed Circuit object
    if not isinstance(program, QuantumProgram) or not isinstance(program.circuit_ref, Circuit):
        raise TypeError("Input must be a QuantumProgram object with an executed circuit_ref for v0.1 get_expval.")
    circuit = program.circuit_ref # Use the executed circuit from the program object

    if not isinstance(hamiltonian, PauliOperator):
        raise TypeError("Input hamiltonian must be a rocQ PauliOperator object.")

    total_expval = 0.0

    # For each term in the Hamiltonian (e.g., 0.5 * Z0Z1)
    for pauli_ops_list, coeff in hamiltonian.terms:
        if not pauli_ops_list: # Identity term
            total_expval += coeff # <psi|I|psi> = coeff * <psi|psi> = coeff * 1.0
            continue

        # For v0.1, we only support single Pauli Z terms for direct expectation value calculation
        # via a new backend function.
        # Example: if pauli_ops_list = [('Z', 0)], calculate <Z0>
        #          if pauli_ops_list = [('Z', 1)], calculate <Z1>
        # More complex terms like "Z0 X1" are NOT YET SUPPORTED by this simplified get_expval.

        term_expval = 1.0 # For products of Paulis, this would be more complex.
                         # For now, we assume only terms like c_i * Z_k or c_j * X_k etc.
                         # The PauliOperator structure allows for "Z0 X1", but this get_expval won't handle it yet.

        if len(pauli_ops_list) == 1:
            pauli_char, qubit_idx = pauli_ops_list[0]
            if pauli_char == 'Z':
                # This is where we'd call the backend function
                # result_tuple = backend.get_expectation_value_single_pauli_z(circuit._sim_handle, circuit._d_state_buffer, circuit.num_qubits, qubit_idx)
                # For now, this backend function doesn't exist.
                # Placeholder: For VQE v0.1, we might need to implement this specific measurement.
                # A Z expectation value is P(0) - P(1) for that qubit.
                # We can use the existing measure function, but it collapses state.
                # This is a critical point for the VQE.
                # Let's assume a backend.get_expectation_value_z(handle, d_state, num_qubits, target_qubit) will exist.
                try:
                    exp_val_z_contrib = backend.get_expectation_value_z( # This function needs to be added to bindings
                        circuit._sim_handle,
                        circuit._get_d_state_for_backend(),
                        circuit.num_qubits,
                        qubit_idx
                    )
                    term_expval = exp_val_z_contrib
                except AttributeError: # Fallback if backend.get_expectation_value_z is missing
                     raise NotImplementedError(
                        "Backend function 'get_expectation_value_z' is not yet bound or implemented. "
                        "VQE get_expval requires this for 'Z' terms."
                    )
                except RuntimeError as e:
                    raise RuntimeError(f"Error calculating <Z{qubit_idx}>: {e}")
            elif pauli_char == 'X':
                try:
                    exp_val_x_contrib = backend.get_expectation_value_x(
                        circuit._sim_handle,
                        circuit._get_d_state_for_backend(),
                        circuit.num_qubits,
                        qubit_idx
                    )
                    term_expval = exp_val_x_contrib
                except AttributeError:
                    raise NotImplementedError(
                        "Backend function 'get_expectation_value_x' is not yet bound or implemented. "
                        "VQE get_expval requires this for 'X' terms."
                    )
                except RuntimeError as e:
                    raise RuntimeError(f"Error calculating <X{qubit_idx}>: {e}")
            elif pauli_char == 'Y':
                try:
                    exp_val_y_contrib = backend.get_expectation_value_y(
                        circuit._sim_handle,
                        circuit._get_d_state_for_backend(),
                        circuit.num_qubits,
                        qubit_idx
                    )
                    term_expval = exp_val_y_contrib
                except AttributeError:
                    raise NotImplementedError(
                        "Backend function 'get_expectation_value_y' is not yet bound or implemented. "
                        "VQE get_expval requires this for 'Y' terms."
                    )
                except RuntimeError as e:
                    raise RuntimeError(f"Error calculating <Y{qubit_idx}>: {e}")
            else: # Should not happen due to PauliOperator parsing
                raise NotImplementedError(f"Expectation value for Pauli '{pauli_char}' not supported in get_expval.")
        elif not pauli_ops_list: # Identity term
             pass # Already handled: term_expval = 1.0, total_expval += coeff * 1.0
        else:
            # Product of Paulis, e.g., Z0 Z1
            raise NotImplementedError("Expectation value for products of Paulis (e.g., 'Z0 Z1') not yet supported in v0.1 get_expval.")

        total_expval += coeff * term_expval

    return total_expval

```
