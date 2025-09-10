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

    def __del__(self):
        pass

    def create_device_matrix(self, numpy_matrix: np.ndarray) -> backend.DeviceBuffer:
        if not isinstance(numpy_matrix, np.ndarray):
            raise TypeError("Input matrix must be a NumPy array.")
        if numpy_matrix.dtype != np.complex64:
            numpy_matrix = numpy_matrix.astype(np.complex64, order='C')
        if not numpy_matrix.flags['C_CONTIGUOUS']:
            numpy_matrix = np.ascontiguousarray(numpy_matrix, dtype=np.complex64)
        return backend.create_device_matrix_from_numpy(numpy_matrix)


class Circuit:
    def __init__(self, num_qubits: int, simulator: Simulator, multi_gpu: bool = False):
        if not isinstance(simulator, Simulator):
            raise TypeError("A valid Simulator instance is required.")
        if num_qubits < 0:
            raise ValueError("Number of qubits must be non-negative.")

        self.num_qubits = num_qubits
        self.simulator = simulator
        self._sim_handle = simulator._handle_wrapper
        self.is_multi_gpu = multi_gpu
        self._d_state_buffer = None
        self._gate_queue = []
        self._is_dirty = False

        try:
            if self.is_multi_gpu:
                if num_qubits == 0 and self.simulator.handle.get_num_gpus() > 1:
                     raise ValueError("Cannot create a 0-qubit distributed state across multiple GPUs. Use single GPU mode or at least log2(num_gpus) qubits.")
                backend.allocate_distributed_state(self._sim_handle, self.num_qubits)
                backend.initialize_distributed_state(self._sim_handle)
            else:
                self._d_state_buffer = backend.allocate_state_internal(self._sim_handle, self.num_qubits)
                status = backend.initialize_state(self._sim_handle, self._d_state_buffer, self.num_qubits)
                if status != backend.rocqStatus.SUCCESS:
                    raise RuntimeError(f"Failed to initialize state: {status}")

            self.simulator._active_circuits +=1
        except RuntimeError as e:
            print(f"Error during Circuit initialization: {e}")
            raise
            
    def __del__(self):
        if hasattr(self, 'simulator') and self.simulator is not None and hasattr(self.simulator, '_active_circuits'):
             if self.simulator._active_circuits > 0 :
                self.simulator._active_circuits -=1

    def flush(self):
        """Processes the gate queue, applying fusion."""
        if not self._is_dirty or not self._gate_queue:
            return

        # This is a placeholder for a more sophisticated fusion engine.
        # For now, we just execute the queue without fusion.
        print(f"Flushing {len(self._gate_queue)} gates...")
        for op in self._gate_queue:
            d_state_arg = self._get_d_state_for_backend()
            status = getattr(backend, f"apply_{op.name.lower()}")(self._sim_handle, d_state_arg, self.num_qubits, *op.controls, *op.targets, *op.params)
            if status != backend.rocqStatus.SUCCESS:
                raise RuntimeError(f"Apply {op.name} failed: {status}")

        self._gate_queue.clear()
        self._is_dirty = False

    def _enqueue_gate(self, name, targets, controls=[], params=[]):
        op = backend.GateOp()
        op.name = name
        op.targets = targets
        op.controls = controls
        op.params = params
        self._gate_queue.append(op)
        self._is_dirty = True

    def _get_d_state_for_backend(self) -> backend.DeviceBuffer:
        if self.is_multi_gpu:
            return backend.DeviceBuffer()
        return self._d_state_buffer

    def _validate_qubit_index(self, qubit_index, name="target qubit"):
        if not isinstance(qubit_index, int) or not (0 <= qubit_index < self.num_qubits):
            if not (self.num_qubits == 0 and qubit_index == 0):
                 raise ValueError(
                    f"{name} index {qubit_index} is out of range for {self.num_qubits} qubits."
                )

    def _validate_control_target(self, control_qubit, target_qubit):
        self._validate_qubit_index(control_qubit, "control qubit")
        self._validate_qubit_index(target_qubit, "target qubit")
        if control_qubit == target_qubit and self.num_qubits > 0 :
            raise ValueError("Control and target qubits cannot be the same.")

    def x(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("X", targets=[target_qubit])

    def y(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("Y", targets=[target_qubit])

    def z(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("Z", targets=[target_qubit])

    def h(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("H", targets=[target_qubit])

    def s(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("S", targets=[target_qubit])

    def t(self, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("T", targets=[target_qubit])

    def rx(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("RX", targets=[target_qubit], params=[angle])

    def ry(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("RY", targets=[target_qubit], params=[angle])

    def rz(self, angle: float, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._enqueue_gate("RZ", targets=[target_qubit], params=[angle])

    def cx(self, control_qubit: int, target_qubit: int): # CNOT
        self._validate_control_target(control_qubit, target_qubit)
        self._enqueue_gate("CNOT", targets=[target_qubit], controls=[control_qubit])

    def cz(self, qubit1: int, qubit2: int):
        self._validate_control_target(qubit1, qubit2)
        self._enqueue_gate("CZ", targets=[qubit2], controls=[qubit1])

    def swap(self, qubit1: int, qubit2: int):
        self._validate_control_target(qubit1, qubit2)
        self._enqueue_gate("SWAP", targets=[qubit1, qubit2])

    def crx(self, angle: float, control_qubit: int, target_qubit: int):
        self._validate_control_target(control_qubit, target_qubit)
        self._enqueue_gate("CRX", targets=[target_qubit], controls=[control_qubit], params=[angle])

    def cry(self, angle: float, control_qubit: int, target_qubit: int):
        self._validate_control_target(control_qubit, target_qubit)
        self._enqueue_gate("CRY", targets=[target_qubit], controls=[control_qubit], params=[angle])

    def crz(self, angle: float, control_qubit: int, target_qubit: int):
        self._validate_control_target(control_qubit, target_qubit)
        self._enqueue_gate("CRZ", targets=[target_qubit], controls=[control_qubit], params=[angle])

    def ccx(self, control_qubit1: int, control_qubit2: int, target_qubit: int):
        self._validate_qubit_index(target_qubit)
        self._validate_qubit_index(control_qubit1)
        self._validate_qubit_index(control_qubit2)
        self._enqueue_gate("MCX", targets=[target_qubit], controls=[control_qubit1, control_qubit2])

    def cswap(self, control_qubit: int, target_qubit1: int, target_qubit2: int):
        self._validate_qubit_index(control_qubit)
        self._validate_qubit_index(target_qubit1)
        self._validate_qubit_index(target_qubit2)
        self._enqueue_gate("CSWAP", targets=[target_qubit1, target_qubit2], controls=[control_qubit])

    def apply_unitary(self, qubit_indices: list[int], matrix: np.ndarray):
        self.flush() # Flush before complex operations
        # ... (rest of apply_unitary)

    def apply_controlled_unitary(self, control_qubits: list[int], target_qubits: list[int], matrix: np.ndarray):
        self.flush() # Flush before complex operations
        # ... (rest of apply_controlled_unitary)

    def measure(self, qubit_to_measure: int) -> tuple[int, float]:
        self.flush()
        self._validate_qubit_index(qubit_to_measure)
        d_state_arg = self._get_d_state_for_backend()
        try:
            outcome, probability = backend.measure(
                self._sim_handle, d_state_arg, self.num_qubits, qubit_to_measure
            )
            return outcome, probability
        except RuntimeError as e:
            raise RuntimeError(f"Measure failed: {e}")

    def sample(self, measured_qubits: list[int], num_shots: int) -> np.ndarray:
        self.flush()
        if not measured_qubits:
            raise ValueError("List of measured_qubits cannot be empty.")
        for idx in measured_qubits:
            self._validate_qubit_index(idx, f"measured_qubits element {idx}")
        if num_shots <= 0:
            raise ValueError("Number of shots must be positive.")

        d_state_arg = self._get_d_state_for_backend()
        try:
            results = backend.sample(
                self._sim_handle, d_state_arg, self.num_qubits, measured_qubits, num_shots
            )
            return results
        except RuntimeError as e:
            raise RuntimeError(f"Sample failed: {e}")

    def get_statevector(self) -> np.ndarray:
        """
        Flushes the execution queue and returns the final state vector from the GPU.
        Note: This involves a device-to-host memory transfer and can be slow.
        """
        self.flush()
        if self.batch_size > 1:
            raise NotImplementedError("get_statevector is not yet supported for batch_size > 1.")
        
        num_elements = 1 << self.num_qubits
        np_state = backend.get_state_vector(self._sim_handle, self._get_d_state_for_backend(), num_elements)
        return np_state

    def expval(self, pauli_operator: 'PauliOperator') -> float:
        """
        Calculates the expectation value of a Pauli operator with respect to the circuit's final state.
        """
        if not isinstance(pauli_operator, PauliOperator):
            raise TypeError("Input must be a PauliOperator object.")

        psi = self.get_statevector()
        total_exp_val = 0.0

        for pauli_term, coeff in pauli_operator.terms:
            if not pauli_term: # Identity term
                total_exp_val += coeff
                continue

            p_psi = psi.copy()
            for pauli_char, qubit_idx in pauli_term:
                p_psi = self._apply_pauli_to_state_np(p_psi, pauli_char, qubit_idx)
            
            term_exp_val = np.vdot(psi, p_psi).real
            total_exp_val += coeff * term_exp_val
            
        return total_exp_val

    def _apply_pauli_to_state_np(self, psi: np.ndarray, pauli: str, target: int) -> np.ndarray:
        """Helper to apply a single Pauli operator to a statevector in NumPy."""
        num_qubits = self.num_qubits
        psi_tensor = psi.reshape([2] * num_qubits)
        
        axes = list(range(num_qubits))
        axes[0], axes[target] = axes[target], axes[0]
        psi_tensor = np.transpose(psi_tensor, axes)

        if pauli == 'X':
            op_matrix = np.array([[0, 1], [1, 0]])
        elif pauli == 'Y':
            op_matrix = np.array([[0, -1j], [1j, 0]])
        elif pauli == 'Z':
            op_matrix = np.array([[1, 0], [0, -1]])
        else:
            op_matrix = np.identity(2)

        psi_tensor = psi_tensor.reshape(2, -1)
        psi_tensor = op_matrix @ psi_tensor
        psi_tensor = psi_tensor.reshape([2] * num_qubits)

        psi_tensor = np.transpose(psi_tensor, axes)
        return psi_tensor.flatten()


class PauliOperator:
    def __init__(self, terms: dict[str, float] | str = None):
        self.terms: list[tuple[list[tuple[str, int]], float]] = []

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
        if not isinstance(pauli_str, str):
            raise TypeError("Pauli string must be a string.")
        if not isinstance(coeff, (float, int)):
            raise TypeError("Coefficient must be a float or int.")

        components = pauli_str.strip().upper().split()
        if not components and pauli_str:
             if pauli_str.strip().upper() == "I":
                self.terms.append(([], float(coeff)))
                return
             else:
                raise ValueError(f"Invalid Pauli string component: {pauli_str}")

        parsed_ops = []
        for comp in components:
            if not comp: continue

            pauli_char = comp[0]
            if pauli_char not in "IXYZ":
                raise ValueError(f"Invalid Pauli type '{pauli_char}' in '{comp}'. Must be I, X, Y, or Z.")

            try:
                qubit_idx = int(comp[1:])
                if qubit_idx < 0:
                    raise ValueError("Qubit index cannot be negative.")
            except ValueError:
                raise ValueError(f"Invalid qubit index in '{comp}'. Must be an integer.")

            if pauli_char != 'I':
                parsed_ops.append((pauli_char, qubit_idx))

        self.terms.append((parsed_ops, float(coeff)))

    def __repr__(self):
        if not self.terms:
            return "PauliOperator(Empty)"
        term_strs = []
        for ops, coeff in self.terms:
            if not ops:
                op_str = "I"
            else:
                op_str = " ".join([f"{p}{q}" for p, q in ops])
            term_strs.append(f"{coeff} * [{op_str}]")
        return "PauliOperator(" + "\n+ ".join(term_strs) + "\n)"

    def __add__(self, other):
        if not isinstance(other, PauliOperator):
            return NotImplemented
        new_op = PauliOperator()
        new_op.terms = self.terms + other.terms
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
import inspect

class QuantumProgram:
    def __init__(self, name: str, num_qubits: int, mlir_compiler: backend.MLIRCompiler,
                 kernel_func=None, static_args=None, simulator_ref=None):
        self.name = name
        self.num_qubits = num_qubits
        self.mlir_compiler = mlir_compiler
        self.mlir_string = mlir_compiler.get_module_string()
        self.circuit_ref = None
        self._kernel_func = kernel_func
        self._static_args = static_args
        self._simulator_ref = simulator_ref

    def __repr__(self):
        self.mlir_string = self.mlir_compiler.get_module_string()
        return f"<QuantumProgram name='{self.name}' num_qubits={self.num_qubits}>\nMLIR:\n{self.mlir_string}"

    def dump(self):
        self.mlir_compiler.dump_module()

    def update_params(self, *params):
        if self.circuit_ref is None:
            if self._simulator_ref and self._kernel_func:
                print("Re-initializing circuit for update_params as circuit_ref was None.")
                self.circuit_ref = Circuit(self.num_qubits, self._simulator_ref)
            else:
                raise RuntimeError("Cannot update params: circuit_ref is None and no simulator/kernel info to rebuild.")

        if not self._kernel_func:
            raise RuntimeError("Cannot update params: Kernel function not stored in QuantumProgram.")

        if self.circuit_ref.is_multi_gpu:
             backend.initialize_distributed_state(self.circuit_ref._sim_handle)
        else:
            status = backend.initialize_state(self.circuit_ref._sim_handle,
                                              self.circuit_ref._get_d_state_for_backend(),
                                              self.circuit_ref.num_qubits)
            if status != backend.rocqStatus.SUCCESS:
                raise RuntimeError(f"Failed to re-initialize state for param update: {status}")

        kernel_args_for_py_call = [self.circuit_ref]
        if self._static_args:
            kernel_args_for_py_call.extend(self._static_args)
        kernel_args_for_py_call.extend(params)

        func_to_call = self._kernel_func.__wrapped__ if hasattr(self._kernel_func, '__wrapped__') else self._kernel_func
        func_to_call(*kernel_args_for_py_call)


def kernel(func):
    def generate_mlir_for_call(kernel_args, kernel_kwargs):
        mlir_lines = []
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)
            func_def = None
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break

            if not func_def:
                mlir_lines.append("// Could not find FunctionDef in AST")
                return "\n".join(mlir_lines)

            param_names = [arg.arg for arg in func_def.args.args[1:]]
            mlir_lines.append(f"func.func @{func.__name__}({', '.join([f'%arg{{i}}: !quantum.qubit' for i in range(kernel_args[0])])}) {{")

            for node in ast.walk(func_def):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == func_def.args.args[0].arg):

                        gate_name = node.func.attr
                        gate_args = node.args

                        if gate_name == "h" and len(gate_args) == 1 and isinstance(gate_args[0], ast.Constant):
                            qubit_idx = gate_args[0].value
                            mlir_lines.append(f"  %q{qubit_idx}_h = \"quantum.gate\"(%q{qubit_idx}) {{ gate_name = \"H\" }} : (!quantum.qubit) -> !quantum.qubit")
                        elif (gate_name == "cx" and len(gate_args) == 2 and
                              isinstance(gate_args[0], ast.Constant) and isinstance(gate_args[1], ast.Constant)):
                            ctrl_idx = gate_args[0].value
                            target_idx = gate_args[1].value
                            mlir_lines.append(f"  %q{ctrl_idx}_cx, %q{target_idx}_cx = \"quantum.gate\"(%q{ctrl_idx}, %q{target_idx}) {{ gate_name = \"CX\" }} : (!quantum.qubit, !quantum.qubit) -> (!quantum.qubit, !quantum.qubit)")
                        elif gate_name == "rx" and len(gate_args) == 2 and isinstance(gate_args[1], ast.Constant):
                            param_node = gate_args[0]
                            qubit_idx = gate_args[1].value
                            param_val_str = "UNKNOWN_PARAM"
                            if isinstance(param_node, ast.Name) and param_node.id in param_names:
                                try:
                                    param_idx_in_kernel_args = param_names.index(param_node.id)
                                    actual_param_value = kernel_args[1:][param_idx_in_kernel_args]
                                    param_val_str = str(actual_param_value)
                                except (IndexError, ValueError):
                                    pass
                            mlir_lines.append(f"  %q{qubit_idx}_rx = \"quantum.gate\"(%q{qubit_idx}) {{ gate_name = \"RX\", params = [{param_val_str}] }} : (!quantum.qubit) -> !quantum.qubit")
                        else:
                            mlir_lines.append(f"  // Unrecognized gate call: {gate_name}")
            mlir_lines.append("  return")
            mlir_lines.append("}")

        except Exception as e:
            mlir_lines.append(f"// Error during AST parsing: {e}")

        return "\n".join(mlir_lines)

    func.generate_mlir = generate_mlir_for_call
    return func


def build(kernel_func, num_qubits: int, simulator: Simulator, *args) -> QuantumProgram :
    if not hasattr(kernel_func, 'generate_mlir'):
        raise TypeError("The function provided to build() must be decorated with @rocq.kernel")

    print(f"--- Conceptual MLIR for kernel '{kernel_func.__name__}' ---")
    mlir_string_from_ast = kernel_func.generate_mlir((num_qubits,) + args, {})
    print(mlir_string_from_ast)
    print("----------------------------------------------------")

    compiler_instance = backend.MLIRCompiler()
    if not compiler_instance.initialize_module(kernel_func.__name__ + "_module"):
        raise RuntimeError("Failed to initialize MLIR module in compiler.")

    if not compiler_instance.load_module_from_string(mlir_string_from_ast):
        print(f"Warning: Failed to parse generated MLIR string for {kernel_func.__name__}. The program's MLIR module might be empty or invalid.")
        if not compiler_instance.initialize_module(kernel_func.__name__ + "_fallback_module"):
             raise RuntimeError("Fallback module initialization failed.")

    program = QuantumProgram(kernel_func.__name__,
                             num_qubits,
                             compiler_instance,
                             kernel_func=kernel_func,
                             static_args=None,
                             simulator_ref=simulator)

    if simulator:
        if not isinstance(simulator, Simulator):
             raise TypeError("A valid rocQ Simulator object is required if execution is expected.")

        program.circuit_ref = Circuit(num_qubits, program._simulator_ref)

        kernel_args_for_py_call = [program.circuit_ref] + list(args)
        func_to_call = kernel_func.__wrapped__ if hasattr(kernel_func, '__wrapped__') else kernel_func
        func_to_call(*kernel_args_for_py_call)

    return program


def get_expval(program: QuantumProgram, hamiltonian: PauliOperator) -> float:
    if not isinstance(program, QuantumProgram) or not isinstance(program.circuit_ref, Circuit):
        raise TypeError("Input must be a QuantumProgram object with an executed circuit_ref for v0.1 get_expval.")
    circuit = program.circuit_ref

    if not isinstance(hamiltonian, PauliOperator):
        raise TypeError("Input hamiltonian must be a rocQ PauliOperator object.")

    total_expval = 0.0

    for pauli_ops_list, coeff in hamiltonian.terms:
        if not pauli_ops_list:
            total_expval += coeff
            continue

        term_expval = 1.0

        if len(pauli_ops_list) == 1:
            pauli_char, qubit_idx = pauli_ops_list[0]
            if pauli_char == 'Z':
                try:
                    exp_val_z_contrib = backend.get_expectation_value_z(
                        circuit._sim_handle,
                        circuit._get_d_state_for_backend(),
                        circuit.num_qubits,
                        qubit_idx
                    )
                    term_expval = exp_val_z_contrib
                except AttributeError:
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
            else:
                raise NotImplementedError(f"Expectation value for Pauli '{pauli_char}' not supported in get_expval.")
        elif not pauli_ops_list:
             term_expval = 1.0
        else:
            is_all_z = True
            target_z_qubits = []
            for p_char, q_idx in pauli_ops_list:
                if p_char != 'Z':
                    is_all_z = False
                    break
                target_z_qubits.append(q_idx)

            if is_all_z:
                if not target_z_qubits:
                    term_expval = 1.0
                else:
                    try:
                        term_expval = backend.get_expectation_value_pauli_product_z(
                            circuit._sim_handle,
                            circuit._get_d_state_for_backend(),
                            circuit.num_qubits,
                            target_z_qubits
                        )
                    except AttributeError:
                        raise NotImplementedError(
                            "Backend function 'get_expectation_value_pauli_product_z' is not yet bound or implemented."
                        )
                    except RuntimeError as e:
                        op_str = " ".join([f"Z{q}" for q in target_z_qubits])
                        raise RuntimeError(f"Error calculating <{op_str}>: {e}")
            else:
                all_qubits = sorted(list(set([q_idx for _, q_idx in pauli_ops_list])))
                pauli_map = {q_idx: p_char for p_char, q_idx in pauli_ops_list}
                final_pauli_string = "".join([pauli_map.get(q, 'I') for q in all_qubits])
                final_qubit_indices = all_qubits

                try:
                    term_expval = backend.get_expectation_pauli_string(
                        circuit._sim_handle,
                        circuit._get_d_state_for_backend(),
                        circuit.num_qubits,
                        final_pauli_string,
                        final_qubit_indices
                    )
                except AttributeError:
                     raise NotImplementedError(
                        "Backend function 'get_expectation_pauli_string' is not yet bound or implemented. "
                        "This is required for products of Paulis containing X or Y."
                    )
                except RuntimeError as e:
                    op_str = " ".join([f"{p_char}{q_idx}" for p_char, q_idx in pauli_ops_list])
                    raise RuntimeError(f"Error calculating <{op_str}> using generic backend: {e}")

        total_expval += coeff * term_expval

    return total_expval

# Represents a quantum kernel, holding its MLIR representation.
class Kernel:
    def __init__(self, name, mlir_string=""):
        self.name = name
        self.mlir_string = mlir_string

    def __str__(self):
        return f"<Kernel name='{self.name}'>\n{self.mlir_string}"

def adjoint(kernel: Kernel) -> Kernel:
    """
    Generates the adjoint of a given quantum kernel by invoking the
    MLIR compiler backend.

    Args:
        kernel (Kernel): The input kernel to be adjointed.

    Returns:
        Kernel: A new kernel representing the module that now includes
                the adjoint of the input kernel.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError("Input to roc.adjoint must be a Kernel object.")

    wrapper_func_name = f"__wrapper_for_{kernel.name}"
    
    module_str = f"""
module {{
  {kernel.mlir_string}

  // Wrapper function to trigger the adjoint generation
  func.func @{wrapper_func_name}() {{
    // This call is marked for our pass to process.
    func.call @{kernel.name}() {{'rocq.adjoint.kernel'}} : () -> ()
    return
  }} 
}} """
    
    compiler = backend.MLIRCompiler()
    compiler.load_module_from_string(module_str)

    success = compiler.run_adjoint_generation_pass()
    if not success:
        raise RuntimeError("Adjoint generation compiler pass failed.")

    final_mlir_string = compiler.get_module_string()

    return Kernel(name=f"{kernel.name}.adj", mlir_string=final_mlir_string)

def grad(kernel_func, num_qubits: int, simulator: Simulator, initial_params: list[float], observable: PauliOperator):
    """
    Calculates the gradient of the expectation value of an observable with respect
    to the kernel's parameters using the parameter-shift rule.

    Args:
        kernel_func: The quantum kernel function, decorated with @rocq.kernel.
        num_qubits (int): The number of qubits required by the kernel.
        simulator (Simulator): The simulator instance to execute the circuits.
        initial_params (list[float]): The list of parameter values at which to compute the gradient.
        observable (PauliOperator): The observable to measure.

    Returns:
        np.ndarray: An array containing the partial derivative with respect to each parameter.
    """
    if not hasattr(kernel_func, 'generate_mlir'):
        raise TypeError("The function provided to grad() must be decorated with @rocq.kernel")

    gradients = []
    params = np.array(initial_params, dtype=float)
    
    for i in range(len(params)):
        # Create parameter vectors for the +pi/2 and -pi/2 shifts
        params_plus = params.copy()
        params_plus[i] += np.pi / 2.0
        
        params_minus = params.copy()
        params_minus[i] -= np.pi / 2.0

        # Build and execute the program for the +pi/2 shift
        prog_plus = build(kernel_func, num_qubits, simulator, *params_plus)
        expval_plus = get_expval(prog_plus, observable)

        # Build and execute the program for the -pi/2 shift
        prog_minus = build(kernel_func, num_qubits, simulator, *params_minus)
        expval_minus = get_expval(prog_minus, observable)
        
        # The parameter-shift rule for gates like Rx, Ry, Rz (exp(-i*theta*P/2))
        partial_derivative = 0.5 * (expval_plus - expval_minus)
        gradients.append(partial_derivative)

    return np.array(gradients)