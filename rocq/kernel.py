# Task 2: QuantumKernel and execute Abstraction
import inspect
from functools import wraps
from .qvec import qvec
from .backends import get_backend

class _KernelBuildContext:
    """A thread-local context for capturing gate operations during kernel definition."""
    _context = None

    def __enter__(self):
        self.gate_sequence = []
        self.qvec_size = 0
        _KernelBuildContext._context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _KernelBuildContext._context = None

    def register_qvec(self, qv):
        self.qvec_size = qv.size

    @staticmethod
    def add_gate(op_name, targets, params=None):
        if _KernelBuildContext._context:
            gate_info = {"op": op_name, "targets": targets, "params": params or {}}
            _KernelBuildContext._context.gate_sequence.append(gate_info)
        else:
            raise RuntimeError("Gate operations can only be called inside a @rocq.kernel function.")

class QuantumKernel:
    """An object representing a parameterized quantum circuit.

    This class is the result of decorating a Python function with `@rocq.kernel`.
    It captures the sequence of quantum operations defined within the function,
    creating a reusable and backend-agnostic representation of the circuit.
    The kernel can have parameters that are specified at execution time.

    Attributes:
        func (Callable): The original Python function that defines the kernel.
        name (str): The name of the original function.
        parameters (dict): A dictionary of the kernel's parameters, extracted
            from the function's signature.
        gate_sequence (list): An internal list of gate operations to be executed.
        num_qubits (int): The number of qubits required for the kernel.
    """
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.parameters = inspect.signature(func).parameters

        # "Record" the gate sequence by running the function once in a build context
        with _KernelBuildContext() as context:
            # We need to provide dummy arguments for the recording pass
            dummy_args = [None] * len(self.parameters)
            qvec._current_kernel_context = context
            self.func(*dummy_args)
            qvec._current_kernel_context = None
            self.gate_sequence = context.gate_sequence
            self.num_qubits = context.qvec_size

    def __call__(self, *args, **kwargs):
        """Allows the kernel object to be called like a function."""
        return self.func(*args, **kwargs)


def kernel(func):
    """A decorator that converts a Python function into a QuantumKernel.

    This is the primary mechanism for defining quantum circuits in rocQuantum-1.
    The decorated function should define a sequence of quantum gate operations.
    Any arguments to the function are treated as runtime parameters for the
    kernel, which is essential for variational algorithms.

    Args:
        func (Callable): The function to be converted into a quantum kernel.

    Returns:
        QuantumKernel: An executable object representing the quantum circuit.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # For this design, we return the kernel object for later execution.
        return QuantumKernel(func)

    return QuantumKernel(func)


def execute(kernel_obj: QuantumKernel, backend: str, noise_model=None, **kwargs):
    """The primary user entry point for executing a quantum kernel.

    This function orchestrates the simulation. It takes a QuantumKernel,
    instantiates the specified backend, binds the runtime parameters to the
    kernel, applies the gate sequence, and optionally applies a noise model.

    Args:
        kernel_obj (QuantumKernel): The quantum kernel to execute.
        backend (str): The name of the simulation backend to use
            (e.g., 'state_vector', 'density_matrix').
        noise_model (Optional[NoiseModel]): A noise model to apply during
            simulation. Only compatible with the 'density_matrix' backend.
        **kwargs: The runtime values for the kernel's parameters.

    Returns:
        Any: The final state from the backend (e.g., a state vector or a
             density matrix). The specific type depends on the backend used.
    """
    print(f"\n--- EXECUTING KERNEL: {kernel_obj.name} ---")
    print(f"Backend: {backend}, Noise Model: {'Yes' if noise_model else 'No'}")

    # 1. Instantiate the correct backend
    sim_backend = get_backend(backend, kernel_obj.num_qubits)

    # 2. Bind runtime arguments to the kernel's parameters
    bound_args = {}
    for name, param in kernel_obj.parameters.items():
        if name in kwargs:
            bound_args[name] = kwargs[name]
        else:
            # This could be extended to handle default values
            raise ValueError(f"Missing required kernel argument: {name}")
    print(f"Bound arguments: {bound_args}")

    # 3. Iterate through the kernel's internal gate sequence
    for gate in kernel_obj.gate_sequence:
        op_name = gate["op"]
        targets = gate["targets"]
        params = gate["params"]

        # Substitute concrete parameter values
        resolved_params = {p_name: bound_args.get(p_val, p_val) for p_name, p_val in params.items()}

        # Apply gate to the backend
        sim_backend.apply_gate(op_name, targets, resolved_params)

        # 4. Apply noise channels if a noise model is provided
        if noise_model:
            for channel in noise_model.get_channels():
                # Check if this channel should be applied
                op_match = (not channel["op"]) or (channel["op"] == op_name)
                
                if op_match:
                    # Determine target qubits for the noise
                    noise_targets = channel["qubits"] if channel["qubits"] is not None else targets
                    sim_backend.apply_noise(channel["type"], noise_targets, channel["prob"])

    # 5. Return the final state or measurement results
    result = sim_backend.get_state()
    print(f"--- EXECUTION COMPLETE ---")
    return result
