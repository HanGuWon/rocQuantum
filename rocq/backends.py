# Integrates the Python frontend with the C++/HIP simulation engines.
# We assume a pre-compiled rocquantum_pybind module is available.
try:
    from rocquantum_pybind import StateVectorState, DensityMatrixState
except ImportError:
    print("WARNING: 'rocquantum_pybind' module not found. Using placeholder C++ objects.")
    # Define mock classes if the real binding isn't available,
    # allowing the Python-level code to still be demonstrated.
    class StateVectorState:
        def __init__(self, n_qubits): print(f"MOCK C++ SV: Initializing for {n_qubits} qubits.")
        def apply_h(self, t): print(f"MOCK C++ SV: apply_h on qubit {t}")
        def apply_x(self, t): print(f"MOCK C++ SV: apply_x on qubit {t}")
        def apply_y(self, t): print(f"MOCK C++ SV: apply_y on qubit {t}")
        def apply_z(self, t): print(f"MOCK C++ SV: apply_z on qubit {t}")
        def apply_cnot(self, c, t): print(f"MOCK C++ SV: apply_cnot on control={c}, target={t}")
        def apply_ry(self, angle, t): print(f"MOCK C++ SV: apply_ry(theta={angle}) on qubit {t}")
        def apply_rz(self, angle, t): print(f"MOCK C++ SV: apply_rz(phi={angle}) on qubit {t}")
        def get_state_vector(self):
            print("MOCK C++ SV: get_state_vector()")
            return "mock_cpp_state_vector_data"

    class DensityMatrixState:
        def __init__(self, n_qubits): print(f"MOCK C++ DM: Initializing for {n_qubits} qubits.")
        def apply_h(self, t): print(f"MOCK C++ DM: apply_h on qubit {t}")
        def apply_x(self, t): print(f"MOCK C++ DM: apply_x on qubit {t}")
        def apply_y(self, t): print(f"MOCK C++ DM: apply_y on qubit {t}")
        def apply_z(self, t): print(f"MOCK C++ DM: apply_z on qubit {t}")
        def apply_cnot(self, c, t): print(f"MOCK C++ DM: apply_cnot on control={c}, target={t}")
        def apply_ry(self, angle, t): print(f"MOCK C++ DM: apply_ry(theta={angle}) on qubit {t}")
        def apply_rz(self, angle, t): print(f"MOCK C++ DM: apply_rz(phi={angle}) on qubit {t}")
        def apply_depolarizing_channel(self, targets, prob): print(f"MOCK C++ DM: apply_depolarizing_channel on {targets} with prob={prob}")
        def apply_bit_flip_channel(self, targets, prob): print(f"MOCK C++ DM: apply_bit_flip_channel on {targets} with prob={prob}")
        def get_density_matrix(self):
            print("MOCK C++ DM: get_density_matrix()")
            return "mock_cpp_density_matrix_data"

class _BaseBackend:
    """Abstract base class for a quantum simulation backend."""
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def apply_gate(self, op_name, targets, params=None):
        raise NotImplementedError

    def apply_noise(self, channel, targets, prob):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

class StateVectorBackend(_BaseBackend):
    """Simulates a quantum state vector by dispatching to the C++ hipStateVec engine."""
    def __init__(self, num_qubits):
        super().__init__(num_qubits)
        self.cpp_state = StateVectorState(num_qubits)

    def apply_gate(self, op_name, targets, params=None):
        """Maps the kernel's gate name to the C++ state object's method."""
        op_name_lower = op_name.lower()
        params = params or {}

        if op_name_lower == 'h': self.cpp_state.apply_h(targets[0])
        elif op_name_lower == 'x': self.cpp_state.apply_x(targets[0])
        elif op_name_lower == 'y': self.cpp_state.apply_y(targets[0])
        elif op_name_lower == 'z': self.cpp_state.apply_z(targets[0])
        elif op_name_lower == 'cnot': self.cpp_state.apply_cnot(targets[0], targets[1])
        elif op_name_lower == 'ry': self.cpp_state.apply_ry(params['theta'], targets[0])
        elif op_name_lower == 'rz': self.cpp_state.apply_rz(params['phi'], targets[0])
        else: raise ValueError(f"Gate '{op_name}' is not supported by the StateVectorBackend.")

    def apply_noise(self, channel, targets, prob):
        """State vector simulator does not support noise. This is a hard constraint."""
        raise NotImplementedError("Noise models are only supported by the 'density_matrix' backend.")

    def get_state(self):
        """Retrieves the final state vector from the C++ object."""
        return self.cpp_state.get_state_vector()

class DensityMatrixBackend(_BaseBackend):
    """Simulates a quantum system using a density matrix by dispatching to the C++ hipDensityMat engine."""
    def __init__(self, num_qubits):
        super().__init__(num_qubits)
        self.cpp_state = DensityMatrixState(num_qubits)

    def apply_gate(self, op_name, targets, params=None):
        """Maps the kernel's gate name to the C++ state object's method."""
        op_name_lower = op_name.lower()
        params = params or {}

        if op_name_lower == 'h': self.cpp_state.apply_h(targets[0])
        elif op_name_lower == 'x': self.cpp_state.apply_x(targets[0])
        elif op_name_lower == 'y': self.cpp_state.apply_y(targets[0])
        elif op_name_lower == 'z': self.cpp_state.apply_z(targets[0])
        elif op_name_lower == 'cnot': self.cpp_state.apply_cnot(targets[0], targets[1])
        elif op_name_lower == 'ry': self.cpp_state.apply_ry(params['theta'], targets[0])
        elif op_name_lower == 'rz': self.cpp_state.apply_rz(params['phi'], targets[0])
        else: raise ValueError(f"Gate '{op_name}' is not supported by the DensityMatrixBackend.")

    def apply_noise(self, channel_type, targets, prob):
        """Maps the noise model's channel name to the C++ state object's method."""
        channel_lower = channel_type.lower()

        if channel_lower == 'depolarizing':
            self.cpp_state.apply_depolarizing_channel(targets, prob)
        elif channel_lower == 'bit_flip':
            self.cpp_state.apply_bit_flip_channel(targets, prob)
        else:
            raise ValueError(f"Noise channel '{channel_type}' is not supported by the DensityMatrixBackend.")

    def get_state(self):
        """Retrieves the final density matrix from the C++ object."""
        return self.cpp_state.get_density_matrix()

def get_backend(backend_name, num_qubits):
    """Factory function to instantiate a simulation backend.

    This function validates the requested backend name and returns an
    initialized backend object ready for simulation.

    Args:
        backend_name (str): The name of the backend to instantiate, e.g.,
            'state_vector' or 'density_matrix'.
        num_qubits (int): The number of qubits required for the simulation.

    Returns:
        _BaseBackend: An instance of the requested backend class.

    Raises:
        ValueError: If the requested `backend_name` is not supported.
    """
    SUPPORTED_BACKENDS = ['state_vector', 'density_matrix']
    if backend_name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend_name}'. Supported backends are: {SUPPORTED_BACKENDS}"
        )

    if backend_name == 'state_vector':
        return StateVectorBackend(num_qubits)
    elif backend_name == 'density_matrix':
        return DensityMatrixBackend(num_qubits)
    # This else is now logically unreachable due to the check above, but kept for clarity.
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
