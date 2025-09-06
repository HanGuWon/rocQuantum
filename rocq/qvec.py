# Defines the quantum register abstraction.

class qvec:
    """
    A quantum vector representing a register of qubits.
    """
    _current_kernel_context = None

    def __init__(self, size: int):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("qvec size must be a positive integer.")
        self.size = size
        self.qubits = list(range(size))

        # Register this qvec with the kernel context if one is active
        if qvec._current_kernel_context:
            qvec._current_kernel_context.register_qvec(self)

    def __getitem__(self, key):
        return self.qubits[key]

    def __len__(self):
        return self.size
