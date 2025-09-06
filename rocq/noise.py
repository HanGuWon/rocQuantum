# Task 1: High-Level NoiseModel Class
import collections

class NoiseModel:
    """A declarative noise model for specifying noise channels.

    This object collects specifications for noise channels that will be applied
    during the execution of a quantum kernel on a compatible backend (e.g.,
    the 'density_matrix' backend).

    Attributes:
        _channels (list): A private list storing the configuration of each
            added noise channel.

    Usage Example:
        >>> noise_model = rocq.NoiseModel()
        >>> # Add a 1% depolarizing channel after any gate on qubits 0 and 1.
        >>> noise_model.add_channel('depolarizing', 0.01, on_qubits=[0, 1])
        >>> # Add a 0.5% bit-flip channel specifically after CNOT gates.
        >>> noise_model.add_channel('bit_flip', 0.005, after_op='cnot')
    """

    def __init__(self):
        """Initializes an empty noise model."""
        self._channels = []

    def add_channel(self, channel_type: str, probability: float, on_qubits=None, after_op: str = None):
        """Adds a noise channel to the model.

        Args:
            channel_type (str): The type of noise (e.g., 'depolarizing',
                'bit_flip'). This must match a channel supported by the
                target backend.
            probability (float): The probability of the noise occurring. Must
                be between 0.0 and 1.0.
            on_qubits (Optional[list[int]]): A specific list of qubits to
                apply the noise to. If None, the noise is applied to all
                qubits involved in the preceding gate operation. Defaults to None.
            after_op (Optional[str]): Apply noise only after a specific gate
                type (e.g., 'cnot'). If None, the noise is applied after any
                gate. Defaults to None.
        """
        if not isinstance(probability, (int, float)) or not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1.")

        channel_spec = {
            "type": channel_type,
            "prob": probability,
            "qubits": on_qubits,
            "op": after_op.lower() if after_op else None
        }
        self._channels.append(channel_spec)
        print(f"NOISE_MODEL: Added '{channel_type}' channel with prob={probability}.")

    def get_channels(self):
        """Returns the list of configured noise channels."""
        return self._channels
