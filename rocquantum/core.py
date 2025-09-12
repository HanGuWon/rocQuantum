# rocquantum/core.py

"""
This module serves as the central management hub for backend clients 
in the rocQuantum framework.

It provides the core functionality for dynamically selecting, instantiating,
and authenticating hardware backends, making them available to the rest of
the platform through a simple, unified interface.
"""

import importlib
from typing import Dict, Type, Optional

from .backends.base import RocqBackend

# ==============================================================================
#  Global State Management
# ==============================================================================

# Holds the currently active and authenticated backend instance.
_ACTIVE_BACKEND: Optional[RocqBackend] = None

# A registry of available backends. Maps a user-friendly name to the
# full import path of the corresponding backend class. This allows for
# easy extension with new hardware providers.
_AVAILABLE_BACKENDS: Dict[str, str] = {
    "ionq": "rocquantum.backends.ionq.IonQBackend",
    # Add other backends here as they are implemented, e.g.:
    # "quantinuum": "rocquantum.backends.quantinuum.QuantinuumBackend",
}

# ==============================================================================
#  Public API Functions
# ==============================================================================

def set_target(name: str, **kwargs) -> None:
    """
    Selects, instantiates, and authenticates a quantum backend.

    This function acts as the primary entry point for configuring the target
    QPU. It dynamically imports the specified backend class, initializes it
    with any provided arguments, and calls its authentication method.

    Args:
        name (str): The name of the backend to activate (e.g., 'ionq').
        **kwargs: Additional keyword arguments to be passed to the backend's
                  constructor (e.g., `backend_name='ionq_simulator'`).

    Raises:
        ValueError: If the specified backend name is not found in the
                    `_AVAILABLE_BACKENDS` registry.
        ImportError: If the backend class cannot be imported.
        Exception: Propagates exceptions from the backend's __init__ or
                   authenticate() methods (e.g., BackendAuthenticationError).
    """
    global _ACTIVE_BACKEND

    if name not in _AVAILABLE_BACKENDS:
        raise ValueError(
            f"Backend '{name}' not recognized. "
            f"Available backends are: {list(_AVAILABLE_BACKENDS.keys())}"
        )

    import_path = _AVAILABLE_BACKENDS[name]
    
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        backend_class: Type[RocqBackend] = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import backend class '{import_path}': {e}")

    # Instantiate the backend with any provided kwargs
    backend_instance = backend_class(**kwargs)
    
    # Authenticate the backend before setting it as active
    backend_instance.authenticate()
    
    # Set the successfully authenticated instance as the active backend
    _ACTIVE_BACKEND = backend_instance


def get_active_backend() -> RocqBackend:
    """
    Retrieves the currently active backend instance.

    Returns:
        RocqBackend: The active and authenticated backend client.

    Raises:
        RuntimeError: If no backend has been set via `set_target()`.
    """
    if _ACTIVE_BACKEND is None:
        raise RuntimeError(
            "No active backend. Please call `set_target('backend_name')` "
            "to select and authenticate a backend first."
        )
    return _ACTIVE_BACKEND
