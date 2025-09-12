# rocquantum/core.py

"""
This module serves as the central management hub for backend clients 
in the rocQuantum framework.
"""

import importlib
from typing import Dict, Type, Optional

from .backends.base import RocqBackend

_AVAILABLE_BACKENDS: Dict[str, str] = {
    # --- Implemented Backends ---
    "ionq": "rocquantum.backends.ionq.IonQBackend",
    "infleqtion": "rocquantum.backends.infleqtion.InfleqtionBackend",
    "pasqal": "rocquantum.backends.pasqal.PasqalBackend",
    "quantinuum": "rocquantum.backends.quantinuum.QuantinuumBackend",
    "qristal": "rocquantum.backends.qristal.QuantumBrillianceBackend",

    # --- Skeleton Backends ---
    "iqm": "rocquantum.backends.iqm.IQMBackend",
    "rigetti": "rocquantum.backends.rigetti.RigettiBackend",
    "xanadu": "rocquantum.backends.xanadu.XanaduBackend",
    "quera": "rocquantum.backends.quera.QuEraBackend",
    "orca": "rocquantum.backends.orca.OrcaBackend",
    "seeqc": "rocquantum.backends.seeqc.SeeqcBackend",
    "quantum_machines": "rocquantum.backends.quantum_machines.QuantumMachinesBackend",
    "alice_bob": "rocquantum.backends.alice_bob.AliceBobBackend",
}

_ACTIVE_BACKEND: Optional[RocqBackend] = None

def set_target(name: str, **kwargs) -> None:
    """Selects, instantiates, and authenticates a quantum backend."""
    global _ACTIVE_BACKEND
    if name not in _AVAILABLE_BACKENDS:
        raise ValueError(f"Backend '{name}' not recognized. Available: {list(_AVAILABLE_BACKENDS.keys())}")
    
    import_path = _AVAILABLE_BACKENDS[name]
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        backend_class: Type[RocqBackend] = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import backend class '{import_path}': {e}")

    instance = backend_class(**kwargs)
    instance.authenticate()
    _ACTIVE_BACKEND = instance

def get_active_backend() -> RocqBackend:
    """Retrieves the currently active backend instance."""
    if _ACTIVE_BACKEND is None:
        raise RuntimeError("No active backend. Call set_target() first.")
    return _ACTIVE_BACKEND
