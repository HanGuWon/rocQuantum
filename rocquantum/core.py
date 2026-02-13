# rocquantum/core.py

"""
This module serves as the central management hub for backend clients 
in the rocQuantum framework.
"""

import importlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set, Type

from .backends.base import RocqBackend

@dataclass(frozen=True)
class BackendSpec:
    """Metadata that drives target selection and capability checks."""

    import_path: str
    backend_type: str
    capabilities: Set[str] = field(default_factory=set)


_AVAILABLE_BACKENDS: Dict[str, BackendSpec] = {
    # --- Implemented Backends ---
    "ionq": BackendSpec(
        import_path="rocquantum.backends.ionq.IonQBackend",
        backend_type="remote_api",
        capabilities={"sampling", "job_lifecycle", "qasm_submission"},
    ),
    "infleqtion": BackendSpec(
        import_path="rocquantum.backends.infleqtion.InfleqtionBackend",
        backend_type="remote_api",
        capabilities={"sampling", "job_lifecycle", "qasm_submission"},
    ),
    "pasqal": BackendSpec(
        import_path="rocquantum.backends.pasqal.PasqalBackend",
        backend_type="remote_api",
        capabilities={"sampling", "job_lifecycle", "qasm_submission"},
    ),
    "quantinuum": BackendSpec(
        import_path="rocquantum.backends.quantinuum.QuantinuumBackend",
        backend_type="remote_api",
        capabilities={"sampling", "job_lifecycle", "qasm_submission"},
    ),
    "qristal": BackendSpec(
        import_path="rocquantum.backends.qristal.QuantumBrillianceBackend",
        backend_type="local_sdk",
        capabilities={"sampling", "qasm_submission"},
    ),

    # --- Skeleton Backends ---
    "iqm": BackendSpec("rocquantum.backends.iqm.IQMBackend", "skeleton", set()),
    "rigetti": BackendSpec("rocquantum.backends.rigetti.RigettiBackend", "cloud_intermediary", {"sampling", "job_lifecycle", "qasm_submission"}),
    "xanadu": BackendSpec("rocquantum.backends.xanadu.XanaduBackend", "skeleton", set()),
    "quera": BackendSpec("rocquantum.backends.quera.QuEraBackend", "skeleton", set()),
    "orca": BackendSpec("rocquantum.backends.orca.OrcaBackend", "skeleton", set()),
    "seeqc": BackendSpec("rocquantum.backends.seeqc.SeeqcBackend", "skeleton", set()),
    "quantum_machines": BackendSpec("rocquantum.backends.quantum_machines.QuantumMachinesBackend", "skeleton", set()),
    "alice_bob": BackendSpec("rocquantum.backends.alice_bob.AliceBobBackend", "skeleton", set()),
}

_ACTIVE_BACKEND: Optional[RocqBackend] = None

def set_target(name: str, **kwargs) -> None:
    """Selects, instantiates, and authenticates a quantum backend."""
    global _ACTIVE_BACKEND
    if name not in _AVAILABLE_BACKENDS:
        raise ValueError(f"Backend '{name}' not recognized. Available: {list(_AVAILABLE_BACKENDS.keys())}")
    
    import_path = _AVAILABLE_BACKENDS[name].import_path
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        backend_class: Type[RocqBackend] = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import backend class '{import_path}': {e}")

    instance = backend_class(**kwargs)
    instance.authenticate()
    _ACTIVE_BACKEND = instance


def get_target_spec(name: str) -> BackendSpec:
    """Returns static metadata for a backend target."""
    if name not in _AVAILABLE_BACKENDS:
        raise ValueError(f"Backend '{name}' not recognized. Available: {list(_AVAILABLE_BACKENDS.keys())}")
    return _AVAILABLE_BACKENDS[name]


def list_targets(required_capabilities: Optional[Iterable[str]] = None) -> Dict[str, BackendSpec]:
    """
    Lists available targets, optionally filtered by required capabilities.

    Examples:
        list_targets()  # all configured backends
        list_targets({"sampling", "job_lifecycle"})
    """
    if required_capabilities is None:
        return dict(_AVAILABLE_BACKENDS)

    required = set(required_capabilities)
    return {
        name: spec
        for name, spec in _AVAILABLE_BACKENDS.items()
        if required.issubset(spec.capabilities)
    }


def require_target_capability(name: str, capability: str) -> None:
    """Validates that a target advertises a given capability before use."""
    spec = get_target_spec(name)
    if capability not in spec.capabilities:
        raise RuntimeError(
            f"Backend '{name}' does not advertise capability '{capability}'. "
            f"Available capabilities: {sorted(spec.capabilities)}"
        )

def get_active_backend() -> RocqBackend:
    """Retrieves the currently active backend instance."""
    if _ACTIVE_BACKEND is None:
        raise RuntimeError("No active backend. Call set_target() first.")
    return _ACTIVE_BACKEND
