# qiskit-rocquantum-provider/setup.py
from setuptools import setup, find_packages

setup(
    name="qiskit-rocquantum-provider",
    version="0.1.0",
    author="Gemini",
    description="Qiskit provider for the rocQuantum simulator.",
    packages=find_packages(),
    install_requires=[
        "qiskit>=0.45",
        "numpy",
    ],
    entry_points={
        "qiskit.providers": [
            "rocquantum = qiskit_rocquantum_provider:RocQuantumProvider",
        ]
    },
)
