# pennylane-rocq/setup.py
from setuptools import setup, find_packages

setup(
    name="pennylane-rocq",
    version="0.1.0",
    author="Gemini",
    description="PennyLane plugin for the rocQuantum simulator.",
    packages=find_packages(),
    install_requires=[
        "pennylane>=0.30",
        "numpy",
    ],
    entry_points={
        "pennylane.plugins": [
            "rocquantum.qpu = pennylane_rocq:RocQDevice",
        ]
    },
)
