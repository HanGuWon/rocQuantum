# cirq-rocm/setup.py
from setuptools import setup, find_packages

setup(
    name="cirq-rocm",
    version="0.1.0",
    author="Gemini",
    description="Cirq simulator plugin for the rocQuantum backend.",
    packages=find_packages(),
    install_requires=[
        "cirq-core>=1.0",
        "numpy",
    ],
)
