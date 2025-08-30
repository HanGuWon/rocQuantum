import rocq
import numpy as np

# Create a simulator
sim = rocq.Simulator()

# Create a multi-GPU circuit
circuit = rocq.Circuit(num_qubits=3, simulator=sim, multi_gpu=True)

# Apply some gates
circuit.h(0)
circuit.cx(0, 1)

# Swap qubit 0 and 2
circuit.swap(0, 2)

# Measure qubit 0
outcome, prob = circuit.measure(0)

# Print the result
print(f"Measured outcome: {outcome} with probability {prob}")
