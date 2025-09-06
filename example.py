# This file demonstrates the intended user experience of the new rocQuantum-1 framework.
# It can be run to see the output of the placeholder backend implementations.

import rocq

# 1. Define a parameterized kernel
@rocq.kernel
def create_bell_state(theta: float, phi: float):
    """A kernel to create a generalized Bell state."""
    q = rocq.qvec(2)
    rocq.h(q[0])
    rocq.ry(theta, q[0])
    rocq.cnot(q[0], q[1])
    rocq.rz(phi, q[1])

# 2. Define a noise model
noise = rocq.NoiseModel()
# Add a depolarizing channel with 1% probability to be applied after any gate on qubits 0 and 1.
noise.add_channel('depolarizing', 0.01, on_qubits=[0, 1])
# Add a bit-flip channel with 0.5% probability after CNOT gates specifically.
noise.add_channel('bit_flip', 0.005, after_op='cnot')

# 3. Execute the kernel with specific parameters and settings
print("--- Running execution example ---")
result = rocq.execute(
    create_bell_state,
    theta=1.57,
    phi=0.5,
    backend='density_matrix',
    noise_model=noise
)
print(f"\nExecution result: {result}")


# 4. Define a Hamiltonian using the operator algebra
print("\n\n--- Running expectation value example ---")
# H = 0.5 * X(0) * Z(1) + 0.2 * Y(0)
# Note: In this placeholder, we represent "X(0) * Z(1)" as a single Pauli string.
hamiltonian = 0.5 * rocq.PauliOperator("X0 Z1") + 0.2 * rocq.PauliOperator("Y0")

# 5. Compute the expectation value
exp_val = rocq.get_expectation_value(
    create_bell_state,
    hamiltonian,
    backend='state_vector',
    theta=0.0,  # Use different parameters for this run
    phi=0.0
)

print(f"\nFinal expectation value: {exp_val}")
