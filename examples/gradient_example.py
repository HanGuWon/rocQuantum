import numpy as np
import rocq.api as rocq

# 1. Define a parameterized quantum kernel.
# The @rocq.kernel decorator will eventually enable AST parsing
# to generate MLIR, but for now, the logic in build() is conceptual.
# The important part is that the function signature and body are
# used by the framework.
@rocq.kernel
def ansatz(circuit: rocq.Circuit, theta: float):
    """
    A simple ansatz with a single rotation gate.
    The expectation value of Z0 for this circuit is cos(theta).
    The gradient of the expectation value is -sin(theta).
    """
    circuit.rx(theta, 0)

def run_gradient_example():
    """
    Demonstrates calculating the gradient of a kernel's expectation
    value with respect to its parameters using the parameter-shift rule.
    """
    print("==========================================")
    print("= Gradient Calculation Example           =")
    print("==========================================")

    # 2. Define the observable.
    hamiltonian = rocq.PauliOperator("Z0")

    # 3. Choose a parameter value.
    theta_value = np.pi / 4.0

    # 4. Calculate the gradient using the new roc.grad function.
    print(f"\nCalculating gradient for RX(theta) at theta = {theta_value:.4f}...")
    
    try:
        # The grad function will execute the circuit twice under the hood
        # (for theta + pi/2 and theta - pi/2) to get the gradient.
        # It requires a simulator instance to run the circuits.
        simulator = rocq.Simulator()
        
        # The grad function needs to know the number of qubits.
        num_qubits = 1
        
        # The kernel function itself, its parameters, the number of qubits,
        # the simulator, and the observable are passed to grad.
        gradient = rocq.grad(
            kernel_func=ansatz,
            num_qubits=num_qubits,
            simulator=simulator,
            initial_params=[theta_value],
            observable=hamiltonian
        )
        
        # 5. Compare with the analytical result.
        analytical_gradient = -np.sin(theta_value)
        
        print(f"\nComputed Gradient: {gradient[0]:.8f}")
        print(f"Analytical Gradient: {analytical_gradient:.8f}")
        
        assert np.isclose(gradient[0], analytical_gradient), "Gradient does not match analytical result!"
        
        print("\nSUCCESS: Gradient calculated successfully and matches the analytical value.")

    except Exception as e:
        print(f"\nAn error occurred during gradient calculation: {e}")
        print("Please ensure all rocQuantum components are built and installed correctly.")


if __name__ == "__main__":
    run_gradient_example()