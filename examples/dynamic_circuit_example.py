import numpy as np
import rocq.api as rocq

def run_teleportation_example():
    """
    Demonstrates a dynamic circuit by implementing quantum teleportation.
    The circuit uses mid-circuit measurement outcomes to conditionally
    apply gates.
    """
    print("==========================================")
    print("= Dynamic Circuit: Teleportation Example =")
    print("==========================================")

    try:
        simulator = rocq.Simulator()
        
        # Define a test angle for the message state RY(angle)|0>
        message_angle = np.pi / 3.0

        # --- Part 1: Prepare the teleportation circuit ---
        print("\nBuilding teleportation circuit...")
        teleport_circuit = rocq.Circuit(3, simulator)

        # Qubit 0: The message qubit to be teleported
        # Qubit 1: Alice's half of the entangled pair
        # Qubit 2: Bob's half of the entangled pair, who will receive the message

        # 1. Create the message state on Qubit 0
        teleport_circuit.ry(message_angle, 0)

        # 2. Create an entangled Bell pair between Qubit 1 and Qubit 2
        teleport_circuit.h(1)
        teleport_circuit.cx(1, 2)

        # 3. Bell measurement part of the protocol
        teleport_circuit.cx(0, 1)
        teleport_circuit.h(0)

        # 4. Measure Qubit 0 and Qubit 1 to get two classical bits
        print("Performing mid-circuit measurements...")
        m1, _ = teleport_circuit.measure(0)
        m2, _ = teleport_circuit.measure(1)
        print(f"Measurement outcomes: m1={m1}, m2={m2}")

        # 5. Apply conditional gates based on measurement outcomes.
        # This is the core of the dynamic circuit.
        print("Applying conditional gates based on measurement outcomes...")
        if m2 == 1:
            print("Applying X gate to Qubit 2 because m2 was 1.")
            teleport_circuit.x(2)
        if m1 == 1:
            print("Applying Z gate to Qubit 2 because m1 was 1.")
            teleport_circuit.z(2)

        # At this point, the state of Qubit 0 should be teleported to Qubit 2.

        # --- Part 2: Verification ---
        print("\nVerifying the result...")
        
        # To verify, we calculate the expectation value <Z> of the final state of Qubit 2.
        observable = rocq.PauliOperator("Z2")
        # The get_expval function requires a QuantumProgram object.
        # We create a dummy one here containing the final circuit state.
        # A more integrated API might streamline this.
        final_program = rocq.QuantumProgram("teleport_final", 3, backend.MLIRCompiler())
        final_program.circuit_ref = teleport_circuit
        teleported_exp_val = rocq.get_expval(final_program, observable)

        # For comparison, create the original message state on a separate 1-qubit circuit
        # and calculate its <Z> expectation value.
        reference_circuit = rocq.Circuit(1, simulator)
        reference_circuit.ry(message_angle, 0)
        ref_program = rocq.QuantumProgram("reference", 1, backend.MLIRCompiler())
        ref_program.circuit_ref = reference_circuit
        reference_exp_val = rocq.get_expval(ref_program, rocq.PauliOperator("Z0"))

        # The analytical expectation value is cos(angle)
        analytical_exp_val = np.cos(message_angle)

        print(f"\nExpectation value <Z> of teleported qubit (Q2): {teleported_exp_val:.8f}")
        print(f"Expectation value <Z> of original message state: {reference_exp_val:.8f}")
        print(f"Analytical expectation value <Z> for angle {message_angle:.4f}: {analytical_exp_val:.8f}")

        assert np.isclose(teleported_exp_val, analytical_exp_val), "Teleportation failed! Expectation values do not match."

        print("\nSUCCESS: Teleportation successful. The final state matches the original message state.")

    except Exception as e:
        print(f"\nAn error occurred during the teleportation example: {e}")

if __name__ == "__main__":
    run_teleportation_example()
