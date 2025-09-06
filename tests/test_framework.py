# Unit tests for the rocQuantum-1 (rocq) framework.
import unittest
import sys
import os

# Add the parent directory to the path to allow importing 'rocq'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rocq

class TestRocqFramework(unittest.TestCase):
    """Test suite for the core rocq framework components."""

    def test_kernel_creation_and_structure(self):
        """
        Tests that a QuantumKernel is created correctly and has the expected
        internal structure (e.g., name, number of gates).
        """
        @rocq.kernel
        def my_kernel(theta: float):
            q = rocq.qvec(2)
            rocq.h(q[0])
            rocq.cnot(q[0], q[1])
            rocq.rz(theta, q[1])

        # The object created by the decorator is the QuantumKernel instance
        self.assertIsInstance(my_kernel, rocq.kernel.QuantumKernel)
        self.assertEqual(my_kernel.name, 'my_kernel')
        # Check that the internal gate sequence has the correct number of operations
        self.assertEqual(len(my_kernel.gate_sequence), 3)
        print("OK: test_kernel_creation_and_structure")

    def test_backend_factory_validation(self):
        """
        Tests that the get_backend factory function (used by rocq.execute)
        correctly raises a ValueError for an unsupported backend name.
        """
        @rocq.kernel
        def dummy_kernel():
            q = rocq.qvec(1)
            rocq.h(q[0])

        with self.assertRaises(ValueError) as cm:
            rocq.execute(dummy_kernel, backend='invalid_backend')
        
        # Check that the error message is user-friendly and lists available backends
        self.assertIn("Unsupported backend 'invalid_backend'", str(cm.exception))
        self.assertIn("['state_vector', 'density_matrix']", str(cm.exception))
        print("OK: test_backend_factory_validation")

    def test_state_vector_backend_noise_rejection(self):
        """
        Tests that the 'state_vector' backend correctly raises a
        NotImplementedError when a noise model is provided.
        """
        @rocq.kernel
        def dummy_kernel():
            q = rocq.qvec(1)
            rocq.h(q[0])

        noise = rocq.NoiseModel()
        noise.add_channel('depolarizing', 0.1)

        with self.assertRaises(NotImplementedError) as cm:
            rocq.execute(dummy_kernel, backend='state_vector', noise_model=noise)
        
        self.assertEqual(
            str(cm.exception),
            "Noise models are only supported by the 'density_matrix' backend."
        )
        print("OK: test_state_vector_backend_noise_rejection")

    def test_operator_and_expectation_value_api(self):
        """
        Tests that the high-level get_expectation_value API is wired correctly
        and returns the expected (mocked) value.
        """
        @rocq.kernel
        def prep_state():
            q = rocq.qvec(1)
            rocq.h(q[0])

        # H = 0.5 * Z(0)
        hamiltonian = 0.5 * rocq.PauliOperator("Z0")

        # The mock backend returns a dummy value of 42.0
        expected_value = 42.0
        
        exp_val = rocq.get_expectation_value(
            prep_state,
            hamiltonian,
            backend='state_vector'
        )

        self.assertEqual(exp_val, expected_value)
        print("OK: test_operator_and_expectation_value_api")

if __name__ == '__main__':
    unittest.main()
