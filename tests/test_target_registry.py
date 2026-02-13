import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rocquantum.core import (
    BackendSpec,
    get_target_spec,
    list_targets,
    require_target_capability,
)


class TestTargetRegistry(unittest.TestCase):
    def test_get_target_spec_has_metadata(self):
        spec = get_target_spec("ionq")
        self.assertIsInstance(spec, BackendSpec)
        self.assertEqual(spec.backend_type, "remote_api")
        self.assertIn("sampling", spec.capabilities)

    def test_list_targets_by_capability(self):
        filtered = list_targets({"job_lifecycle"})
        self.assertIn("ionq", filtered)
        self.assertNotIn("qristal", filtered)

    def test_require_target_capability(self):
        require_target_capability("ionq", "sampling")

        with self.assertRaises(RuntimeError):
            require_target_capability("qristal", "job_lifecycle")


if __name__ == "__main__":
    unittest.main()
