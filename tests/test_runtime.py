from __future__ import annotations

import sys
import unittest
from importlib.metadata import version


class RuntimeContractTest(unittest.TestCase):
    def test_project_and_runtime_versions_are_exact(self) -> None:
        import mujoco
        import robosuite

        import latency_meta_mdp

        self.assertEqual(sys.version_info[:2], (3, 10))
        self.assertEqual(latency_meta_mdp.__version__, "0.1.0")
        self.assertEqual(version("robosuite"), "1.5.2")
        self.assertEqual(version("mujoco"), "3.3.3")
        self.assertEqual(robosuite.__version__, "1.5.2")
        self.assertEqual(mujoco.__version__, "3.3.3")


if __name__ == "__main__":
    unittest.main()
