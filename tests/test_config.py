from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latency_meta_mdp.config import RuntimeConfig, load_runtime_config


class RuntimeConfigTest(unittest.TestCase):
    def test_default_contract_is_exactly_ten_physics_steps_per_formal_tick(self) -> None:
        config = RuntimeConfig.default()

        self.assertEqual(config.runtime_version, "robosuite_native_v1")
        self.assertEqual(config.physics_dt_us, 2_000)
        self.assertEqual(config.formal_tick_us, 20_000)
        self.assertEqual(config.physics_steps_per_tick, 10)
        self.assertEqual(config.camera_stride_ticks, 1)
        self.assertEqual(config.compatibility_stride_ticks, 5)
        self.assertEqual(config.control_freq_hz, 50)

    def test_non_integral_physics_ratio_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            RuntimeConfig(
                runtime_version="invalid",
                physics_dt_us=3_000,
                formal_tick_us=20_000,
                camera_stride_ticks=1,
                compatibility_stride_ticks=5,
                control_freq_hz=50,
                lite_physics=True,
            )

    def test_unknown_mapping_key_is_rejected(self) -> None:
        mapping = RuntimeConfig.default().to_mapping()
        mapping["hidden_frequency"] = 10

        with self.assertRaisesRegex(ValueError, "unknown runtime config keys"):
            RuntimeConfig.from_mapping(mapping)

    def test_locked_values_and_runtime_types_are_enforced(self) -> None:
        base = RuntimeConfig.default().to_mapping()
        invalid = [
            {**base, "runtime_version": "other"},
            {**base, "physics_dt_us": 4_000},
            {**base, "formal_tick_us": 10_000},
            {**base, "camera_stride_ticks": 2},
            {**base, "compatibility_stride_ticks": 10},
            {**base, "control_freq_hz": 100},
            {**base, "lite_physics": "true"},
            {**base, "runtime_version": 123},
        ]
        for mapping in invalid:
            with self.subTest(mapping=mapping):
                with self.assertRaises(ValueError):
                    RuntimeConfig.from_mapping(mapping)

    def test_runtime_config_loads_from_yaml(self) -> None:
        text = """
runtime_version: robosuite_native_v1
physics_dt_us: 2000
formal_tick_us: 20000
camera_stride_ticks: 1
compatibility_stride_ticks: 5
control_freq_hz: 50
lite_physics: true
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.yaml"
            path.write_text(text)
            self.assertEqual(load_runtime_config(path), RuntimeConfig.default())


if __name__ == "__main__":
    unittest.main()
