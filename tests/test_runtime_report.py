from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from latency_meta_mdp.envs.resources import load_resource_manifest
from latency_meta_mdp.runtime.loop import (
    _git_revision,
    _sha256,
    _source_sha256,
    publish_g0_run,
    validate_runtime_report,
)


def valid_report() -> dict:
    resource_count = len(
        load_resource_manifest("assets/resource_manifest.json", repository_root=".")
    )
    config = {
        "camera_stride_ticks": 1,
        "compatibility_stride_ticks": 5,
        "control_freq_hz": 50,
        "formal_tick_us": 20_000,
        "lite_physics": True,
        "physics_dt_us": 2_000,
        "runtime_version": "robosuite_native_v1",
    }
    return {
        "action_dim": 7,
        "compiled_timestep_seconds": 0.002,
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "control_freq_hz": 50,
        "control_timestep_seconds": 0.02,
        "controller": "Panda/default_panda/BASIC",
        "environment": "Lift",
        "external_control_callback_present": False,
        "formal_tick_us": 20_000,
        "full_state_fields": ["time", "qpos", "qvel", "act"],
        "implementation_revision": _git_revision(),
        "implementation_source_sha256": _source_sha256(),
        "integrator": "Euler",
        "integrator_configured_by_project": True,
        "mujoco_version": "3.3.3",
        "physics_dt_us": 2_000,
        "physics_steps_per_tick": 10,
        "python_version": "3.10.20",
        "renderer": "headless-no-camera",
        "resource_count": resource_count,
        "resource_manifest_sha256": _sha256(Path("assets/resource_manifest.json")),
        "robosuite_version": "1.5.2",
        "runtime_version": "robosuite_native_v1",
        "same_seed_full_act_max_abs_error": 0.0,
        "same_seed_full_qpos_max_abs_error": 0.0,
        "same_seed_full_qvel_max_abs_error": 0.0,
        "same_seed_initial_object_qpos_max_abs_error": 0.0,
        "same_seed_initial_qpos_max_abs_error": 0.0,
        "same_seed_sim_time_abs_error": 0.0,
        "schema_version": 2,
        "seed": 7,
        "stock_step_elapsed_seconds": 0.02,
        "uv_lock_sha256": _sha256(Path("uv.lock")),
    }


class RuntimeReportTest(unittest.TestCase):
    def test_valid_report_derives_eligibility(self) -> None:
        report = valid_report()
        validated = validate_runtime_report(report)
        self.assertTrue(validated["eligible"])
        self.assertEqual(validated["blockers"], [])

    def test_caller_cannot_forge_eligibility(self) -> None:
        report = {**valid_report(), "compiled_timestep_seconds": 0.004}
        report["eligible"] = True
        report["blockers"] = []
        validated = validate_runtime_report(report)
        self.assertFalse(validated["eligible"])
        self.assertIn("compiled_timestep_mismatch", validated["blockers"])

    def test_missing_self_describing_field_is_rejected(self) -> None:
        report = valid_report()
        del report["uv_lock_sha256"]
        with self.assertRaisesRegex(ValueError, "missing runtime report fields"):
            validate_runtime_report(report)

    def test_nonfinite_timing_is_ineligible(self) -> None:
        report = {**valid_report(), "compiled_timestep_seconds": float("nan")}
        validated = validate_runtime_report(report)
        self.assertFalse(validated["eligible"])
        self.assertIn("compiled_timestep_seconds_nonfinite", validated["blockers"])

    def test_full_state_evidence_fields_are_exact(self) -> None:
        report = {**valid_report(), "full_state_fields": []}
        validated = validate_runtime_report(report)
        self.assertFalse(validated["eligible"])
        self.assertIn("full_state_fields_mismatch", validated["blockers"])

    def test_provenance_hashes_are_reconciled_with_current_source(self) -> None:
        report = {**valid_report(), "uv_lock_sha256": "f" * 64}
        validated = validate_runtime_report(report)
        self.assertFalse(validated["eligible"])
        self.assertIn("uv_lock_hash_mismatch", validated["blockers"])

    def test_ineligible_report_is_retained_but_not_indexed(self) -> None:
        report = {**valid_report(), "compiled_timestep_seconds": 0.004}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = publish_g0_run(root, "failed_probe", report)
            manifest = json.loads(manifest_path.read_text())
            self.assertFalse(manifest["eligible"])
            self.assertIn("compiled_timestep_mismatch", manifest["blockers"])
            self.assertFalse((root / "artifact_index.json").exists())

    def test_malformed_report_leaves_no_staging_or_run(self) -> None:
        report = valid_report()
        del report["uv_lock_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing runtime report fields"):
                publish_g0_run(root, "malformed", report)
            self.assertEqual(list(root.rglob("*")), [])

    def test_existing_g0_index_blocks_before_any_run_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact_index.json").write_text(
                json.dumps(
                    {
                        "gates": {"g0": {"manifest": "existing/manifest.json", "sha256": "0" * 64}},
                        "schema_version": 1,
                    }
                )
            )
            with self.assertRaisesRegex(FileExistsError, "g0 artifact index entry already exists"):
                publish_g0_run(root, "other", valid_report())
            self.assertFalse((root / "certification").exists())

    def test_report_write_failure_removes_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "latency_meta_mdp.runtime.loop._write_json", side_effect=OSError("disk failure")
                ),
                self.assertRaisesRegex(OSError, "disk failure"),
            ):
                publish_g0_run(root, "write_failure", valid_report())
            self.assertFalse((root / "certification/g0/write_failure").exists())
            self.assertEqual(list((root / "certification/g0").glob(".*.building-*")), [])

    def test_index_replace_failure_rolls_back_new_run_and_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "artifact_index.json"
            original_index = {"gates": {}, "schema_version": 1}
            index_path.write_text(json.dumps(original_index))
            with (
                patch(
                    "latency_meta_mdp.runtime.loop.os.replace", side_effect=OSError("disk failure")
                ),
                self.assertRaisesRegex(OSError, "disk failure"),
            ):
                publish_g0_run(root, "index_failure", valid_report())
            self.assertEqual(json.loads(index_path.read_text()), original_index)
            self.assertFalse((root / "certification/g0/index_failure").exists())
            self.assertEqual(list(root.glob(".artifact_index.json.building-*")), [])
            self.assertEqual(list((root / "certification/g0").glob(".*.building-*")), [])


if __name__ == "__main__":
    unittest.main()
