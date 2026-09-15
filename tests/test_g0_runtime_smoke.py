from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_panda_runtime_smoke_and_no_overwrite(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "latency_meta_mdp.envs.check",
        "--run-id",
        "g0_test",
        "--output-root",
        str(tmp_path),
    ]

    first = subprocess.run(command, check=True, capture_output=True, text=True)
    printed = json.loads(first.stdout)
    run_root = tmp_path / "certification/g0/g0_test"
    report_path = run_root / "runtime_report.json"
    manifest_path = run_root / "manifest.json"
    index_path = tmp_path / "artifact_index.json"

    assert printed == {
        "eligible": True,
        "manifest": "certification/g0/g0_test/manifest.json",
        "run_id": "g0_test",
    }
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert index_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 2
    assert report["runtime_version"] == "robosuite_native_v1"
    assert report["python_version"].startswith("3.10.")
    assert report["robosuite_version"] == "1.5.2"
    assert report["mujoco_version"] == "3.3.3"
    assert report["physics_dt_us"] == 2_000
    assert report["formal_tick_us"] == 20_000
    assert report["physics_steps_per_tick"] == 10
    assert report["control_freq_hz"] == 50
    assert report["compiled_timestep_seconds"] == 0.002
    assert report["control_timestep_seconds"] == 0.02
    assert report["stock_step_elapsed_seconds"] == 0.02
    assert report["integrator"] == "Euler"
    assert report["external_control_callback_present"] is False
    assert report["same_seed_initial_qpos_max_abs_error"] == 0.0
    assert report["same_seed_initial_object_qpos_max_abs_error"] == 0.0
    assert report["same_seed_full_qpos_max_abs_error"] == 0.0
    assert report["same_seed_full_qvel_max_abs_error"] == 0.0
    assert report["same_seed_full_act_max_abs_error"] == 0.0
    assert report["same_seed_sim_time_abs_error"] == 0.0
    assert report["integrator_configured_by_project"] is True
    assert report["config"]["camera_stride_ticks"] == 1
    assert report["config"]["compatibility_stride_ticks"] == 5
    assert len(report["config_sha256"]) == 64
    assert len(report["uv_lock_sha256"]) == 64
    assert len(report["implementation_source_sha256"]) == 64
    assert report["action_dim"] > 0
    assert report["eligible"] is True
    assert report["blockers"] == []

    manifest = json.loads(manifest_path.read_text())
    assert manifest["eligible"] is True
    assert manifest["artifacts"] == {
        "runtime_report.json": sha256(report_path),
    }
    index = json.loads(index_path.read_text())
    assert index == {
        "gates": {
            "g0": {
                "manifest": "certification/g0/g0_test/manifest.json",
                "sha256": sha256(manifest_path),
            }
        },
        "schema_version": 1,
    }

    before = {
        path.relative_to(tmp_path).as_posix(): sha256(path)
        for path in (report_path, manifest_path, index_path)
    }
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr
    after = {
        path.relative_to(tmp_path).as_posix(): sha256(path)
        for path in (report_path, manifest_path, index_path)
    }
    assert after == before

    different_run = [*command]
    different_run[different_run.index("g0_test")] = "g0_other"
    third = subprocess.run(different_run, check=False, capture_output=True, text=True)
    assert third.returncode != 0
    assert "g0 artifact index entry already exists" in third.stderr
    assert not (tmp_path / "certification/g0/g0_other").exists()
    assert not list(tmp_path.rglob("*.building-*"))
