"""G0 RoboSuite / MuJoCo runtime probe and immutable evidence writer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.envs.config import RuntimeConfig, load_runtime_config
from latency_meta_mdp.envs.resources import load_resource_manifest
from latency_meta_mdp.io.paths import repository_root

_PROJECT_ROOT = repository_root()
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    paths = [
        _PROJECT_ROOT / "pyproject.toml",
        _PROJECT_ROOT / "uv.lock",
        _PROJECT_ROOT / "configs/runtime/robosuite_v1.yaml",
        _PROJECT_ROOT / "assets/resource_manifest.json",
        *sorted((_PROJECT_ROOT / "src/latency_meta_mdp").rglob("*.py")),
    ]
    for path in paths:
        relative = path.relative_to(_PROJECT_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be one safe path component")


def _max_abs(left: Any, right: Any) -> float:
    left_array = np.asarray([] if left is None else left, dtype=float)
    right_array = np.asarray([] if right is None else right, dtype=float)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right_array)))


def _make_environment(seed: int, config: RuntimeConfig) -> Any:
    import mujoco
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    controller = load_composite_controller_config(robot="Panda")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=config.control_freq_hz,
        lite_physics=config.lite_physics,
        horizon=2,
        ignore_done=False,
        hard_reset=False,
        seed=seed,
    )
    env.sim.model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    env.sim.forward()
    return env


def validate_runtime_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate report schema and derive eligibility rather than trusting the caller."""
    required = {
        "action_dim",
        "compiled_timestep_seconds",
        "config",
        "config_sha256",
        "control_freq_hz",
        "control_timestep_seconds",
        "controller",
        "environment",
        "external_control_callback_present",
        "formal_tick_us",
        "full_state_fields",
        "implementation_revision",
        "implementation_source_sha256",
        "integrator",
        "integrator_configured_by_project",
        "mujoco_version",
        "physics_dt_us",
        "physics_steps_per_tick",
        "python_version",
        "renderer",
        "resource_count",
        "resource_manifest_sha256",
        "robosuite_version",
        "runtime_version",
        "same_seed_full_act_max_abs_error",
        "same_seed_full_qpos_max_abs_error",
        "same_seed_full_qvel_max_abs_error",
        "same_seed_initial_object_qpos_max_abs_error",
        "same_seed_initial_qpos_max_abs_error",
        "same_seed_sim_time_abs_error",
        "schema_version",
        "seed",
        "stock_step_elapsed_seconds",
        "uv_lock_sha256",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"missing runtime report fields: {missing}")
    config = RuntimeConfig.from_mapping(report["config"])
    blockers: list[str] = []

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def finite_number(name: str) -> float | None:
        value = report[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            blockers.append(f"{name}_invalid_type")
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            blockers.append(f"{name}_nonfinite")
            return None
        return numeric

    if not is_int(report["schema_version"]) or report["schema_version"] != 2:
        blockers.append("wrong_report_schema_version")
    if report["runtime_version"] != config.runtime_version:
        blockers.append("runtime_version_mismatch")
    if report["config_sha256"] != _json_sha256(config.to_mapping()):
        blockers.append("config_hash_mismatch")
    if not is_int(report["physics_dt_us"]) or report["physics_dt_us"] != config.physics_dt_us:
        blockers.append("physics_dt_contract_mismatch")
    if not is_int(report["formal_tick_us"]) or report["formal_tick_us"] != config.formal_tick_us:
        blockers.append("formal_tick_contract_mismatch")
    if (
        not is_int(report["physics_steps_per_tick"])
        or report["physics_steps_per_tick"] != config.physics_steps_per_tick
    ):
        blockers.append("physics_step_ratio_mismatch")
    if not is_int(report["control_freq_hz"]) or report["control_freq_hz"] != config.control_freq_hz:
        blockers.append("control_frequency_mismatch")
    if not isinstance(report["python_version"], str) or report["python_version"].split(".")[:2] != [
        "3",
        "10",
    ]:
        blockers.append("python_version_mismatch")
    if report["robosuite_version"] != "1.5.2":
        blockers.append("robosuite_version_mismatch")
    if report["mujoco_version"] != "3.3.3":
        blockers.append("mujoco_version_mismatch")
    compiled_timestep = finite_number("compiled_timestep_seconds")
    control_timestep = finite_number("control_timestep_seconds")
    stock_step_elapsed = finite_number("stock_step_elapsed_seconds")
    if compiled_timestep is not None and abs(compiled_timestep - 0.002) > 1e-12:
        blockers.append("compiled_timestep_mismatch")
    if control_timestep is not None and abs(control_timestep - 0.02) > 1e-12:
        blockers.append("control_timestep_mismatch")
    if stock_step_elapsed is not None and abs(stock_step_elapsed - 0.02) > 1e-12:
        blockers.append("stock_step_elapsed_mismatch")
    if (
        report["integrator"] != "Euler"
        or not isinstance(report["integrator_configured_by_project"], bool)
        or not report["integrator_configured_by_project"]
    ):
        blockers.append("integrator_is_not_project_configured_euler")
    if (
        not isinstance(report["external_control_callback_present"], bool)
        or report["external_control_callback_present"]
    ):
        blockers.append("external_control_callback_present")
    for name in (
        "same_seed_full_act_max_abs_error",
        "same_seed_full_qpos_max_abs_error",
        "same_seed_full_qvel_max_abs_error",
        "same_seed_initial_object_qpos_max_abs_error",
        "same_seed_initial_qpos_max_abs_error",
        "same_seed_sim_time_abs_error",
    ):
        value = finite_number(name)
        if value is not None and value != 0.0:
            blockers.append(f"{name}_nonzero")
    if report["full_state_fields"] != ["time", "qpos", "qvel", "act"]:
        blockers.append("full_state_fields_mismatch")
    for name in (
        "config_sha256",
        "implementation_source_sha256",
        "resource_manifest_sha256",
        "uv_lock_sha256",
    ):
        if not isinstance(report[name], str) or not _SHA256_RE.fullmatch(report[name]):
            blockers.append(f"{name}_malformed")
    if not isinstance(report["implementation_revision"], str) or not _REVISION_RE.fullmatch(
        report["implementation_revision"]
    ):
        blockers.append("implementation_revision_malformed")
    expected_hashes = {
        "implementation_source_sha256": _source_sha256(),
        "resource_manifest_sha256": _sha256(_PROJECT_ROOT / "assets/resource_manifest.json"),
        "uv_lock_sha256": _sha256(_PROJECT_ROOT / "uv.lock"),
    }
    for name, expected in expected_hashes.items():
        if report[name] != expected:
            blockers.append(f"{name.removesuffix('_sha256')}_hash_mismatch")
    if report["implementation_revision"] != _git_revision():
        blockers.append("implementation_revision_mismatch")
    if report["environment"] != "Lift" or report["renderer"] != "headless-no-camera":
        blockers.append("environment_identity_mismatch")
    if report["controller"] != "Panda/default_panda/BASIC":
        blockers.append("controller_identity_mismatch")
    if not is_int(report["action_dim"]) or report["action_dim"] <= 0:
        blockers.append("invalid_action_dimension")
    if not is_int(report["resource_count"]) or report["resource_count"] < 0:
        blockers.append("invalid_resource_count")
    else:
        resources = load_resource_manifest(
            _PROJECT_ROOT / "assets/resource_manifest.json", repository_root=_PROJECT_ROOT
        )
        if report["resource_count"] != len(resources):
            blockers.append("resource_count_mismatch")
    if not is_int(report["seed"]):
        blockers.append("invalid_seed")

    validated = {key: value for key, value in report.items() if key not in {"eligible", "blockers"}}
    validated["blockers"] = sorted(set(blockers))
    validated["eligible"] = not validated["blockers"]
    return validated


def probe_runtime(seed: int = 7) -> dict[str, Any]:
    """Run a real headless Panda reset and one stock formal-time action."""
    import mujoco

    config = load_runtime_config(_PROJECT_ROOT / "configs/runtime/robosuite_v1.yaml")
    resources = load_resource_manifest(
        _PROJECT_ROOT / "assets/resource_manifest.json", repository_root=_PROJECT_ROOT
    )
    env_a = None
    env_b = None
    with redirect_stdout(sys.stderr):
        import robosuite

        try:
            env_a = _make_environment(seed, config)
            env_b = _make_environment(seed, config)
            env_a.reset()
            env_b.reset()
            state_a = env_a.sim.get_state()
            state_b = env_b.sim.get_state()
            act_a = np.array(env_a.sim.data.act, dtype=float, copy=True)
            act_b = np.array(env_b.sim.data.act, dtype=float, copy=True)

            robot_indexes_a = np.asarray(env_a.robots[0]._ref_joint_pos_indexes, dtype=int)
            robot_indexes_b = np.asarray(env_b.robots[0]._ref_joint_pos_indexes, dtype=int)
            robot_qpos_a = np.array(env_a.sim.data.qpos[robot_indexes_a], dtype=float, copy=True)
            robot_qpos_b = np.array(env_b.sim.data.qpos[robot_indexes_b], dtype=float, copy=True)
            object_qpos_a = np.array(
                env_a.sim.data.get_joint_qpos(env_a.cube.joints[0]), dtype=float, copy=True
            )
            object_qpos_b = np.array(
                env_b.sim.data.get_joint_qpos(env_b.cube.joints[0]), dtype=float, copy=True
            )

            before_time = float(env_a.sim.data.time)
            env_a.step(np.zeros(env_a.action_dim, dtype=float))
            after_time = float(env_a.sim.data.time)
            compiled_timestep = float(env_a.sim.model.opt.timestep)
            control_timestep = float(env_a.control_timestep)
            integrator_value = int(env_a.sim.model.opt.integrator)
            callback_present = mujoco.get_mjcb_control() is not None
            action_dim = int(env_a.action_dim)
            controller_name = str(env_a.robots[0].composite_controller.name)
        finally:
            if env_a is not None:
                env_a.close()
            if env_b is not None:
                env_b.close()

    raw_report = {
        "action_dim": action_dim,
        "compiled_timestep_seconds": round(compiled_timestep, 12),
        "config": config.to_mapping(),
        "config_sha256": _json_sha256(config.to_mapping()),
        "control_freq_hz": config.control_freq_hz,
        "control_timestep_seconds": round(control_timestep, 12),
        "controller": f"Panda/default_panda/{controller_name}",
        "environment": "Lift",
        "external_control_callback_present": callback_present,
        "formal_tick_us": config.formal_tick_us,
        "full_state_fields": ["time", "qpos", "qvel", "act"],
        "implementation_revision": _git_revision(),
        "implementation_source_sha256": _source_sha256(),
        "integrator": "Euler" if integrator_value == 0 else f"MuJoCoIntegrator({integrator_value})",
        "integrator_configured_by_project": True,
        "mujoco_version": mujoco.__version__,
        "physics_dt_us": config.physics_dt_us,
        "physics_steps_per_tick": config.physics_steps_per_tick,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "renderer": "headless-no-camera",
        "resource_count": len(resources),
        "resource_manifest_sha256": _sha256(_PROJECT_ROOT / "assets/resource_manifest.json"),
        "robosuite_version": robosuite.__version__,
        "runtime_version": config.runtime_version,
        "same_seed_full_act_max_abs_error": _max_abs(act_a, act_b),
        "same_seed_full_qpos_max_abs_error": _max_abs(state_a.qpos, state_b.qpos),
        "same_seed_full_qvel_max_abs_error": _max_abs(state_a.qvel, state_b.qvel),
        "same_seed_initial_object_qpos_max_abs_error": _max_abs(object_qpos_a, object_qpos_b),
        "same_seed_initial_qpos_max_abs_error": _max_abs(robot_qpos_a, robot_qpos_b),
        "same_seed_sim_time_abs_error": abs(float(state_a.time) - float(state_b.time)),
        "schema_version": 2,
        "seed": seed,
        "stock_step_elapsed_seconds": round(after_time - before_time, 12),
        "uv_lock_sha256": _sha256(_PROJECT_ROOT / "uv.lock"),
    }
    return validate_runtime_report(raw_report)


def _load_index(output_root: Path) -> dict[str, Any]:
    index_path = output_root / "artifact_index.json"
    if not index_path.exists():
        return {"gates": {}, "schema_version": 1}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not isinstance(index.get("gates"), dict):
        raise ValueError("artifact index must use schema_version 1 and a gates mapping")
    return index


def preflight_g0_output(output_root: Path, run_id: str) -> None:
    _validate_run_id(run_id)
    run_root = output_root / "certification/g0" / run_id
    if run_root.exists():
        raise FileExistsError(f"run already exists: {run_root}")
    if "g0" in _load_index(output_root)["gates"]:
        raise FileExistsError("g0 artifact index entry already exists")


def publish_g0_run(output_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    """Publish one immutable G0 bundle and update the canonical gate index."""
    validated = validate_runtime_report(report)
    preflight_g0_output(output_root, run_id)
    certification_root = output_root / "certification/g0"
    certification_root.mkdir(parents=True, exist_ok=True)
    run_root = certification_root / run_id
    staging = certification_root / f".{run_id}.building-{os.getpid()}"
    try:
        staging.mkdir()
        report_path = staging / "runtime_report.json"
        manifest_path = staging / "manifest.json"
        _write_json(report_path, validated)
        manifest = {
            "artifacts": {"runtime_report.json": _sha256(report_path)},
            "blockers": list(validated["blockers"]),
            "eligible": bool(validated["eligible"]),
            "gate": "g0",
            "run_id": run_id,
            "schema_version": 2,
        }
        _write_json(manifest_path, manifest)
        staging.rename(run_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if validated["eligible"]:
        index = _load_index(output_root)
        index["gates"]["g0"] = {
            "manifest": f"certification/g0/{run_id}/manifest.json",
            "sha256": _sha256(run_root / "manifest.json"),
        }
        index_path = output_root / "artifact_index.json"
        index_staging = output_root / f".artifact_index.json.building-{os.getpid()}"
        try:
            _write_json(index_staging, index)
            if index_path.exists():
                os.replace(index_staging, index_path)
            else:
                index_staging.rename(index_path)
        except BaseException:
            index_staging.unlink(missing_ok=True)
            shutil.rmtree(run_root, ignore_errors=True)
            raise
    return run_root / "manifest.json"
