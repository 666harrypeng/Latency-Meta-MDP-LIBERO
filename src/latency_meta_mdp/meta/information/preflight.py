"""Read-only checks before local information-scheduler experiments."""

import json
import shutil
import subprocess
from pathlib import Path

import yaml

from latency_meta_mdp.meta.information.config import identity_hash, load_config, sha256


def runtime_dependencies(root, config):
    """Resolve the non-Python inputs loaded by the fixed collection services."""
    root = Path(root).resolve()
    paths = {
        config["task_config"],
        config["rtc_config"],
        "configs/contracts/policy/pi05_state16_h50.yaml",
        "configs/models/jepa/model.yaml",
        "configs/models/jepa/stride4_80ms_history_160ms.yaml",
        "configs/runtime/control/panda_osc_pose_delta_conveyor_v2.yaml",
        "configs/runtime/latency/truncated_beta_8_65_400ms_v1.yaml",
        "configs/runtime/latency/truncated_beta_family_8_65_400ms_v1.yaml",
    }
    for name in ("run.json", "checkpoint.json", "model.safetensors"):
        paths.add(str(Path(config["assets"]["paths"]["decoder_dir"]) / name))
    for source, keys, relative in (
        (
            "configs/models/jepa/model.yaml",
            ("source_protocol_config", "vision_config", "upstream_reference_config"),
            True,
        ),
        (config["rtc_config"], ("temporal_contract",), True),
        ("configs/contracts/policy/pi05_state16_h50.yaml", ("temporal_contract",), True),
        (config["task_config"], ("reference_scene", "action_contract_path"), False),
    ):
        file = root / source
        raw = yaml.safe_load(file.read_text())
        for key in keys:
            dependency = ((file.parent if relative else root) / raw[key]).resolve()
            paths.add(str(dependency.relative_to(root)))
    paths.update(
        str(p.relative_to(root)) for p in (root / "patches/openpi").glob("000[1-9]-*.patch")
    )
    return paths


def verify_runtime_dependencies(root, config):
    missing = runtime_dependencies(root, config) - config["assets"]["files"].keys()
    if missing:
        raise ValueError(f"runtime dependency outside asset binding: {sorted(missing)}")


def verify_assets(root, assets):
    for relative, expected in assets["files"].items():
        path = Path(root) / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"asset content differs: {relative}")
    if "paths" in assets:
        for kind in ("conditioned", "clean"):
            checkpoint = Path(root) / assets["paths"][kind]
            actual = {str(p.relative_to(root)) for p in checkpoint.rglob("*") if p.is_file()}
            covered = {
                name for name in assets["files"] if Path(name).is_relative_to(assets["paths"][kind])
            }
            if actual != covered:
                raise ValueError("checkpoint file coverage differs from asset manifest")
    return {"verified_files": len(assets["files"]), "asset_binding": identity_hash(assets)}


def blocking_gpu_processes(active):
    """A synchronous evaluator may share only its explicitly parked learner context."""
    import os

    parent = os.environ.get("CONVEYOR_IDLE_LEARNER_PID")
    if parent is None:
        return active
    if parent != str(os.getppid()):
        raise ValueError("idle learner must be the evaluator's direct parent")
    return "\n".join(
        line for line in active.splitlines() if line.split(",", 1)[0].strip() != parent
    )


def preflight(config_path, output_dir, *, root):
    root, output_dir = Path(root), Path(output_dir)
    config = load_config(config_path)
    verify_runtime_dependencies(root, config)
    report = verify_assets(root, config["assets"])
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    from latency_meta_mdp.meta.information.resources import gpu_processes

    blockers, active = gpu_processes()
    sources = {
        str(p.relative_to(root)): sha256(p)
        for directory in ("src/latency_meta_mdp", "scripts")
        for p in sorted((root / directory).rglob("*.py"))
    }
    report.update(
        config_sha256=sha256(config_path),
        code_hash=identity_hash(sources),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        gpu=gpu,
        compute_processes=blockers,
        all_compute_processes=active,
        disk_free_bytes=shutil.disk_usage(root).free,
        split_counts={k: len(v) for k, v in config["splits"].items()},
        scene_inventory_audited=False,
        scope="asset/integration preflight; scene provenance audit required before collection",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
