"""CuRobo v0.8.0 qualification worker and its fail-closed report contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

PINNED_CUROBO_COMMIT = "4ea77366ca48ee453e7df139e39fa6532af49f3b"
APPROVED_LICENSE_ID = "Apache-2.0"
APPROVED_TORCH_VERSION = "2.7.1+cu126"
APPROVED_TORCH_CUDA_VERSION = "12.6"
APPROVED_WARP_VERSION = "1.17.0"
APPROVED_CUDA_CORE_VERSION = "1.1.1"
PEAK_VRAM_MEASUREMENT_SCOPE = "torch_allocator_max_memory_allocated"
REPORT_FIELDS = frozenset(
    {
        "upstream_commit",
        "license_id",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "cuda_driver_version",
        "warp_version",
        "cuda_core_version",
        "gpu_name",
        "compute_capability",
        "official_franka_fk_passed",
        "motion_gen_smoke_passed",
        "peak_vram_bytes",
        "peak_vram_measurement_scope",
        "eligible",
    }
)


class QualificationContractError(ValueError):
    """The isolated worker did not provide an approved qualification report."""


@dataclass(frozen=True)
class CuroboRuntimeQualification:
    upstream_commit: str
    license_id: str
    python_version: str
    torch_version: str
    torch_cuda_version: str
    cuda_driver_version: str
    warp_version: str
    cuda_core_version: str
    gpu_name: str
    compute_capability: tuple[int, int]
    official_franka_fk_passed: bool
    motion_gen_smoke_passed: bool
    peak_vram_bytes: int
    peak_vram_measurement_scope: str
    eligible: bool


def qualification_from_json(payload: Mapping[str, Any]) -> CuroboRuntimeQualification:
    """Validate a worker report before the normal RoboSuite process consumes it."""
    missing = REPORT_FIELDS.difference(payload)
    if missing:
        fields = ", ".join(sorted(missing))
        raise QualificationContractError(f"missing qualification fields: {fields}")
    unknown = set(payload).difference(REPORT_FIELDS)
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise QualificationContractError(f"unknown qualification fields: {fields}")
    if payload["upstream_commit"] != PINNED_CUROBO_COMMIT:
        raise QualificationContractError("upstream_commit does not match the approved CuRobo pin")
    if payload["license_id"] != APPROVED_LICENSE_ID:
        raise QualificationContractError("license_id does not match Apache-2.0")

    python_version = payload["python_version"]
    python_parts = python_version.split(".") if isinstance(python_version, str) else []
    if (
        len(python_parts) != 3
        or not all(part.isdecimal() for part in python_parts)
        or python_parts[:2] != ["3", "10"]
    ):
        raise QualificationContractError("python_version must be an approved Python 3.10.x version")
    for field, expected in {
        "torch_version": APPROVED_TORCH_VERSION,
        "torch_cuda_version": APPROVED_TORCH_CUDA_VERSION,
        "warp_version": APPROVED_WARP_VERSION,
        "cuda_core_version": APPROVED_CUDA_CORE_VERSION,
        "peak_vram_measurement_scope": PEAK_VRAM_MEASUREMENT_SCOPE,
    }.items():
        if payload[field] != expected:
            raise QualificationContractError(f"{field} does not match the approved runtime")

    capability = payload["compute_capability"]
    if not isinstance(capability, (list, tuple)) or len(capability) != 2:
        raise QualificationContractError("compute_capability must contain exactly two integers")
    major, minor = capability
    if type(major) is not int or type(minor) is not int or major <= 0 or minor < 0:
        raise QualificationContractError("compute_capability must contain exactly two integers")

    string_fields = {
        "cuda_driver_version",
        "gpu_name",
    }
    for field in string_fields:
        if not isinstance(payload[field], str) or not payload[field]:
            raise QualificationContractError(f"{field} must be a non-empty string")
    for field in {"official_franka_fk_passed", "motion_gen_smoke_passed", "eligible"}:
        if not isinstance(payload[field], bool):
            raise QualificationContractError(f"{field} must be a boolean")
    driver_parts = payload["cuda_driver_version"].split(".")
    if len(driver_parts) < 2 or not all(part.isdecimal() for part in driver_parts):
        raise QualificationContractError(
            "cuda_driver_version must be a dotted numeric driver version"
        )
    if type(payload["peak_vram_bytes"]) is not int or payload["peak_vram_bytes"] < 0:
        raise QualificationContractError("peak_vram_bytes must be a non-negative integer")
    if payload["eligible"] and not (
        payload["official_franka_fk_passed"] and payload["motion_gen_smoke_passed"]
    ):
        raise QualificationContractError("eligible report requires FK and MotionGen qualification")

    return CuroboRuntimeQualification(
        upstream_commit=payload["upstream_commit"],
        license_id=payload["license_id"],
        python_version=payload["python_version"],
        torch_version=payload["torch_version"],
        torch_cuda_version=payload["torch_cuda_version"],
        cuda_driver_version=payload["cuda_driver_version"],
        warp_version=payload["warp_version"],
        cuda_core_version=payload["cuda_core_version"],
        gpu_name=payload["gpu_name"],
        compute_capability=(capability[0], capability[1]),
        official_franka_fk_passed=payload["official_franka_fk_passed"],
        motion_gen_smoke_passed=payload["motion_gen_smoke_passed"],
        peak_vram_bytes=payload["peak_vram_bytes"],
        peak_vram_measurement_scope=payload["peak_vram_measurement_scope"],
        eligible=payload["eligible"],
    )


def load_qualification_report(path: Path) -> CuroboRuntimeQualification:
    """Read and validate the JSON artifact emitted by the isolated worker."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationContractError(f"invalid qualification report: {path}") from error
    if not isinstance(payload, dict):
        raise QualificationContractError("qualification report must be a JSON object")
    return qualification_from_json(payload)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _verified_checkout() -> Path:
    checkout = _repository_root() / "third_party" / "curobo"
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    license_text = (checkout / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise QualificationContractError("CuRobo license fingerprint is not Apache-2.0")
    if commit != PINNED_CUROBO_COMMIT:
        raise QualificationContractError("CuRobo checkout does not match the approved v0.8.0 pin")
    worktree_status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if worktree_status:
        raise QualificationContractError("CuRobo checkout is not clean")
    return checkout


def _verify_imported_curobo_provenance() -> tuple[str, str]:
    """Bind imported CuRobo code to the clean, exact approved checkout."""
    import curobo

    checkout = _verified_checkout().resolve()
    if version("nvidia-curobo") != "0.8.0":
        raise QualificationContractError("installed nvidia-curobo version is not 0.8.0")
    imported_file = Path(curobo.__file__).resolve()
    try:
        imported_file.relative_to(checkout)
    except ValueError as error:
        raise QualificationContractError(
            "imported curobo package does not resolve under the approved checkout"
        ) from error
    return PINNED_CUROBO_COMMIT, APPROVED_LICENSE_ID


def _cuda_driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    driver = result.stdout.strip().splitlines()[0]
    if not driver:
        raise QualificationContractError("nvidia-smi did not report an NVIDIA driver version")
    return driver


def _run_official_franka_fk() -> bool:
    """Run the v0.8.0 bundled Franka FK API on CUDA, not an import-only check."""
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
    q = torch.zeros((1, robot.get_dof()), device="cuda", dtype=torch.float32)
    state = robot.compute_kinematics(JointState.from_position(q, joint_names=robot.joint_names))
    pose = state.tool_poses.get_link_pose(robot.tool_frames[0])
    return robot.get_dof() == 7 and bool(torch.isfinite(pose.position).all())


def _run_obstacle_free_motiongen_plan() -> bool:
    """Run CuRobo v0.8.0's MotionPlanner replacement for the MotionGen smoke plan."""
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState

    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot="franka.yml",
            self_collision_check=True,
            use_cuda_graph=False,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
        )
    )
    planner.warmup(enable_graph=False, num_warmup_iterations=1)
    current_state = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor(
            [[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32
        ),
    )
    result = planner.plan_pose(goal, current_state, enable_graph_attempt=0)
    return result is not None and bool(result.success.any())


def run_worker_qualification() -> CuroboRuntimeQualification:
    """Execute the CUDA-only checks from the isolated worker environment."""
    import torch
    import warp as wp

    if not torch.cuda.is_available():
        raise QualificationContractError("CuRobo qualification requires a CUDA-visible GPU")
    commit, license_id = _verify_imported_curobo_provenance()
    device = torch.cuda.get_device_properties(0)
    torch.cuda.reset_peak_memory_stats()
    official_franka_fk_passed = _run_official_franka_fk()
    motion_gen_smoke_passed = _run_obstacle_free_motiongen_plan()
    torch.cuda.synchronize()
    qualification = CuroboRuntimeQualification(
        upstream_commit=commit,
        license_id=license_id,
        python_version=".".join(map(str, sys.version_info[:3])),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda or "unavailable",
        cuda_driver_version=_cuda_driver_version(),
        warp_version=wp.__version__,
        cuda_core_version=version("cuda-core"),
        gpu_name=device.name,
        compute_capability=(device.major, device.minor),
        official_franka_fk_passed=official_franka_fk_passed,
        motion_gen_smoke_passed=motion_gen_smoke_passed,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        peak_vram_measurement_scope=PEAK_VRAM_MEASUREMENT_SCOPE,
        eligible=official_franka_fk_passed and motion_gen_smoke_passed,
    )
    return qualification_from_json(asdict(qualification))


def _worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the isolated CuRobo qualification worker.")
    parser.add_argument("--write-report", type=Path, required=True)
    args = parser.parse_args(argv)
    qualification = run_worker_qualification()
    args.write_report.parent.mkdir(parents=True, exist_ok=True)
    args.write_report.write_text(
        json.dumps(asdict(qualification), sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the parent CLI boundary.
    raise SystemExit(_worker_main())
