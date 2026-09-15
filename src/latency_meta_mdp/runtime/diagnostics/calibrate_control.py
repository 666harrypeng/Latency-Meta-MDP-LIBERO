"""Run and publish the selected Panda control calibration."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from latency_meta_mdp.envs.control_calibration import run_control_calibration
from latency_meta_mdp.io.artifacts import sha256_file

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--control-config",
        type=Path,
        default=Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml"),
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/runtime/control/panda_control_calibration_v1.yaml"),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/tasks/moving_ball/task/dynamic_grasp_lift_l0.yaml"),
    )
    return parser


def _run_root(*, output_root: Path, run_id: str) -> Path:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be one safe path component")
    return output_root / "calibration" / "control" / run_id


def _preflight(*, output_root: Path, run_id: str) -> Path:
    run_root = _run_root(output_root=output_root, run_id=run_id)
    if run_root.exists():
        raise FileExistsError(f"control calibration run already exists: {run_root}")
    return run_root


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish(*, run_root: Path, report: dict) -> Path:
    run_root.parent.mkdir(parents=True, exist_ok=True)
    staging = run_root.parent / f".{run_root.name}.building-{os.getpid()}"
    try:
        staging.mkdir()
        report_path = staging / "control_report.json"
        _write_json(report_path, report)
        _write_json(
            staging / "manifest.json",
            {
                "artifacts": {"control_report.json": sha256_file(report_path)},
                "blockers": report["blockers"],
                "calibration_id": report["calibration_id"],
                "contract_id": report["contract_id"],
                "eligible": report["eligible"],
                "schema_version": 1,
            },
        )
        staging.rename(run_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return run_root / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = _preflight(output_root=args.output_root, run_id=args.run_id)
    report = run_control_calibration(
        control_config_path=args.control_config,
        calibration_config_path=args.calibration_config,
        task_config_path=args.task_config,
    )
    manifest = _publish(run_root=run_root, report=report)
    print(
        json.dumps(
            {
                "eligible": report["eligible"],
                "manifest": manifest.relative_to(args.output_root).as_posix(),
                "run_id": args.run_id,
            },
            sort_keys=True,
        )
    )
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
