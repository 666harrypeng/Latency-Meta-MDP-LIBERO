"""Run and publish the Panda stationary endpoint reachability calibration."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.endpoint_reachability import (
    EndpointReachabilityResult,
    load_endpoint_reachability_spec,
    render_endpoint_reachability_svg,
    run_endpoint_reachability_calibration,
)

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/control/panda_endpoint_reachability_v1.yaml"),
    )
    return parser


def _run_root(*, output_root: Path, run_id: str) -> Path:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be one safe path component")
    return output_root / "calibration" / "endpoint_reachability" / run_id


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _publish(*, run_root: Path, report: dict) -> Path:
    if run_root.exists():
        raise FileExistsError(f"endpoint reachability run already exists: {run_root}")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    staging = run_root.parent / f".{run_root.name}.building-{os.getpid()}"
    try:
        staging.mkdir()
        report_path = staging / "endpoint_reachability_report.json"
        heatmap_path = staging / "endpoint_reachability_heatmap.svg"
        _write_json(report_path, report)
        _write_text(heatmap_path, render_endpoint_reachability_svg(report))
        artifacts = {
            report_path.name: sha256_file(report_path),
            heatmap_path.name: sha256_file(heatmap_path),
        }
        manifest_path = staging / "manifest.json"
        _write_json(
            manifest_path,
            {
                "artifacts": artifacts,
                "blockers": report["blockers"],
                "calibration_id": report["calibration_id"],
                "eligible": report["eligible"],
                "implementation_dirty": report["implementation_dirty"],
                "schema_version": 1,
            },
        )
        staging.rename(run_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return run_root / "manifest.json"


def _progress(index: int, total: int, result: EndpointReachabilityResult) -> None:
    print(
        f"[{index:03d}/{total:03d}] {result.point.point_id} "
        f"x={result.point.x_m:+.3f} y={result.point.y_m:+.3f} "
        f"{result.terminal_status}:{result.terminal_reason}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = _run_root(output_root=args.output_root, run_id=args.run_id)
    if run_root.exists():
        raise FileExistsError(f"endpoint reachability run already exists: {run_root}")
    spec = load_endpoint_reachability_spec(args.calibration_config)
    report = run_endpoint_reachability_calibration(
        project_root=args.project_root,
        spec=spec,
        on_result=_progress,
    )
    manifest = _publish(run_root=run_root, report=report)
    print(
        json.dumps(
            {
                "eligible": report["eligible"],
                "manifest": manifest.relative_to(args.output_root).as_posix(),
                "successful_point_count": report["successful_point_count"],
                "point_count": report["point_count"],
                "run_id": args.run_id,
            },
            sort_keys=True,
        )
    )
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
