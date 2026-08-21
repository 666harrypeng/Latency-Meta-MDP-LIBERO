"""Run and publish the G1 synchronous-clock calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.artifacts import preflight_gate_output, publish_gate_report
from latency_meta_mdp.calibration import run_g1_calibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prerequisites = preflight_gate_output(
        output_root=args.output_root, gate="g1", run_id=args.run_id
    )
    report = run_g1_calibration(
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        prerequisites=prerequisites,
    )
    manifest_path = publish_gate_report(
        output_root=args.output_root,
        gate="g1",
        run_id=args.run_id,
        report_name="timing_report.json",
        report=report,
    )
    print(
        json.dumps(
            {
                "eligible": report["eligible"],
                "manifest": manifest_path.relative_to(args.output_root).as_posix(),
                "run_id": args.run_id,
            },
            sort_keys=True,
        )
    )
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
