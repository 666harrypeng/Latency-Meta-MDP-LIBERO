"""Run and publish the G0 RoboSuite / MuJoCo runtime probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from latency_meta_mdp.runtime import preflight_g0_output, probe_runtime, publish_g0_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        preflight_g0_output(args.output_root, args.run_id)
        report = probe_runtime(seed=args.seed)
        manifest = publish_g0_run(args.output_root, args.run_id, report)
    except (FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
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
    if not report["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
