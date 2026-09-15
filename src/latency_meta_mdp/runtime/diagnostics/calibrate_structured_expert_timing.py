"""Collect the bounded non-training matched-K6 timing calibration artifact."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from latency_meta_mdp.data.collection.calibration import (
    TimingCalibrationAttemptRequest,
    TimingCalibrationRow,
    collect_timing_calibration,
    load_timing_calibration_attempt_result,
    run_timing_calibration_attempt_process,
    write_timing_calibration_attempt_request,
)


def main(
    argv: list[str] | None = None,
    *,
    attempt_runner: Callable[[TimingCalibrationAttemptRequest], TimingCalibrationRow] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)

    if attempt_runner is None:

        def run_attempt(request: TimingCalibrationAttemptRequest) -> TimingCalibrationRow:
            with tempfile.TemporaryDirectory(prefix="timing-calibration-") as directory:
                root = Path(directory)
                request_path = root / "request.json"
                result_path = root / "result.json"
                write_timing_calibration_attempt_request(request, request_path)
                run_timing_calibration_attempt_process(
                    request_path,
                    result_path,
                    worker_python=args.worker_python,
                )
                return load_timing_calibration_attempt_result(result_path)

        resolved_runner = run_attempt
    else:
        resolved_runner = attempt_runner

    def progress(value: dict[str, int]) -> None:
        print(
            f"completed={value['completed']} total={value['total']}",
            file=sys.stderr,
            flush=True,
        )

    manifest = collect_timing_calibration(
        project_root=Path.cwd(),
        config_path=args.config,
        target=args.target,
        attempt_runner=resolved_runner,
        on_progress=progress,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
