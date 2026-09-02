"""Invoke the isolated CuRobo qualification worker from the normal runtime."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from latency_meta_mdp.expert_realization.curobo_runtime import (
    QualificationContractError,
    load_qualification_report,
)

MAX_WORKER_DIAGNOSTIC_CHARACTERS = 1_024
DIAGNOSTIC_TRUNCATION_MARKER = "... [head/tail truncated] ..."


def _bounded_worker_diagnostic(worker: subprocess.CompletedProcess[str]) -> str:
    """Expose a concise failure reason without forwarding credentials or environment values."""
    diagnostic = worker.stderr if worker.stderr.strip() else worker.stdout
    diagnostic = re.sub(
        r"\b([A-Z][A-Z0-9_]{1,})\s*=\s*[^\s]+",
        r"\1=[REDACTED]",
        diagnostic,
    )
    diagnostic = re.sub(
        r"(?i)\b((?:api[_-]?key|access[_-]?token|token|password|secret)\s*[:=]\s*)\S+",
        r"\1[REDACTED]",
        diagnostic,
    )
    diagnostic = " ".join(piece for piece in diagnostic.split() if piece.isprintable())
    if not diagnostic:
        return "no worker diagnostic was emitted"
    if len(diagnostic) > MAX_WORKER_DIAGNOSTIC_CHARACTERS:
        retained_characters = MAX_WORKER_DIAGNOSTIC_CHARACTERS - len(
            DIAGNOSTIC_TRUNCATION_MARKER
        )
        head_characters = retained_characters // 2
        tail_characters = retained_characters - head_characters
        return (
            diagnostic[:head_characters]
            + DIAGNOSTIC_TRUNCATION_MARKER
            + diagnostic[-tail_characters:]
        )
    return diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify the structured expert planning runtime.")
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=Path(".venv-expert-realization/bin/python"),
        help="Python executable inside the isolated CuRobo environment.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/expert_realization/curobo_runtime_qualification.json"),
        help="Destination JSON report; stdout contains this path only on success.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".curobo-qualification-", dir=args.report_path.parent
    ) as temporary_directory:
        fresh_report_path = Path(temporary_directory) / "worker-report.json"
        worker = subprocess.run(
            [
                str(args.worker_python),
                "-m",
                "latency_meta_mdp.expert_realization.curobo_runtime",
                "--write-report",
                str(fresh_report_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if worker.returncode != 0:
            diagnostic = _bounded_worker_diagnostic(worker)
            raise QualificationContractError(
                f"isolated CuRobo worker exited without qualifying: {diagnostic}"
            )
        if not fresh_report_path.is_file():
            raise QualificationContractError("isolated CuRobo worker did not write a fresh report")
        qualification = load_qualification_report(fresh_report_path)
        if not qualification.eligible:
            raise QualificationContractError(
                "isolated CuRobo worker reported an ineligible runtime"
            )
        fresh_report_path.replace(args.report_path)
    print(args.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
