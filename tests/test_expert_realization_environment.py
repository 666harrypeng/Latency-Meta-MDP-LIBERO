from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_CUROBO_COMMIT = "4ea77366ca48ee453e7df139e39fa6532af49f3b"
APPROVED_WARP_VERSION = "1.17.0"
APPROVED_CUDA_CORE_VERSION = "1.1.1"
VRAM_MEASUREMENT_SCOPE = "torch_allocator_max_memory_allocated"


def _valid_report() -> dict[str, object]:
    return {
        "upstream_commit": PINNED_CUROBO_COMMIT,
        "license_id": "Apache-2.0",
        "python_version": "3.10.20",
        "torch_version": "2.7.1+cu126",
        "torch_cuda_version": "12.6",
        "cuda_driver_version": "580.173.02",
        "warp_version": APPROVED_WARP_VERSION,
        "cuda_core_version": APPROVED_CUDA_CORE_VERSION,
        "gpu_name": "NVIDIA GeForce RTX 4080",
        "compute_capability": [8, 9],
        "official_franka_fk_passed": True,
        "motion_gen_smoke_passed": True,
        "peak_vram_bytes": 1,
        "peak_vram_measurement_scope": VRAM_MEASUREMENT_SCOPE,
        "eligible": True,
    }


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_pinned_curobo_checkout_has_approved_identity() -> None:
    """A changed upstream checkout must prevent expert-runtime qualification."""
    checkout = REPOSITORY_ROOT / "third_party" / "curobo"
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert commit == PINNED_CUROBO_COMMIT

    license_text = (checkout / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text


def test_qualification_report_rejects_missing_or_mismatched_contract_values() -> None:
    """A stale checkout or incomplete worker report must fail closed."""
    from latency_meta_mdp.expert_realization.curobo_runtime import (
        QualificationContractError,
        qualification_from_json,
    )

    report = _valid_report()
    qualification = qualification_from_json(report)
    assert qualification.upstream_commit == PINNED_CUROBO_COMMIT
    assert qualification.compute_capability == (8, 9)
    assert qualification.eligible is True

    mismatched = dict(report, upstream_commit="0" * 40)
    with pytest.raises(QualificationContractError, match="upstream_commit"):
        qualification_from_json(mismatched)

    missing = dict(report)
    del missing["warp_version"]
    with pytest.raises(QualificationContractError, match="warp_version"):
        qualification_from_json(missing)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("python_version", "3.11.0", "python_version"),
        ("torch_version", "2.7.1+cpu", "torch_version"),
        ("torch_cuda_version", "12.8", "torch_cuda_version"),
        ("warp_version", "1.16.0", "warp_version"),
        ("cuda_core_version", "1.0.0", "cuda_core_version"),
        ("compute_capability", [True, 9], "compute_capability"),
        ("compute_capability", [0, 9], "compute_capability"),
        ("peak_vram_measurement_scope", "total_process_vram", "peak_vram_measurement_scope"),
    ],
)
def test_qualification_report_rejects_runtime_drift_and_invalid_measurements(
    field: str, value: object, message: str
) -> None:
    """A report that does not bind the approved runtime must not be consumable."""
    from latency_meta_mdp.expert_realization.curobo_runtime import (
        QualificationContractError,
        qualification_from_json,
    )

    with pytest.raises(QualificationContractError, match=message):
        qualification_from_json(dict(_valid_report(), **{field: value}))


def test_qualification_report_rejects_unknown_fields() -> None:
    """A silently accepted schema extension could hide an incompatible worker contract."""
    from latency_meta_mdp.expert_realization.curobo_runtime import (
        QualificationContractError,
        qualification_from_json,
    )

    with pytest.raises(QualificationContractError, match="unknown"):
        qualification_from_json(dict(_valid_report(), unreviewed_runtime_detail="ignored"))


@pytest.mark.parametrize("capability", ([8, 0], [9, 0], [8, 9]))
def test_qualification_report_accepts_valid_zero_minor_compute_capability(
    capability: list[int],
) -> None:
    """Valid NVIDIA architectures such as sm_80 and sm_90 must remain qualifiable."""
    from latency_meta_mdp.expert_realization.curobo_runtime import qualification_from_json

    qualification = qualification_from_json(
        dict(_valid_report(), compute_capability=capability)
    )

    assert qualification.compute_capability == tuple(capability)


@pytest.mark.parametrize("capability", ([True, 0], ["8", 0], [0, 0], [8, -1]))
def test_qualification_report_rejects_invalid_compute_capability_components(
    capability: list[object],
) -> None:
    """Boolean, non-integer, nonpositive-major, and negative-minor values are invalid."""
    from latency_meta_mdp.expert_realization.curobo_runtime import (
        QualificationContractError,
        qualification_from_json,
    )

    with pytest.raises(QualificationContractError, match="compute_capability"):
        qualification_from_json(dict(_valid_report(), compute_capability=capability))


def test_non_worker_import_keeps_curobo_out_of_robosuite_process() -> None:
    """Adding a normal-process import of CuRobo must break this isolation contract."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import latency_meta_mdp.expert_realization; "
                "import latency_meta_mdp.expert_realization.curobo_runtime; "
                "import latency_meta_mdp.cli.qualify_structured_expert_planner; "
                "import sys; "
                "raise SystemExit(any(name == 'curobo' or name.startswith('curobo.') "
                "for name in sys.modules))"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_worker_module_help_has_no_runpy_reimport_warning() -> None:
    """Eager package imports must not make the worker's module execution ambiguous."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "latency_meta_mdp.expert_realization.curobo_runtime",
            "--help",
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr


def _write_worker(tmp_path: Path, script: str) -> Path:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "#!" + sys.executable + "\n"
        + script,
        encoding="utf-8",
    )
    worker.chmod(0o755)
    return worker


def _run_cli(worker: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "latency_meta_mdp.cli.qualify_structured_expert_planner",
            "--worker-python",
            str(worker),
            "--report-path",
            str(report_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_qualification_cli_publishes_only_fresh_validated_worker_report(tmp_path: Path) -> None:
    """A stale final artifact must not let a non-writing worker qualify successfully."""
    report_path = tmp_path / "qualification.json"
    report_path.write_text(json.dumps(_valid_report()), encoding="utf-8")

    result = _run_cli(Path("/bin/echo"), report_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert json.loads(report_path.read_text(encoding="utf-8")) == _valid_report()


def test_qualification_cli_captures_noisy_worker_output_and_publishes_report(
    tmp_path: Path,
) -> None:
    """Worker chatter must not violate the parent CLI's stdout-only result contract."""
    report_path = tmp_path / "qualification.json"
    worker = _write_worker(
        tmp_path,
        "from pathlib import Path\n"
        "import json\n"
        "print('untrusted worker output')\n"
        f"Path(__import__('sys').argv[-1]).write_text(json.dumps({_valid_report()!r}))\n",
    )

    result = _run_cli(worker, report_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{report_path}\n"
    assert result.stderr == ""
    assert json.loads(report_path.read_text(encoding="utf-8"))["eligible"] is True


def test_qualification_cli_rejects_an_ineligible_fresh_report(tmp_path: Path) -> None:
    """A worker that completes FK/planning unsuccessfully must not publish a final report."""
    report_path = tmp_path / "qualification.json"
    ineligible = dict(_valid_report(), eligible=False, motion_gen_smoke_passed=False)
    worker = _write_worker(
        tmp_path,
        "from pathlib import Path\n"
        "import json\n"
        f"Path(__import__('sys').argv[-1]).write_text(json.dumps({ineligible!r}))\n",
    )

    result = _run_cli(worker, report_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert not report_path.exists()


def test_qualification_cli_reports_bounded_sanitized_worker_stderr(tmp_path: Path) -> None:
    """A worker failure reason must survive without leaking a credential or unbounded output."""
    report_path = tmp_path / "qualification.json"
    worker = _write_worker(
        tmp_path,
        "import sys\n"
        "print('API_TOKEN=not-for-logs', file=sys.stderr)\n"
        "print('CUROBO_DRIVER_INCOMPATIBILITY ' + 'x' * 10000, file=sys.stderr)\n"
        "raise SystemExit(23)\n",
    )

    result = _run_cli(worker, report_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "CUROBO_DRIVER_INCOMPATIBILITY" in result.stderr
    assert "not-for-logs" not in result.stderr
    assert "API_TOKEN=[REDACTED]" in result.stderr
    assert len(result.stderr) < 2_000


def test_qualification_cli_preserves_tail_root_cause_in_bounded_diagnostic(
    tmp_path: Path,
) -> None:
    """Long traceback prefixes must not hide the final CUDA/CuRobo failure reason."""
    report_path = tmp_path / "qualification.json"
    worker = _write_worker(
        tmp_path,
        "import sys\n"
        "print('API_TOKEN=head-secret', file=sys.stderr)\n"
        "print('Traceback frame ' + 'h' * 1600, file=sys.stderr)\n"
        "print('PASSWORD=tail-secret', file=sys.stderr)\n"
        "print('CUROBO_ROOT_CAUSE_AT_TRACEBACK_END', file=sys.stderr)\n"
        "raise SystemExit(25)\n",
    )

    result = _run_cli(worker, report_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "CUROBO_ROOT_CAUSE_AT_TRACEBACK_END" in result.stderr
    assert "head-secret" not in result.stderr
    assert "tail-secret" not in result.stderr
    assert "API_TOKEN=[REDACTED]" in result.stderr
    assert "PASSWORD=[REDACTED]" in result.stderr
    assert "[head/tail truncated]" in result.stderr
    assert len(result.stderr) < 2_000


def test_qualification_cli_uses_worker_stdout_only_when_stderr_is_empty(tmp_path: Path) -> None:
    """A silent-stderr worker still exposes its bounded failure reason through parent stderr."""
    report_path = tmp_path / "qualification.json"
    worker = _write_worker(
        tmp_path,
        "print('CUROBO_RUNTIME_STDOUT_FAILURE')\nraise SystemExit(24)\n",
    )

    result = _run_cli(worker, report_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "CUROBO_RUNTIME_STDOUT_FAILURE" in result.stderr


def test_integration_marker_is_registered_for_strict_collection(tmp_path: Path) -> None:
    """An unregistered integration marker must fail before real qualification is scheduled."""
    integration_test = tmp_path / "test_marker_contract.py"
    integration_test.write_text(
        "import pytest\n\n@pytest.mark.integration\ndef test_marker_contract():\n    pass\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--strict-markers",
            "--collect-only",
            "-q",
            "-c",
            "pyproject.toml",
            str(integration_test),
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1 test collected" in result.stdout


def test_expert_realization_environment_is_ignored_before_creation() -> None:
    """The dedicated environment must never appear as user-owned repository state."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".venv-expert-realization/"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
