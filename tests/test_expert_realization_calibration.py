from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

CALIBRATION_CONFIG = Path("configs/analysis/panda_ball_structured_timing_calibration.yaml")


def _row(task_index: int, level: int, *, success: bool = True):
    from latency_meta_mdp.data.collection.calibration import TimingCalibrationRow

    optional = {
        "pregrasp_tick": 50,
        "approach_tick": 50,
        "close_tick": 83,
        "lift_start_tick": 92,
        "first_contact_time_us": 1_742_000,
        "stable_contact_time_us": 1_838_000,
        "handoff_time_us": 1_840_000,
        "lift_threshold_time_us": 2_480_000,
        "terminal_time_us": 2_580_000,
        "close_distance_m": 0.012,
        "close_relative_speed_mps": 0.04,
        "handoff_relative_speed_mps": 0.02,
    }
    if not success:
        optional = {name: None for name in optional}
    return TimingCalibrationRow(
        logical_task_index=task_index,
        level=level,
        terminal_status="success" if success else "failure",
        terminal_reason="lift_succeeded" if success else "grasp_deadline_missed",
        **optional,
        saturation_fraction=0.01,
        maximum_eef_speed_mps=0.30,
        maximum_eef_acceleration_mps2=2.0,
        maximum_eef_jerk_mps3=40.0,
        k6_planning_start_sha256="a" * 64,
    )


def _implementation():
    from latency_meta_mdp.data.collection.recording_contracts import ImplementationIdentity

    return ImplementationIdentity(revision="test-revision", source_sha256="f" * 64, dirty=False)


def test_calibration_config_is_fixed_nontraining_and_complete() -> None:
    """Break caught: calibration scope silently authorizes training or changes sample coverage."""
    from latency_meta_mdp.data.collection.calibration import (
        load_timing_calibration_config,
    )

    config = load_timing_calibration_config(CALIBRATION_CONFIG)
    assert config.calibration_id == "panda-ball-feedback-calibration-v1"
    assert config.logical_task_indices == tuple(range(50))
    assert config.levels == (1, 2, 3)
    assert config.quantiles == (0.10, 0.50, 0.90)
    assert config.attempt_count == 150
    assert config.bounded_review_only is True
    assert config.training_authorized is False


def test_calibration_attempt_request_binds_seed_and_config_identity() -> None:
    """Break caught: caller can change task seed or runtime config behind one request identity."""
    from latency_meta_mdp.data.collection.calibration import (
        TimingCalibrationAttemptRequest,
    )

    request = TimingCalibrationAttemptRequest(
        calibration_id="panda-ball-feedback-calibration-v1",
        logical_task_index=0,
        level=1,
        master_task_seed=12985087823104956951,
        task_config_sha256="a" * 64,
        motion_config_sha256="b" * 64,
        runtime_config_sha256="c" * 64,
        controller_config_sha256="d" * 64,
        expert_config_sha256="e" * 64,
    )
    assert type(request).from_mapping(request.to_mapping()) == request
    with pytest.raises(ValueError, match="seed"):
        replace(request, master_task_seed=0)


def test_calibration_attempt_request_and_result_use_strict_file_protocol(tmp_path: Path) -> None:
    """Break caught: parent/worker exchange unbound Python objects or ambiguous JSON files."""
    from latency_meta_mdp.data.collection.calibration import (
        TimingCalibrationAttemptRequest,
        load_timing_calibration_attempt_result,
        write_timing_calibration_attempt_request,
        write_timing_calibration_attempt_result,
    )

    request = TimingCalibrationAttemptRequest(
        calibration_id="panda-ball-feedback-calibration-v1",
        logical_task_index=0,
        level=1,
        master_task_seed=12985087823104956951,
        task_config_sha256="a" * 64,
        motion_config_sha256="b" * 64,
        runtime_config_sha256="c" * 64,
        controller_config_sha256="d" * 64,
        expert_config_sha256="e" * 64,
    )
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_timing_calibration_attempt_request(request, request_path)
    assert (
        TimingCalibrationAttemptRequest.from_mapping(json.loads(request_path.read_text()))
        == request
    )

    row = _row(0, 1)
    write_timing_calibration_attempt_result(row, result_path)
    assert load_timing_calibration_attempt_result(result_path) == row
    result_path.write_text(result_path.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_timing_calibration_attempt_result(result_path)


def test_calibration_row_requires_complete_success_and_preserves_partial_failure_events() -> None:
    """Break caught: success drops metrics or failure diagnostics are discarded/misordered."""
    success = _row(0, 1)
    assert type(success).from_mapping(success.to_mapping()) == success
    with pytest.raises(ValueError, match="success metrics"):
        replace(success, handoff_time_us=None)

    failure = _row(0, 1, success=False)
    assert type(failure).from_mapping(failure.to_mapping()) == failure
    partial = replace(
        failure,
        first_contact_time_us=1_742_000,
        terminal_time_us=1_900_000,
    )
    assert type(partial).from_mapping(partial.to_mapping()) == partial
    with pytest.raises(ValueError, match="monotonic"):
        replace(
            failure,
            first_contact_time_us=1_900_000,
            terminal_time_us=1_800_000,
        )


def test_calibration_preserves_physics_contact_that_precedes_formal_close_decision() -> None:
    """Break caught: calibration censors a successful approach-contact timing pattern."""
    row = replace(
        _row(0, 1),
        close_tick=80,
        first_contact_time_us=1_532_000,
        stable_contact_time_us=1_790_000,
        handoff_time_us=1_800_000,
        lift_start_tick=90,
        lift_threshold_time_us=2_440_000,
        terminal_time_us=2_540_000,
    )
    assert row.first_contact_time_us < row.close_tick * 20_000
    assert type(row).from_mapping(row.to_mapping()) == row


def test_report_accounts_for_all_150_rows_and_uses_only_success_quantiles() -> None:
    """Break caught: failed attempts enter timing quantiles or one task/level row disappears."""
    from latency_meta_mdp.data.collection.calibration import (
        TimingCalibrationReport,
        build_timing_calibration_report,
        load_timing_calibration_config,
    )

    config = load_timing_calibration_config(CALIBRATION_CONFIG)
    rows = tuple(
        _row(task_index, level, success=not (task_index == 0 and level == 3))
        for task_index in config.logical_task_indices
        for level in config.levels
    )
    report = build_timing_calibration_report(
        config=config,
        request_sha256="e" * 64,
        rows=rows,
        implementation=_implementation(),
    )

    assert len(report.rows) == 150
    assert report.success_count == 149
    assert report.failure_count == 1
    assert report.per_level_success_count == {1: 50, 2: 50, 3: 49}
    assert report.per_level_quantiles[3]["handoff_time_us"] == (
        1_840_000.0,
        1_840_000.0,
        1_840_000.0,
    )
    assert report.artifact_eligible is True
    assert report.training_authorized is False
    assert TimingCalibrationReport.from_mapping(report.to_mapping()) == report


def test_report_rejects_duplicate_or_missing_task_level_rows() -> None:
    """Break caught: a self-consistent summary hides incomplete calibration accounting."""
    from latency_meta_mdp.data.collection.calibration import (
        build_timing_calibration_report,
        load_timing_calibration_config,
    )

    config = load_timing_calibration_config(CALIBRATION_CONFIG)
    rows = tuple(
        _row(task_index, level)
        for task_index in config.logical_task_indices
        for level in config.levels
    )
    for malformed in (rows[:-1], rows[:-1] + (rows[0],)):
        with pytest.raises(ValueError, match="row universe"):
            build_timing_calibration_report(
                config=config,
                request_sha256="e" * 64,
                rows=malformed,
                implementation=_implementation(),
            )


def test_calibration_report_publication_is_verified_and_no_overwrite(tmp_path: Path) -> None:
    """Break caught: a non-training calibration report is mutable or loads after corruption."""
    from latency_meta_mdp.data.collection.calibration import (
        build_timing_calibration_report,
        load_timing_calibration_config,
        load_verified_timing_calibration_report,
        publish_timing_calibration_report,
    )

    config = load_timing_calibration_config(CALIBRATION_CONFIG)
    report = build_timing_calibration_report(
        config=config,
        request_sha256="e" * 64,
        rows=tuple(
            _row(task_index, level)
            for task_index in config.logical_task_indices
            for level in config.levels
        ),
        implementation=_implementation(),
    )
    target = tmp_path / "calibration"
    manifest = publish_timing_calibration_report(report, target)
    assert manifest == target / "manifest.json"
    assert load_verified_timing_calibration_report(target) == report
    assert load_verified_timing_calibration_report(target).implementation == _implementation()
    with pytest.raises(FileExistsError):
        publish_timing_calibration_report(report, target)

    payload = bytearray((target / "report.json").read_bytes())
    payload[-2] ^= 1
    (target / "report.json").write_bytes(payload)
    with pytest.raises(ValueError, match="hash"):
        load_verified_timing_calibration_report(target)


def test_importing_calibration_contract_does_not_load_historical_expert() -> None:
    """Break caught: normal structured imports acquire the old behavioral expert."""
    script = """
import json, sys
import latency_meta_mdp.data.collection.calibration
print(json.dumps({'historical_loaded': 'latency_meta_mdp.envs.expert' in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src"},
    )
    assert completed.stdout.strip() == '{"historical_loaded": false}'


def test_calibration_collection_materializes_exact_universe_before_publication(
    tmp_path: Path,
) -> None:
    """Break caught: orchestration publishes incomplete or reordered attempt accounting."""
    from latency_meta_mdp.data.collection.calibration import (
        collect_timing_calibration,
        load_verified_timing_calibration_report,
    )

    requests = []
    progress = []

    def run_attempt(request):
        requests.append(request)
        return _row(request.logical_task_index, request.level)

    target = tmp_path / "calibration"
    manifest = collect_timing_calibration(
        project_root=Path.cwd(),
        config_path=CALIBRATION_CONFIG,
        target=target,
        attempt_runner=run_attempt,
        on_progress=progress.append,
    )
    report = load_verified_timing_calibration_report(target)

    assert manifest == target / "manifest.json"
    assert len(requests) == 150
    assert [(row.logical_task_index, row.level) for row in requests[:4]] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
    ]
    assert len(report.rows) == 150
    assert progress[-1] == {"completed": 150, "total": 150}


def test_scoped_provenance_ignores_unrelated_files_but_detects_source_drift(
    tmp_path: Path,
) -> None:
    """Break caught: unrelated files invalidate calibration or source edits go unnoticed."""
    from latency_meta_mdp.data.collection.calibration import (
        collect_scoped_implementation_identity,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

    clean = collect_scoped_implementation_identity(repo, source_paths=("source.py",))
    assert clean.dirty is False
    (repo / "unrelated.py").write_text("UNRELATED = True\n", encoding="utf-8")
    unrelated = collect_scoped_implementation_identity(repo, source_paths=("source.py",))
    assert unrelated == clean

    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = collect_scoped_implementation_identity(repo, source_paths=("source.py",))
    assert changed.dirty is True
    assert changed.source_sha256 != clean.source_sha256
