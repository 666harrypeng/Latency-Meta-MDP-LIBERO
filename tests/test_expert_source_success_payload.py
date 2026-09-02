from __future__ import annotations

from pathlib import Path

import pytest
from expert_realization_test_support import make_formal_source_episode


def _recording():
    from latency_meta_mdp.expert_realization.qualification import QualificationDecision
    from latency_meta_mdp.expert_realization.safety import ActualRolloutSafetyReport
    from latency_meta_mdp.expert_realization.source_corpus.recording import (
        QualifiedSourceRecording,
    )

    report = ActualRolloutSafetyReport(
        terminal_success=True,
        physical_handoff=True,
        phase_order_valid=True,
        minimum_non_contact_environment_clearance_m=0.02,
        maximum_intentional_contact_penetration_m=0.008,
        maximum_pad_ball_impulse_ns=0.03,
        unintended_pregrasp_ball_contacts=0,
        other_link_ball_contacts=0,
        robot_environment_contacts=0,
        robot_self_contacts=0,
        minimum_joint_position_margin_rad=0.4,
        maximum_joint_velocity_fraction=0.3,
        maximum_eef_speed_mps=0.2,
        maximum_eef_acceleration_mps2=3.0,
        maximum_eef_jerk_mps3=101.0,
        maximum_reference_tracking_error_m=0.06,
        maximum_pregrasp_translation_error_m=0.02,
        maximum_pregrasp_rotation_error_degrees=0.8,
        minimum_osc_action=-1.0,
        maximum_osc_action=1.0,
        pre_handoff_saturation_fraction=0.0,
    )
    return QualifiedSourceRecording(
        episode=make_formal_source_episode(
            boundary_count=4, camera_height=256, camera_width=256
        ),
        safety_report=report,
        qualification=QualificationDecision(eligible=True, failures=()),
    )


def test_success_payload_round_trips_typed_episode_and_lossless_images(
    tmp_path: Path,
) -> None:
    """Break caught: resume changes source values or drops actual-physics qualification."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_corpus_config,
    )
    from latency_meta_mdp.expert_realization.source_corpus.parquet import (
        episode_to_frame_table,
    )
    from latency_meta_mdp.expert_realization.source_corpus.success_payload import (
        load_source_success_payload,
        write_source_success_payload,
    )

    config = load_source_corpus_config(
        Path("configs/source_corpus/panda_ball_source_parquet.yaml")
    )
    recording = _recording()
    target = tmp_path / "success"
    manifest = write_source_success_payload(
        target=target,
        recording=recording,
        source_config=config,
        strategy_parameters={"family": "canonical_direct", "lead_seconds": 0.2},
        selected_planner_fingerprint="a" * 64,
    )
    loaded = load_source_success_payload(target, source_config=config)

    assert manifest == target / "manifest.json"
    assert {path.name for path in target.iterdir()} == {
        "episode.parquet",
        "record.json",
        "manifest.json",
    }
    assert loaded.episode.metadata == recording.episode.metadata
    assert loaded.episode.physical_events == recording.episode.physical_events
    assert loaded.strategy_parameters == {
        "family": "canonical_direct",
        "lead_seconds": 0.2,
    }
    assert loaded.selected_planner_fingerprint == "a" * 64
    assert loaded.qualification["eligible"] is True
    assert loaded.qualification["failures"] == []
    assert loaded.qualification["actual_rollout_safety"]["terminal_success"] is True
    assert episode_to_frame_table(loaded.episode, config=config).equals(
        episode_to_frame_table(recording.episode, config=config)
    )


def test_success_payload_rejects_overwrite_tamper_and_nonqualified_input(
    tmp_path: Path,
) -> None:
    """Break caught: failed or mutable data enters the resumable success namespace."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_corpus_config,
    )
    from latency_meta_mdp.expert_realization.source_corpus.success_payload import (
        load_source_success_payload,
        write_source_success_payload,
    )

    config = load_source_corpus_config(
        Path("configs/source_corpus/panda_ball_source_parquet.yaml")
    )
    kwargs = dict(
        target=tmp_path / "success",
        recording=_recording(),
        source_config=config,
        strategy_parameters={"family": "canonical_direct"},
        selected_planner_fingerprint="b" * 64,
    )
    write_source_success_payload(**kwargs)
    with pytest.raises(FileExistsError):
        write_source_success_payload(**kwargs)
    with pytest.raises(TypeError, match="QualifiedSourceRecording"):
        write_source_success_payload(**{**kwargs, "recording": object()})

    (tmp_path / "success/record.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_source_success_payload(tmp_path / "success", source_config=config)
