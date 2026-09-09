from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    BranchRollout,
    ControlContinuationSpec,
    ExecutablePrefix,
    SourceContextIdentity,
    SourceSelectionExclusion,
    load_branch_corpus_config,
)

VALID_BRANCH_CONFIG = """schema_version: 1
config_id: conditional_return_control_branch_corpus
contexts_per_phase_per_episode: 1
phases:
  - pregrasp
  - approach
  - close
  - lift
arm_scale_factors:
  - 0.5
  - 0.8
prefix_hold_ticks:
  - 5
  - 10
"""


def _source_identity() -> SourceContextIdentity:
    digest = "a" * 64
    return SourceContextIdentity(
        source_context_id="L3-seed001000-tick000040",
        episode_id="L3-seed001000",
        level=3,
        scene_seed=1000,
        split="train",
        source_tick=40,
        source_phase="approach",
        history_start_tick=35,
        source_episode_manifest_sha256=digest,
        source_arrays_sha256="b" * 64,
        source_metadata_sha256="c" * 64,
        source_observation_sha256="d" * 64,
        motion_profile_sha256="e" * 64,
    )


def _valid_prefix() -> ExecutablePrefix:
    controls = np.zeros((20, 7), dtype=np.float32)
    controls[:, 6] = -1.0
    return ExecutablePrefix(
        controls=controls,
        from_active_buffer_mask=np.ones(20, dtype=bool),
        last_executed_gripper_command=-1.0,
    )


def test_branch_config_owns_only_branch_specific_fields(tmp_path: Path) -> None:
    path = tmp_path / "branch.yaml"
    path.write_text(VALID_BRANCH_CONFIG, encoding="utf-8")

    config = load_branch_corpus_config(path)

    assert config.config_id == "conditional_return_control_branch_corpus"
    assert config.phases == ("pregrasp", "approach", "close", "lift")
    assert config.arm_scale_factors == (0.5, 0.8)
    assert config.prefix_hold_ticks == (5, 10)
    assert not hasattr(config, "maximum_delay_ticks")


def test_branch_config_rejects_temporal_duplication(tmp_path: Path) -> None:
    path = tmp_path / "branch.yaml"
    path.write_text(VALID_BRANCH_CONFIG + "maximum_delay_ticks: 20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fields"):
        load_branch_corpus_config(path)


def test_branch_config_rejects_undefined_multi_context_selection(tmp_path: Path) -> None:
    path = tmp_path / "branch.yaml"
    path.write_text(
        VALID_BRANCH_CONFIG.replace(
            "contexts_per_phase_per_episode: 1",
            "contexts_per_phase_per_episode: 2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        load_branch_corpus_config(path)


def test_executable_prefix_requires_d20_shapes_and_immutable_float32() -> None:
    prefix = _valid_prefix()

    assert prefix.controls.shape == (20, 7)
    assert prefix.controls.dtype == np.float32
    assert prefix.from_active_buffer_mask.shape == (20,)
    assert prefix.from_active_buffer_mask.dtype == np.bool_
    assert not prefix.controls.flags.writeable
    assert not prefix.from_active_buffer_mask.flags.writeable


def test_branch_rollout_rejects_nonfinite_targets() -> None:
    targets = np.zeros((20, 22), dtype=np.float32)
    targets[3, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        BranchRollout(
            source=_source_identity(),
            branch=ControlContinuationSpec.nominal(),
            prefix=_valid_prefix(),
            target_states=targets,
            target_absorbing=np.zeros(20, dtype=bool),
            target_handoff_state=np.full(20, "driven"),
            target_outcome_status=np.full(20, "running"),
            source_replay_max_abs=0.0,
            source_fingerprint_match=True,
            nominal_future_valid=True,
            nominal_future_max_abs=0.0,
        )


def test_source_identity_rejects_non_digest_provenance() -> None:
    with pytest.raises(ValueError, match="SHA256"):
        SourceContextIdentity(
            source_context_id="L3-seed001000-tick000040",
            episode_id="L3-seed001000",
            level=3,
            scene_seed=1000,
            split="train",
            source_tick=40,
            source_phase="approach",
            history_start_tick=35,
            source_episode_manifest_sha256="not-a-digest",
            source_arrays_sha256="b" * 64,
            source_metadata_sha256="c" * 64,
            source_observation_sha256="d" * 64,
            motion_profile_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("episode_id", 123), ("source_context_id", 123), ("level", True)),
)
def test_source_identity_rejects_noncanonical_identity_types(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_source_identity(), **{field: value})


@pytest.mark.parametrize(
    ("episode_id", "level"),
    ((123, 1), ("l1-seed-001000-attempt-000", True)),
)
def test_selection_exclusion_rejects_non_string_id_or_boolean_level(
    episode_id: object,
    level: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SourceSelectionExclusion(
            episode_id=episode_id,
            level=level,
            scene_seed=1000,
            split="train",
            source_phase="pregrasp",
            reason="phase_absent",
            source_interval_minimum=25,
            source_interval_maximum=80,
            phase_tick_count=0,
            interval_phase_tick_count=0,
            running_interval_phase_tick_count=0,
        )
