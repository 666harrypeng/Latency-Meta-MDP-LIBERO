from __future__ import annotations

import numpy as np
import pytest
import yaml

from latency_meta_mdp.belief.flow.data_sufficiency import (
    build_nested_episode_subsets,
    load_flow_data_sufficiency_config,
    summarize_episode_motion,
)


def test_nested_episode_subsets_are_deterministic_nested_and_disjoint() -> None:
    train_ids = tuple(f"episode-{seed}" for seed in range(1000, 1180))
    validation_ids = tuple(f"episode-{seed}" for seed in range(1180, 1200))

    first = build_nested_episode_subsets(
        training_episode_ids=train_ids,
        validation_episode_ids=validation_ids,
        sizes=(60, 120, 180),
        seed=20260828,
    )
    repeat = build_nested_episode_subsets(
        training_episode_ids=train_ids,
        validation_episode_ids=validation_ids,
        sizes=(60, 120, 180),
        seed=20260828,
    )

    assert first == repeat
    assert set(first[60]) < set(first[120]) < set(first[180])
    assert set(first[180]) == set(train_ids)
    assert not set(first[180]).intersection(validation_ids)


def test_nested_episode_subsets_reject_validation_leakage() -> None:
    with pytest.raises(ValueError, match="overlap"):
        build_nested_episode_subsets(
            training_episode_ids=("a", "b", "c"),
            validation_episode_ids=("c", "d"),
            sizes=(2, 3),
            seed=7,
        )


def test_episode_motion_summary_tracks_geometry_speed_and_handoff() -> None:
    metadata = {
        "episode_id": "l3-seed-001000-attempt-000",
        "level": 3,
        "scene_seed": 1000,
        "motion_profile": {
            "type": "piecewise_polynomial",
            "segments": [
                {
                    "kind": "line",
                    "start_xy": [0.0, 0.0],
                    "end_xy": [0.02, 0.01],
                },
                {
                    "kind": "cubic",
                    "start_xy": [0.02, 0.01],
                    "end_xy": [0.01, 0.03],
                },
                {
                    "kind": "line",
                    "start_xy": [0.01, 0.03],
                    "end_xy": [0.05, 0.04],
                },
            ],
        },
    }
    events = {
        "events": [
            {"kind": "first_contact", "time_us": 900_000},
            {"kind": "handoff", "time_us": 1_200_000},
            {"kind": "success", "time_us": 2_000_000},
        ]
    }
    arrays = {
        "boundary_time_us": np.arange(6, dtype=np.int64) * 20_000,
        "commanded_motion_position": np.asarray(
            [
                [0.00, 0.00, 0.83],
                [0.01, 0.00, 0.83],
                [0.02, 0.01, 0.83],
                [0.02, 0.02, 0.83],
                [0.01, 0.03, 0.83],
                [0.00, 0.03, 0.83],
            ]
        ),
        "commanded_motion_velocity": np.asarray(
            [
                [0.10, 0.00, 0.0],
                [0.08, 0.04, 0.0],
                [0.00, 0.10, 0.0],
                [-0.08, 0.04, 0.0],
                [-0.10, 0.00, 0.0],
                [-0.10, 0.00, 0.0],
            ]
        ),
        "commanded_motion_acceleration": np.asarray(
            [
                [0.0, 0.1, 0.0],
                [-0.1, 0.1, 0.0],
                [-0.1, 0.0, 0.0],
                [-0.1, -0.1, 0.0],
                [0.0, -0.1, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        "commanded_motion_segment_index": np.asarray([0, 0, 1, 1, 2, 2]),
    }

    summary = summarize_episode_motion(
        metadata=metadata,
        events=events,
        arrays=arrays,
    )

    assert summary.episode_id == metadata["episode_id"]
    assert summary.level == 3
    assert summary.profile_type == "piecewise_polynomial"
    assert summary.segment_count == 3
    assert summary.segment_kinds == ("line", "cubic", "line")
    assert summary.segment_transition_count == 2
    assert summary.handoff_time_seconds == 1.2
    assert summary.success_time_seconds == 2.0
    assert summary.planned_start_xy == (0.0, 0.0)
    assert summary.planned_end_xy == (0.05, 0.04)
    assert summary.handoff_xy == (0.0, 0.03)
    assert summary.pre_handoff_path_length_m > summary.pre_handoff_chord_length_m
    assert summary.pre_handoff_path_chord_ratio > 1.0
    assert summary.maximum_speed_m_s == pytest.approx(0.1)
    assert summary.positive_curvature_fraction > 0.0
    assert summary.maximum_velocity_jump_m_s > 0.0


def test_data_sufficiency_config_locks_nested_scaling_protocol(tmp_path) -> None:
    path = tmp_path / "audit.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "audit_id": "dinov3_flow_belief_data_scaling_v1",
                "subset_sizes": [60, 120, 180],
                "subset_seed": 20260828,
                "model_seeds": [20260829, 20260830],
                "workspace_bin_count_per_axis": 4,
                "minimum_occupied_bin_count": 5,
                "scaling_improvement_trigger": 0.10,
                "model_seed_disagreement_trigger": 0.10,
                "additional_tranche_size": 100,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_flow_data_sufficiency_config(path)

    assert config.subset_sizes == (60, 120, 180)
    assert config.model_seeds == (20260829, 20260830)
    assert config.additional_tranche_size == 100
