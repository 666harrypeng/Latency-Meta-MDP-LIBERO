from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _structured_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "expert_id": "panda_ball_structured_v1",
        "action_contract_id": "panda_osc_pose_delta_v1",
        "decision_source_tick": 5,
        "shared_prefix_policy": "settle_open_hold_v1",
        "families": [
            "canonical_direct",
            "early_high_arc",
            "lateral_arc",
            "time_shifted_smooth",
        ],
        "samples_per_family": 2,
        "interception_lead_seconds": [0.14, 0.26],
        "pregrasp_height_m": [0.08, 0.13],
        "lateral_offset_m": [0.015, 0.04],
        "tracking_error_clip_m": [0.022, 0.038],
        "close_dwell_ticks": [0, 1, 2, 3, 4],
        "lift_lateral_offset_m": [0.0, 0.02],
        "lift_vertical_offset_m": [0.14, 0.19],
        "interception_tick_ranges": {
            "canonical_direct": [75, 105],
            "early_high_arc": [60, 85],
            "lateral_arc": [75, 110],
            "time_shifted_smooth": [100, 125],
        },
        "fixed_orientation": True,
        "rotation_action_variation": False,
        "iid_per_tick_action_noise": False,
        "subseed_tags": ["strategy", "keypose", "planner", "timing"],
    }


def _write_yaml(tmp_path: Path, name: str, mapping: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_structured_config_rejects_unknown_and_missing_fields(
    tmp_path: Path, mutation: str
) -> None:
    """Break caught: a semantic field may otherwise be silently accepted/defaulted."""
    from latency_meta_mdp.expert_realization.config import load_structured_expert_config

    mapping = _structured_mapping()
    if mutation == "unknown":
        mapping["unapproved_default"] = True
    else:
        del mapping["decision_source_tick"]

    with pytest.raises(ValueError):
        load_structured_expert_config(_write_yaml(tmp_path, "structured.yaml", mapping))


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_pilot_config_rejects_unknown_and_missing_fields(tmp_path: Path, mutation: str) -> None:
    """Break caught: pilot authorization or scope could otherwise silently change."""
    from latency_meta_mdp.expert_realization.config import load_pilot_config

    mapping: dict[str, object] = {
        "schema_version": 1,
        "pilot_id": "panda-ball-structured-paired-4000-4019-v1",
        "task_instance_seed_start": 4000,
        "task_instance_seed_stop": 4020,
        "levels": [1, 2, 3],
        "families": [
            "canonical_direct",
            "early_high_arc",
            "lateral_arc",
            "time_shifted_smooth",
        ],
        "samples_per_family": 2,
        "maximum_infrastructure_attempts_per_realization": 2,
        "record_profile": "pilot_debug",
        "camera_width": 256,
        "camera_height": 256,
        "review_video_fps": 50,
        "bounded_review_only": True,
        "formal_training_authorized": False,
        "formal_dino_cache_authorized": False,
        "jepa_training_authorized": False,
        "policy_training_authorized": False,
        "meta_policy_training_authorized": False,
    }
    if mutation == "unknown":
        mapping["training_authorized"] = True
    else:
        del mapping["bounded_review_only"]

    with pytest.raises(ValueError):
        load_pilot_config(_write_yaml(tmp_path, "pilot.yaml", mapping))


def test_checked_in_configs_preserve_paired_pilot_and_exact_bounds() -> None:
    """Break caught: a change to the checked-in pilot loses its specified realization domain."""
    from latency_meta_mdp.expert_realization.config import (
        load_curobo_planner_config,
        load_pilot_config,
        load_pilot_gate_config,
        load_structured_expert_config,
    )

    structured = load_structured_expert_config(
        PROJECT_ROOT / "configs/expert_realization/panda_ball_structured.yaml"
    )
    curobo = load_curobo_planner_config(
        PROJECT_ROOT / "configs/expert_realization/curobo_panda.yaml"
    )
    pilot = load_pilot_config(PROJECT_ROOT / "configs/collection/panda_ball_structured_pilot.yaml")
    gate = load_pilot_gate_config(
        PROJECT_ROOT / "configs/analysis/panda_ball_structured_pilot_gate.yaml"
    )

    assert structured.decision_source_tick == 5
    assert structured.families == (
        "canonical_direct",
        "early_high_arc",
        "lateral_arc",
        "time_shifted_smooth",
    )
    assert structured.interception_tick_ranges == {
        "canonical_direct": (75, 105),
        "early_high_arc": (60, 85),
        "lateral_arc": (75, 110),
        "time_shifted_smooth": (100, 125),
    }
    assert structured.interception_lead_seconds == (0.14, 0.26)
    assert structured.pregrasp_height_m == (0.08, 0.13)
    assert structured.lateral_offset_m == (0.015, 0.04)
    assert structured.tracking_error_clip_m == (0.022, 0.038)
    assert structured.close_dwell_ticks == (0, 1, 2, 3, 4)
    assert structured.lift_lateral_offset_m == (0.0, 0.02)
    assert structured.lift_vertical_offset_m == (0.14, 0.19)
    assert curobo.planner_candidate_count == 8
    assert curobo.planner_invocation_timeout_seconds == 5.0
    assert pilot.seeds == tuple(range(4000, 4020))
    assert pilot.levels == (1, 2, 3)
    assert pilot.families == structured.families
    assert pilot.samples_per_family == 2
    assert gate.pregrasp_tracking_translation_m == 0.01
    assert gate.pregrasp_tracking_rotation_degrees == 2.0
    assert gate.near_duplicate_d20_xyz_action_rms == 0.02
    assert gate.near_duplicate_eef_frechet_m == 0.005


@pytest.mark.parametrize(
    ("loader_name", "path", "field", "wrong_value"),
    [
        (
            "load_structured_expert_config",
            "configs/expert_realization/panda_ball_structured.yaml",
            "decision_source_tick",
            5.0,
        ),
        (
            "load_curobo_planner_config",
            "configs/expert_realization/curobo_panda.yaml",
            "planner_candidate_count",
            True,
        ),
        (
            "load_pilot_config",
            "configs/collection/panda_ball_structured_pilot.yaml",
            "camera_width",
            256.0,
        ),
        (
            "load_pilot_gate_config",
            "configs/analysis/panda_ball_structured_pilot_gate.yaml",
            "eef_speed_mps",
            1,
        ),
    ],
)
def test_config_loaders_reject_wrong_yaml_scalar_types(
    tmp_path: Path, loader_name: str, path: str, field: str, wrong_value: object
) -> None:
    """Break caught: YAML-equivalent values have different serialized config identities."""
    import latency_meta_mdp.expert_realization.config as config_module

    mapping = yaml.safe_load((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    mapping[field] = wrong_value

    with pytest.raises((TypeError, ValueError)):
        getattr(config_module, loader_name)(_write_yaml(tmp_path, "wrong-type.yaml", mapping))
