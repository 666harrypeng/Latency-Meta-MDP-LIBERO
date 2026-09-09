from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    BranchCorpusConfig,
)
from latency_meta_mdp.belief.conditional_return_flow.source_corpus import (
    load_one_verified_source,
    load_verified_source_episodes,
    select_source_contexts,
    source_observation_sha256,
)
from latency_meta_mdp.temporal_contract import TemporalContract


@dataclass(frozen=True)
class FormalFixture:
    project_root: Path
    manifest: Path
    split_config: Path
    episode_dir: Path
    motion_config: Path
    motion_profile: dict[str, object]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_paths(project_root: Path, *, level: int) -> dict[str, Path]:
    return {
        "runtime": project_root / "configs/runtime/robosuite_v1.yaml",
        "task": project_root / "configs/task/dynamic_grasp_lift_l0.yaml",
        "motion": project_root / f"configs/motion/dynamic_grasp_lift_l{level}.yaml",
        "control": project_root / "configs/control/panda_osc_pose_delta_v1.yaml",
        "expert": project_root / "configs/expert/panda_ball_feedback_v1.yaml",
    }


def _formal_fixture(tmp_path: Path) -> FormalFixture:
    project_root = tmp_path / "project"
    config_paths = _config_paths(project_root, level=3)
    for name, path in config_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {name}\n", encoding="utf-8")

    split_config = project_root / "configs/data/formal_belief_train_val_v1.yaml"
    split_config.parent.mkdir(parents=True, exist_ok=True)
    split_config.write_text(
        """schema_version: 1
split_id: formal_belief_train_val_v1
source_seed_start: 1000
source_seed_count: 1
splits:
  train:
    start: 1000
    count: 1
""",
        encoding="utf-8",
    )

    root = tmp_path / "formal"
    episode_dir = root / "episodes/L3/seed_001000"
    episode_dir.mkdir(parents=True)
    boundary_count = 61
    transition_count = 60
    arrays = {
        "boundary_formal_tick": np.arange(boundary_count, dtype=np.int64),
        "boundary_time_us": np.arange(boundary_count, dtype=np.int64) * 20_000,
        "boundary_outcome_status": np.full(boundary_count, "running"),
        "agentview_rgb": np.zeros((boundary_count, 8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_rgb": np.ones((boundary_count, 8, 8, 3), dtype=np.uint8),
        "robot_qpos": np.zeros((boundary_count, 7), dtype=np.float64),
        "robot_qvel": np.zeros((boundary_count, 7), dtype=np.float64),
        "gripper_qpos": np.zeros((boundary_count, 2), dtype=np.float64),
        "gripper_qvel": np.zeros((boundary_count, 2), dtype=np.float64),
        "eef_position_world": np.zeros((boundary_count, 3), dtype=np.float64),
        "eef_orientation_matrix_world": np.repeat(
            np.eye(3, dtype=np.float64)[None], boundary_count, axis=0
        ),
        "object_pose": np.zeros((boundary_count, 7), dtype=np.float64),
        "object_velocity": np.zeros((boundary_count, 6), dtype=np.float64),
        "expert_action": np.zeros((transition_count, 7), dtype=np.float64),
        "expert_phase": np.asarray(
            ["pregrasp"] * 15 + ["approach"] * 15 + ["close"] * 15 + ["lift"] * 15
        ),
    }
    arrays_path = episode_dir / "arrays.npz"
    with arrays_path.open("xb") as handle:
        np.savez(handle, **arrays)
    events_path = episode_dir / "events.json"
    _write_json(events_path, {"events": [], "schema_version": 1})
    motion_profile = {"schema_version": 2, "type": "fixture", "segments": []}
    metadata = {
        "schema_version": 1,
        "episode_id": "l3-seed-001000-attempt-000",
        "task_id": "dynamic_grasp_lift",
        "instruction": "Grasp the moving ball and lift it.",
        "level": 3,
        "scene_seed": 1000,
        "motion_seed": 1000,
        "expert_seed": 1000,
        "physics_dt_us": 2_000,
        "formal_tick_us": 20_000,
        "action_contract_id": "panda_osc_pose_delta_v1",
        "action_dim": 7,
        "actuator_dim": 9,
        "expert_id": "panda_ball_feedback_v1",
        "record_profile": "belief",
        "config_sha256": {name: sha256_file(path) for name, path in config_paths.items()},
        "motion_profile": motion_profile,
        "terminal_status": "success",
        "terminal_reason": "lift_succeeded",
    }
    metadata_path = episode_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    nested = {
        "schema_version": 3,
        "format_id": "synchronized_episode_npz_v3",
        "record_profile": "belief",
        "episode_id": metadata["episode_id"],
        "boundary_count": boundary_count,
        "transition_count": transition_count,
        "physical_event_count": 0,
        "terminal_status": "success",
        "terminal_reason": "lift_succeeded",
        "artifacts": {
            name: sha256_file(episode_dir / name)
            for name in ("arrays.npz", "events.json", "metadata.json")
        },
    }
    episode_manifest = episode_dir / "manifest.json"
    _write_json(episode_manifest, nested)
    relative_manifest = episode_manifest.relative_to(root).as_posix()
    artifacts = {
        (episode_dir / name).relative_to(root).as_posix(): sha256_file(episode_dir / name)
        for name in ("arrays.npz", "events.json", "metadata.json", "manifest.json")
    }
    manifest = root / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "format_id": "panda_ball_formal_corpus_v1",
            "eligible": True,
            "implementation_dirty": False,
            "levels": [3],
            "admitted_episode_manifests": [relative_manifest],
            "artifacts": artifacts,
        },
    )
    return FormalFixture(
        project_root=project_root,
        manifest=manifest,
        split_config=split_config,
        episode_dir=episode_dir,
        motion_config=config_paths["motion"],
        motion_profile=motion_profile,
    )


def _load(fixture: FormalFixture):
    return load_verified_source_episodes(
        project_root=fixture.project_root,
        source_bulk_manifest=fixture.manifest,
        split_config_path=fixture.split_config,
        levels=(3,),
    )


def _temporal() -> TemporalContract:
    return TemporalContract(
        schema_version=1,
        contract_id="h50_e25_d20_k6_v1",
        formal_tick_us=20_000,
        control_frequency_hz=50,
        prediction_horizon=50,
        launch_trigger_horizon=25,
        maximum_delay_ticks=20,
        history_sample_count=6,
    )


def test_verified_source_rejects_nested_array_hash_mismatch(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)
    arrays_path = fixture.episode_dir / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"x")

    with pytest.raises(ValueError, match="hash mismatch"):
        _load(fixture)


def test_verified_source_uses_stored_motion_profile_and_explicit_split(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)

    episode = _load(fixture)[0]

    assert episode.motion_profile_mapping == fixture.motion_profile
    assert episode.level == 3
    assert episode.scene_seed == 1000
    assert episode.split == "train"
    assert episode.camera_width == 8
    assert episode.camera_height == 8
    assert episode.motion_config_path == fixture.motion_config


def test_load_one_verified_source_resolves_exact_level_seed(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)

    episode = load_one_verified_source(
        project_root=fixture.project_root,
        source_bulk_manifest=fixture.manifest,
        split_config_path=fixture.split_config,
        level=3,
        scene_seed=1000,
    )

    assert episode.episode_id == "l3-seed-001000-attempt-000"
    assert episode.level == 3
    assert episode.scene_seed == 1000


def test_verified_source_rejects_current_config_drift(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)
    fixture.motion_config.write_text("fixture: drifted-motion\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source config hash mismatch"):
        _load(fixture)


def test_level_scoped_load_does_not_open_unrequested_episode_artifacts(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    missing_l1 = "episodes/L1/seed_001000/manifest.json"
    manifest["levels"] = [1, 3]
    manifest["admitted_episode_manifests"].insert(0, missing_l1)
    manifest["artifacts"][missing_l1] = "0" * 64
    _write_json(fixture.manifest, manifest)

    episodes = _load(fixture)

    assert len(episodes) == 1
    assert episodes[0].level == 3


def test_seed_scoped_load_filters_before_opening_episode_artifacts(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    missing_seed = "episodes/L3/seed_001060/manifest.json"
    manifest["admitted_episode_manifests"].append(missing_seed)
    manifest["artifacts"][missing_seed] = "0" * 64
    _write_json(fixture.manifest, manifest)

    episodes = load_verified_source_episodes(
        project_root=fixture.project_root,
        source_bulk_manifest=fixture.manifest,
        split_config_path=fixture.split_config,
        levels=(3,),
        allowed_scene_seed_ranges=((1000, 1060),),
    )

    assert tuple(row.scene_seed for row in episodes) == (1000,)


@pytest.mark.parametrize(
    "ranges",
    (
        (),
        ((1060, 1000),),
        ((1000, 1060), (1050, 1070)),
        ((True, 1060),),
    ),
)
def test_seed_scoped_load_rejects_invalid_half_open_ranges(
    tmp_path: Path,
    ranges: tuple[tuple[int, int], ...],
) -> None:
    fixture = _formal_fixture(tmp_path)

    with pytest.raises(ValueError, match="seed ranges"):
        load_verified_source_episodes(
            project_root=fixture.project_root,
            source_bulk_manifest=fixture.manifest,
            split_config_path=fixture.split_config,
            levels=(3,),
            allowed_scene_seed_ranges=ranges,
        )


def test_source_selection_keeps_split_and_uses_central_eligible_phase_tick(
    tmp_path: Path,
) -> None:
    fixture = _formal_fixture(tmp_path)
    config = BranchCorpusConfig(
        schema_version=1,
        config_id="conditional_return_control_branch_corpus",
        contexts_per_phase_per_episode=1,
        phases=("pregrasp", "approach", "close", "lift"),
        arm_scale_factors=(0.5, 0.8),
        prefix_hold_ticks=(5, 10),
    )

    selection = select_source_contexts(
        episodes=_load(fixture), config=config, temporal=_temporal()
    )
    selected = selection.contexts

    assert selected
    assert {row.identity.split for row in selected} == {"train"}
    assert len({row.identity.source_context_id for row in selected}) == len(selected)
    assert all(row.identity.history_start_tick == row.identity.source_tick - 5 for row in selected)


def test_source_selection_accounts_for_phase_without_eligible_tick(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)
    config = BranchCorpusConfig(
        schema_version=1,
        config_id="conditional_return_control_branch_corpus",
        contexts_per_phase_per_episode=1,
        phases=("pregrasp", "approach", "close", "lift"),
        arm_scale_factors=(0.5, 0.8),
        prefix_hold_ticks=(5, 10),
    )

    selection = select_source_contexts(
        episodes=_load(fixture), config=config, temporal=_temporal()
    )

    assert selection.expected_slot_count == 4
    assert len(selection.contexts) == 2
    assert {row.identity.source_phase for row in selection.contexts} == {"approach", "close"}
    assert len(selection.exclusions) == 2
    assert {row.source_phase for row in selection.exclusions} == {"pregrasp", "lift"}
    exclusion = next(row for row in selection.exclusions if row.source_phase == "lift")
    assert exclusion.episode_id == "l3-seed-001000-attempt-000"
    assert exclusion.level == 3
    assert exclusion.scene_seed == 1000
    assert exclusion.split == "train"
    assert exclusion.source_phase == "lift"
    assert exclusion.reason == "no_phase_tick_in_source_interval"
    assert exclusion.source_interval_minimum == 25
    assert exclusion.source_interval_maximum == 35
    assert exclusion.phase_tick_count == 15
    assert exclusion.interval_phase_tick_count == 0
    assert exclusion.running_interval_phase_tick_count == 0


def test_source_observation_hash_changes_with_any_k6_input(tmp_path: Path) -> None:
    fixture = _formal_fixture(tmp_path)
    episode = _load(fixture)[0]
    source_tick = 30
    arrays = episode.load_arrays(
        names=(
            "boundary_formal_tick",
            "boundary_time_us",
            "agentview_rgb",
            "robot0_eye_in_hand_rgb",
            "robot_qpos",
            "robot_qvel",
            "gripper_qpos",
            "gripper_qvel",
        )
    )
    original = source_observation_sha256(arrays=arrays, source_tick=source_tick, k=6)
    tampered = {name: np.array(value, copy=True) for name, value in arrays.items()}
    tampered["robot_qpos"][source_tick - 2, 0] += 1e-4

    changed = source_observation_sha256(arrays=tampered, source_tick=source_tick, k=6)

    assert changed != original
