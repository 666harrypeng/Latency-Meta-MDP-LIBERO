from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _episode(*, seed: int = 1000, boundary_count: int = 8):
    tick = np.arange(boundary_count, dtype=np.float64)
    robot_qpos = np.repeat(tick[:, None], 7, axis=1)
    robot_qvel = robot_qpos + 0.5
    gripper_qpos = np.stack((tick + 0.04, tick - 0.04), axis=1)
    gripper_qvel = np.stack((tick + 0.01, tick - 0.01), axis=1)
    object_pose = np.zeros((boundary_count, 7), dtype=np.float64)
    object_pose[:, :3] = np.stack((tick, tick + 1.0, tick + 2.0), axis=1)
    object_velocity = np.zeros((boundary_count, 6), dtype=np.float64)
    object_velocity[:, :3] = np.stack((tick + 3.0, tick + 4.0, tick + 5.0), axis=1)
    relative = np.stack((tick + 6.0, tick + 7.0, tick + 8.0), axis=1)
    handoff = np.full(boundary_count, "driven", dtype="<U8")
    handoff[-1] = "physical"
    return SimpleNamespace(
        episode_id=f"l1-seed-{seed:06d}-attempt-000",
        level=1,
        scene_seed=seed,
        boundary_count=boundary_count,
        deployment=SimpleNamespace(
            robot_qpos=robot_qpos,
            robot_qvel=robot_qvel,
            gripper_qpos=gripper_qpos,
            gripper_qvel=gripper_qvel,
        ),
        supervision=SimpleNamespace(
            object_pose=object_pose,
            object_velocity=object_velocity,
            relative_geometry=relative,
            handoff_state=handoff,
        ),
    )


@pytest.mark.parametrize(
    ("seed", "expected"),
    ((1000, "train"), (1019, "train"), (1020, "validation"), (1021, "validation"),
     (1022, "holdout"), (1024, "holdout")),
)
def test_first_tranche_probe_split_is_episode_seed_based(seed: int, expected: str) -> None:
    from latency_meta_mdp.vision_probe_data import first_tranche_probe_split

    assert first_tranche_probe_split(seed).value == expected


def test_first_tranche_probe_split_rejects_seed_outside_the_bank() -> None:
    from latency_meta_mdp.vision_probe_data import first_tranche_probe_split

    with pytest.raises(ValueError, match="outside"):
        first_tranche_probe_split(1025)


@pytest.mark.parametrize(
    ("seed", "expected"),
    ((1000, "train"), (1159, "train"), (1160, "validation"), (1179, "validation"),
     (1180, "holdout"), (1199, "holdout")),
)
def test_formal_probe_split_uses_160_20_20_episode_banks(
    seed: int,
    expected: str,
) -> None:
    from latency_meta_mdp.vision_probe_data import formal_probe_split

    assert formal_probe_split(seed).value == expected


def test_probe_indices_use_unpadded_six_boundary_histories() -> None:
    from latency_meta_mdp.vision_probe_data import build_probe_sample_indices

    indices = build_probe_sample_indices(
        episode=_episode(),
        history_sample_count=6,
    )

    assert [(item.history_start_tick, item.source_tick) for item in indices] == [
        (0, 5),
        (1, 6),
        (2, 7),
    ]
    assert indices[0].split.value == "train"
    assert indices[-1].pre_handoff is False


def test_probe_sample_materializes_deployment_history_and_nine_dimensional_target() -> None:
    from latency_meta_mdp.vision_probe_data import (
        build_probe_sample_indices,
        materialize_probe_sample,
    )

    episode = _episode()
    features = (
        np.arange(
        episode.boundary_count * 2 * 196 * 384,
            dtype=np.int64,
        )
        % 1024
    ).astype(np.float16).reshape(episode.boundary_count, 2, 196, 384)
    index = build_probe_sample_indices(
        episode=episode,
        history_sample_count=6,
    )[0]

    sample = materialize_probe_sample(
        episode=episode,
        features=features,
        index=index,
        history_sample_count=6,
    )

    assert sample.vision_history.shape == (6, 2, 196, 384)
    assert sample.vision_history.dtype == np.float16
    assert sample.robot_proprio_history.shape == (6, 16)
    np.testing.assert_array_equal(sample.robot_proprio_history[-1, :7], 5.0)
    np.testing.assert_array_equal(sample.robot_proprio_history[-1, 7:14], 5.5)
    assert sample.robot_proprio_history[-1, 14] == pytest.approx(0.08)
    assert sample.robot_proprio_history[-1, 15] == pytest.approx(0.02)
    np.testing.assert_array_equal(
        sample.target_state,
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    )
    assert sample.pre_handoff is True
    assert not sample.vision_history.flags.writeable
    assert not sample.robot_proprio_history.flags.writeable
    assert not sample.target_state.flags.writeable
