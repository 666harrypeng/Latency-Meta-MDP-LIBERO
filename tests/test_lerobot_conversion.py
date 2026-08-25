from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.lerobot_conversion import write_lerobot_policy_dataset
from latency_meta_mdp.policy_data import ACTION_DIM, ACTION_HORIZON, POLICY_STATE_DIM, PolicyEpisode


def _policy_episode(*, episode_id: str, level: int = 1, frame_count: int = 52) -> PolicyEpisode:
    return PolicyEpisode(
        episode_id=episode_id,
        task_id="dynamic_grasp_lift",
        instruction="Grasp the moving ball and lift it.",
        level=level,
        action_horizon=ACTION_HORIZON,
        source_formal_tick=np.arange(frame_count),
        source_time_us=np.arange(frame_count) * 20_000,
        agentview_rgb=np.zeros((frame_count, 6, 8, 3), dtype=np.uint8),
        wrist_rgb=np.ones((frame_count, 6, 8, 3), dtype=np.uint8),
        state=np.arange(frame_count * POLICY_STATE_DIM, dtype=np.float32).reshape(
            frame_count, POLICY_STATE_DIM
        ),
        actions=np.arange(frame_count * ACTION_DIM, dtype=np.float32).reshape(
            frame_count, ACTION_DIM
        ),
        valid_action_chunk_sources=np.arange(frame_count - ACTION_HORIZON + 1),
    )


class _FakeLeRobotDataset:
    def __init__(self, *, root: Path, create_kwargs: dict) -> None:
        self.root = root
        self.create_kwargs = create_kwargs
        self.current_frames: list[dict] = []
        self.saved_episodes: list[list[dict]] = []
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}\n", encoding="utf-8")

    def add_frame(self, frame: dict) -> None:
        self.current_frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes.append(self.current_frames)
        self.current_frames = []


class _FakeDatasetFactory:
    def __init__(self) -> None:
        self.instance: _FakeLeRobotDataset | None = None

    def __call__(self, **kwargs) -> _FakeLeRobotDataset:
        self.instance = _FakeLeRobotDataset(root=Path(kwargs["root"]), create_kwargs=kwargs)
        return self.instance


def test_writer_preserves_50hz_policy_fields_and_declares_complete_h50_sampling(
    tmp_path: Path,
) -> None:
    factory = _FakeDatasetFactory()
    output = tmp_path / "lerobot_l1"
    episodes = [
        _policy_episode(episode_id="l1-seed-000010-attempt-000"),
        _policy_episode(episode_id="l1-seed-000011-attempt-000"),
    ]

    manifest_path = write_lerobot_policy_dataset(
        episodes=episodes,
        output_dir=output,
        repo_id="local/metamdp_panda_ball_l1_pilot",
        dataset_factory=factory,
    )

    assert manifest_path == output / "metamdp_dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "metamdp_lerobot_v21"
    assert manifest["fps"] == 50
    assert manifest["temporal_contract_id"] == "h50_e25_d20_k6_v1"
    assert manifest["action_horizon"] == 50
    assert manifest["launch_trigger_horizon"] == 25
    assert manifest["drop_n_last_frames"] == 49
    assert manifest["episode_count"] == 2
    assert manifest["frame_count"] == 104
    assert manifest["valid_action_chunk_source_count"] == 6
    assert [row["valid_action_chunk_source_count"] for row in manifest["episodes"]] == [3, 3]

    assert factory.instance is not None
    assert factory.instance.create_kwargs["fps"] == 50
    assert factory.instance.create_kwargs["robot_type"] == "panda"
    assert factory.instance.create_kwargs["use_videos"] is False
    assert set(factory.instance.create_kwargs["features"]) == {
        "image",
        "wrist_image",
        "state",
        "actions",
    }
    assert len(factory.instance.saved_episodes) == 2
    first_frame = factory.instance.saved_episodes[0][0]
    assert set(first_frame) == {"image", "wrist_image", "state", "actions", "task"}
    np.testing.assert_array_equal(first_frame["image"], episodes[0].agentview_rgb[0])
    np.testing.assert_array_equal(first_frame["wrist_image"], episodes[0].wrist_rgb[0])
    np.testing.assert_array_equal(first_frame["state"], episodes[0].state[0])
    np.testing.assert_array_equal(first_frame["actions"], episodes[0].actions[0])
    assert first_frame["task"] == episodes[0].instruction


def test_writer_accepts_a_single_pass_episode_iterable(tmp_path: Path) -> None:
    factory = _FakeDatasetFactory()
    episodes = (_policy_episode(episode_id=f"l1-seed-{seed:06d}-attempt-000") for seed in (10, 11))

    manifest_path = write_lerobot_policy_dataset(
        episodes=episodes,
        output_dir=tmp_path / "streamed",
        repo_id="local/metamdp_panda_ball_l1_streamed",
        dataset_factory=factory,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["episode_count"] == 2
    assert factory.instance is not None
    assert len(factory.instance.saved_episodes) == 2


def test_writer_rejects_cross_level_dataset_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "mixed"
    with pytest.raises(ValueError, match="one task level"):
        write_lerobot_policy_dataset(
            episodes=[
                _policy_episode(episode_id="l1", level=1),
                _policy_episode(episode_id="l2", level=2),
            ],
            output_dir=output,
            repo_id="local/mixed",
            dataset_factory=_FakeDatasetFactory(),
        )
    assert not output.exists()
