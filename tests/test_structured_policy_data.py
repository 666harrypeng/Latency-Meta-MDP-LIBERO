from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from latency_meta_mdp.expert_realization.source_corpus.loader import VerifiedSourceCorpus


def _source(tmp_path: Path) -> VerifiedSourceCorpus:
    png = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 12, dtype=np.uint8)).save(png, format="PNG")
    rows = []
    for tick in range(4):
        rows.append(
            {
                "episode_id": "test-L1",
                "formal_tick": tick,
                "time_us": tick * 20_000,
                "agentview_rgb": {"bytes": png.getvalue(), "path": "agent.png"},
                "wrist_rgb": {"bytes": png.getvalue(), "path": "wrist.png"},
                "robot_qpos": [tick * 0.1] * 7,
                "robot_qvel": [0.2] * 7,
                "gripper_qpos": [0.03, -0.02],
                "gripper_qvel": [0.01, -0.03],
                "expert_action": [tick * 0.1] * 6 + [1.0] if tick < 3 else None,
                "action_mask": [True] * 7 if tick < 3 else None,
                "object_pose": [900.0] * 7,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "episode.parquet")
    return VerifiedSourceCorpus(
        root=tmp_path,
        manifest=SimpleNamespace(),
        episode_rows=(
            {
                "episode_id": "test-L1",
                "level": 1,
                "logical_master_task_index": 42,
                "data_shard": "episode.parquet",
                "row_group_index": 0,
                "row_count": 4,
                "terminal_tick": 3,
            },
        ),
    )


def test_structured_policy_reader_preserves_final_sources_and_proprio(tmp_path):
    from latency_meta_mdp.policy_data import load_structured_policy_episode

    episode = load_structured_policy_episode(_source(tmp_path), episode_id="test-L1")
    assert episode.state.shape == (3, 16)
    assert episode.state_contract == "joint_qpos_qvel_gripper_width_velocity"
    assert episode.logical_master_task_index == 42
    assert episode.valid_action_chunk_sources.tolist() == [0, 1, 2]
    np.testing.assert_allclose(episode.state[2], [0.2] * 14 + [0.05, 0.04])
    assert episode.agentview_rgb.shape == (3, 8, 8, 3)
    np.testing.assert_array_equal(episode.agentview_rgb, 12)
    assert not hasattr(episode, "object_pose")


def test_final_action_target_uses_explicit_loss_mask(tmp_path):
    from latency_meta_mdp.policy_data import (
        load_structured_policy_episode,
        materialize_policy_action_target,
    )

    episode = load_structured_policy_episode(_source(tmp_path), episode_id="test-L1")
    target, mask = materialize_policy_action_target(episode, target_start_tick=2)
    assert target.shape == (50, 7)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [True] + [False] * 49)
    np.testing.assert_allclose(target[0], [0.2] * 6 + [1.0])
    assert np.isfinite(target).all()
