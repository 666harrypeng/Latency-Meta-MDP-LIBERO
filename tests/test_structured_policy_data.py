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


def test_structured_export_uses_only_train_episode_ids(tmp_path, monkeypatch):
    import json

    from test_lerobot_conversion import _FakeDatasetFactory

    from latency_meta_mdp import policy_dataset_run as run
    from latency_meta_mdp.expert_realization.source_corpus import loader, split_view

    source = _source(tmp_path)
    source.manifest.schema_version = 3
    source.manifest.corpus_id = "test-structured"
    (tmp_path / "manifest.json").write_text("{}")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    split = SimpleNamespace(
        train_episode_ids=("test-L1",),
        validation_episode_ids=("forbidden-L1",),
        train_master_task_indices=(42,),
        split_id="grouped",
    )
    monkeypatch.setattr(loader, "load_verified_source_corpus", lambda root: source)
    monkeypatch.setattr(split_view, "load_verified_source_split", lambda path, corpus: split)
    manifest_path = run.convert_structured_source_to_lerobot(
        source_root=tmp_path,
        split_manifest=split_path,
        profile_path=Path("configs/policy/pi05_structured_state16_h50_v1.yaml"),
        output_dir=tmp_path / "exported",
        levels=(1,),
        dataset_factory=_FakeDatasetFactory(),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["split"] == "train"
    assert manifest["episode_count"] == 1
    assert manifest["datasets"][0]["valid_action_chunk_source_count"] == 3
    nested = json.loads(
        (manifest_path.parent / manifest["datasets"][0]["dataset_manifest"]).read_text()
    )
    assert [e["episode_id"] for e in nested["episodes"]] == ["test-L1"]
