import json

import numpy as np


def test_task_stream_has_explicit_identity_and_no_fabricated_level(tmp_path):
    from test_lerobot_conversion import _FakeLeRobotDataset

    from latency_meta_mdp.data.lerobot_conversion import write_lerobot_frame_dataset

    saved = []

    def factory(**kwargs):
        result = _FakeLeRobotDataset(root=kwargs["root"], create_kwargs=kwargs)
        saved.append(result)
        return result

    def frames():
        for h in range(3):
            yield {
                "image": np.zeros((4, 4, 3), np.uint8),
                "wrist_image": np.zeros((4, 4, 3), np.uint8),
                "state": np.full(16, h, np.float32),
                "actions": np.full(7, h, np.float32),
                "task": "Sort incoming parcels.",
            }

    header = dict(
        task_id="conveyor_sort",
        variant="surface",
        instruction="Sort incoming parcels.",
        state_contract="joint_qpos_qvel_gripper_width_velocity",
        state_dim=16,
        source_format_id="conveyor_expert_corpus_v1",
        action_contract="panda_osc_pose_delta_conveyor_v2",
        drop_n_last_frames=0,
        action_target_contract="masked_h50_real_actions_v1",
    )
    path = write_lerobot_frame_dataset(
        episodes=[
            (
                {
                    "episode_id": "seed1",
                    "group_id": "seed1",
                    "frame_count": 3,
                    "valid_action_chunk_source_count": 3,
                },
                frames(),
            )
        ],
        output_dir=tmp_path / "dataset",
        repo_id="local/conveyor",
        header=header,
        image_shape=(4, 4, 3),
        dataset_factory=factory,
    )
    manifest = json.loads(path.read_text())
    assert "level" not in manifest and "level" not in manifest["episodes"][0]
    assert manifest["action_contract"] == header["action_contract"]
    assert manifest["frame_count"] == 3
    assert len(saved[0].saved_episodes) == 1
    assert len(saved[0].saved_episodes[0]) == 3
    assert "file_sizes" in manifest and "artifacts" not in manifest
