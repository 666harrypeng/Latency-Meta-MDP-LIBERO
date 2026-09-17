import numpy as np
import pytest

pytest.importorskip("lerobot")


def test_native_stream_can_reopen_multiple_episodes_after_cache_release(tmp_path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from latency_meta_mdp.data.lerobot_conversion import write_lerobot_frame_dataset

    def streams():
        for ep in range(2):
            frames = [
                dict(
                    image=np.full((8, 8, 3), ep * 20 + h, np.uint8),
                    wrist_image=np.full((8, 8, 3), ep * 20 + h, np.uint8),
                    state=np.full(16, ep + h, np.float32),
                    actions=np.full(7, h / 10, np.float32),
                    task="Sort.",
                )
                for h in range(3)
            ]
            yield (
                {"episode_id": str(ep), "frame_count": 3, "valid_action_chunk_source_count": 3},
                frames,
            )

    output = tmp_path / "dataset"
    write_lerobot_frame_dataset(
        episodes=streams(),
        output_dir=output,
        repo_id="local/streamtest",
        image_shape=(8, 8, 3),
        header={"task_id": "conveyor_sort", "state_dim": 16, "instruction": "Sort."},
    )
    reopened = LeRobotDataset("local/streamtest", root=output)
    assert len(reopened) == 6
    assert int(reopened[3]["episode_index"]) == 1
    np.testing.assert_allclose(reopened[5]["state"].numpy(), np.full(16, 3))
