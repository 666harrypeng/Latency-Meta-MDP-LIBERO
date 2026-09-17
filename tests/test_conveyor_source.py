from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.runtime.policy_execution import PolicyObservation


def observation(tick):
    image = np.full((256, 256, 3), tick, dtype=np.uint8)
    return PolicyObservation(tick, image, image, np.full(16, tick), prompt="Deliver the balls.")


def write_episode(path: Path):
    from latency_meta_mdp.data.conveyor.source import ConveyorRecorder

    with ConveyorRecorder(
        path, seed=7, action_contract_id="panda_osc_pose_delta_conveyor_v2"
    ) as recorder:
        recorder.start(observation(0))
        for h in range(32):
            recorder.append(
                np.full(7, h / 32),
                observation(h + 1),
                success_delta=int(h in (11, 31)),
                done=h == 31,
            )
        recorder.finish({"spawned": 2, "successes": 2, "misses": 0, "timeouts": 0, "end_tick": 32})


def test_continuous_source_preserves_internal_delivery_and_true_terminal(tmp_path):
    from latency_meta_mdp.data.conveyor.source import ConveyorSource

    write_episode(tmp_path / "episode")
    episode = ConveyorSource(tmp_path / "episode")
    assert episode.states.shape == (33, 16)
    assert episode.actions.shape == (32, 7)
    assert np.flatnonzero(episode.success_delta).tolist() == [11, 31]
    assert np.flatnonzero(episode.done).tolist() == [31]
    sample = episode.policy_sample(8)
    assert set(sample) == {"image", "wrist_image", "state", "prompt", "actions", "actions_is_pad"}
    np.testing.assert_array_equal(sample["actions"][:24], episode.actions[8:])
    assert (~sample["actions_is_pad"]).sum() == 24
    assert (sample["actions"][24:] == 0).all()
    assert (sample["image"] == 8).all()
    assert episode.manifest["group_id"] == "conveyor_sort/surface/seed-7"
    assert episode.manifest["purpose"] == "development_smoke"
    assert episode.manifest["action_contract"] == "panda_osc_pose_delta_conveyor_v2"
    with pytest.raises(IndexError):
        episode.policy_sample(32)


def test_direct_window_crosses_delivery_without_future_leak_or_false_padding(tmp_path):
    from latency_meta_mdp.data.conveyor.source import ConveyorSource

    write_episode(tmp_path / "episode")
    episode = ConveyorSource(tmp_path / "episode")
    window = episode.belief_window(10, 5)
    inputs, target = window["inputs"], window["target"]
    np.testing.assert_array_equal(inputs["history_ticks"], [2, 6, 10])
    np.testing.assert_array_equal(inputs["executed_controls"].reshape(8, 7), episode.actions[2:10])
    np.testing.assert_array_equal(inputs["executable_controls"][:5], episode.actions[10:15])
    assert (inputs["executable_controls"][5:] == 0).all()
    assert inputs["control_mask"].sum() == 5
    assert (target["rgb"] == 15).all()
    assert (inputs["vision_history_rgb"][-1] == 10).all()
    assert (episode.belief_window(12, 20)["target"]["rgb"] == 32).all()
    with pytest.raises(ValueError):
        episode.belief_window(9, 1)
    with pytest.raises(ValueError):
        episode.belief_window(13, 20)


def test_partial_or_misaligned_recording_is_not_a_completed_source(tmp_path):
    from latency_meta_mdp.data.conveyor.source import ConveyorRecorder, ConveyorSource

    path = tmp_path / "partial"
    with ConveyorRecorder(
        path, seed=7, action_contract_id="panda_osc_pose_delta_conveyor_v2"
    ) as recorder:
        recorder.start(observation(0))
        with pytest.raises(ValueError):
            recorder.append(np.zeros(7), observation(2), success_delta=0, done=False)
    assert not (path / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        ConveyorSource(path)


def test_failed_expert_episode_cannot_be_finalized_for_training(tmp_path):
    from latency_meta_mdp.data.conveyor.source import ConveyorRecorder

    path = tmp_path / "failed"
    with ConveyorRecorder(
        path, seed=7, action_contract_id="panda_osc_pose_delta_conveyor_v2"
    ) as recorder:
        recorder.start(observation(0))
        recorder.append(np.zeros(7), observation(1), success_delta=0, done=True)
        with pytest.raises(ValueError, match="fully successful"):
            recorder.finish(
                {"spawned": 1, "successes": 0, "misses": 1, "timeouts": 0, "end_tick": 1}
            )
    assert not (path / "manifest.json").exists()
