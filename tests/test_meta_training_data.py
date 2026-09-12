import numpy as np
import pytest
from test_meta_replay import state

from latency_meta_mdp.meta_replay import MetaEpisodeReplay
from latency_meta_mdp.meta_transitions import DecisionStageAccumulator


def write_episode(path, master, partition):
    from latency_meta_mdp.cli.train_meta_q import BINDING_KEYS

    writer = MetaEpisodeReplay()
    c = DecisionStageAccumulator(gamma=0.999)
    c.begin(formal_tick=10, state=state(10), action="wait")
    for h in range(10, 14):
        c.add_reward(formal_tick=h, reward=0)
    writer(c.finish(next_formal_tick=14, next_state=state(14)))
    c.begin(formal_tick=14, state=state(14), action="launch")
    c.add_reward(formal_tick=14, reward=1)
    writer(c.finish(next_formal_tick=15, next_state=None, terminated=True))
    writer.save(
        path,
        metadata={
            "identity": {**dict.fromkeys(BINDING_KEYS, "same"), "discount_per_tick": 0.999},
            "case": {"master_index": master, "meta_partition": partition},
        },
    )


def test_loader_offsets_episode_indices_without_crossing_boundaries(tmp_path):
    from latency_meta_mdp.cli.train_meta_q import load_replay

    write_episode(tmp_path / "a.npz", 1, "train")
    write_episode(tmp_path / "b.npz", 2, "validation")
    x, vector, legal, data, binding, inventory, masters = load_replay(tmp_path)
    assert len(x) == len(vector) == len(legal) == 4
    np.testing.assert_array_equal(data["state"], [0, 1, 2, 3])
    np.testing.assert_array_equal(data["next"], [1, 0, 3, 0])
    np.testing.assert_array_equal(data["train"], [True, True, False, False])
    assert masters == {"train": [1], "validation": [2]}


def test_loader_rejects_task_instance_leakage_between_fit_and_validation(tmp_path):
    from latency_meta_mdp.cli.train_meta_q import load_replay

    write_episode(tmp_path / "a.npz", 1, "train")
    write_episode(tmp_path / "b.npz", 1, "validation")
    with pytest.raises(ValueError, match="disjoint"):
        load_replay(tmp_path)
