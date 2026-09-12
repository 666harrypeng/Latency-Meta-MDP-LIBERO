from types import SimpleNamespace

import numpy as np

from latency_meta_mdp.meta_transitions import DecisionStageAccumulator


def state(tick, remaining=30):
    return {
        "observation": SimpleNamespace(formal_tick=tick, state=np.arange(16, dtype=np.float32)),
        "buffer": SimpleNamespace(
            buffer_version=0,
            plan_age=50 - remaining,
            remaining_actions=remaining,
            estimated_delay_ticks=4,
            delay_history_ticks=(4, 3),
            unread_action_buffer=np.zeros((50, 7)),
            unread_action_mask=np.arange(50) < remaining,
        ),
        "belief": SimpleNamespace(
            available=True,
            current_visual_latents=np.ones((2, 196, 384), np.float16),
            future_visual_latents=np.full((2, 196, 384), 2, np.float16),
            future_proprio=np.arange(16, dtype=np.float32) + 1,
        ),
    }


def test_replay_stores_public_features_once_and_preserves_smdp_targets(tmp_path):
    from latency_meta_mdp.meta_replay import MetaEpisodeReplay

    writer = MetaEpisodeReplay()
    c = DecisionStageAccumulator(gamma=0.9)
    a, b = state(10), state(14)
    c.begin(formal_tick=10, state=a, action="wait")
    for h in range(10, 14):
        c.add_reward(formal_tick=h, reward=0)
    writer(c.finish(next_formal_tick=14, next_state=b))
    c.begin(formal_tick=14, state=b, action="launch")
    c.add_reward(formal_tick=14, reward=0)
    c.add_reward(formal_tick=15, reward=1)
    writer(c.finish(next_formal_tick=16, next_state=None, terminated=True))
    path = tmp_path / "episode.npz"
    writer.save(path, metadata={"case": "test"})
    with np.load(path, allow_pickle=False) as data:
        assert data["visual"].shape == (2, 2, 2, 196, 384)
        assert data["vector"].shape == (2, 501)
        np.testing.assert_array_equal(data["state_index"], [0, 1])
        np.testing.assert_array_equal(data["next_state_index"], [1, -1])
        np.testing.assert_array_equal(data["action"], [0, 1])
        np.testing.assert_allclose(data["reward"], [0, 0.9])
        np.testing.assert_allclose(data["bootstrap_discount"], [0.9**4, 0])
        assert not any("delay_realized" in key or "hidden" in key for key in data.files)


def test_behavior_is_reproducible_and_covers_both_actions():
    from latency_meta_mdp.meta_replay import ExploratoryCursorScheduler

    a = ExploratoryCursorScheduler(seed=7)
    b = ExploratoryCursorScheduler(seed=7)
    public = state(10)["buffer"]
    x = [a(public, None, None) for _ in range(100)]
    assert x == [b(public, None, None) for _ in range(100)]
    assert set(x) == {True, False}


def test_stratified_scheduler_visits_late_opportunities_and_resets_on_buffer():
    from latency_meta_mdp.meta_replay import StratifiedLaunchScheduler

    s = StratifiedLaunchScheduler(seed=7, master_ordinal=0, replica_index=2)
    b = state(10, remaining=40)["buffer"]
    assert not s(b, None, None)
    assert s.target_age in (26, 30)
    first = s.target_age
    b.plan_age = first
    assert s(b, None, None)
    assert s.target_age == first
    b.buffer_version = 1
    b.plan_age = 29
    b.remaining_actions = 21
    s(b, None, None)
    assert s.target_age in (29, 30)
    b.plan_age, b.remaining_actions = 30, 20
    assert s(b, None, None)


def test_q_exploration_preserves_legal_actions_and_q_logging():
    from latency_meta_mdp.meta_replay import ExploratoryQScheduler

    class Greedy:
        last_q_values = [0.8, 0.2]

        def __call__(self, state, observation, belief):
            return state.remaining_actions <= 20

    s = ExploratoryQScheduler(Greedy(), seed=7, epsilon=0)
    assert not s(state(10)["buffer"], None, None)
    assert s.last_q_values == [0.8, 0.2]
    s = ExploratoryQScheduler(Greedy(), seed=7, epsilon=1)
    assert {s(state(10)["buffer"], None, None) for _ in range(100)} == {False, True}
    assert all(s(state(10, remaining=20)["buffer"], None, None) for _ in range(50))
