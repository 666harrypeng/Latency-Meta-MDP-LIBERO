import pytest


def test_wait_and_launch_use_actual_holding_times_and_discounted_tick_rewards():
    from latency_meta_mdp.meta.transitions import DecisionStageAccumulator

    collector = DecisionStageAccumulator(gamma=0.9)
    collector.begin(formal_tick=10, state="a", action="wait")
    for tick, reward in zip(range(10, 14), (1, 2, 3, 4), strict=True):
        collector.add_reward(formal_tick=tick, reward=reward)
    transition = collector.finish(next_formal_tick=14, next_state="b")
    assert transition.duration_ticks == 4
    assert transition.discount == pytest.approx(0.9**4)
    assert transition.reward == pytest.approx(1 + 0.9 * 2 + 0.9**2 * 3 + 0.9**3 * 4)
    assert transition.undiscounted_reward == 10
    collector.begin(formal_tick=14, state="b", action="launch")
    collector.add_reward(formal_tick=14, reward=5)
    collector.add_reward(formal_tick=15, reward=6)
    transition = collector.finish(next_formal_tick=16, next_state="c")
    assert transition.duration_ticks == 2
    assert transition.reward == pytest.approx(5 + 0.9 * 6)
    assert transition.bootstrap_discount == pytest.approx(0.9**2)


def test_terminal_inside_pending_interval_truncates_duration_and_zeroes_bootstrap():
    from latency_meta_mdp.meta.transitions import DecisionStageAccumulator

    collector = DecisionStageAccumulator(gamma=0.99)
    collector.begin(
        formal_tick=0, state="a", action="launch", proposed_action="wait", shielded=True
    )
    collector.add_reward(formal_tick=0, reward=0)
    collector.add_reward(formal_tick=1, reward=1)
    transition = collector.finish(next_formal_tick=2, next_state="terminal", terminated=True)
    assert transition.duration_ticks == 2 and transition.bootstrap_discount == 0
    assert transition.reward == pytest.approx(0.99)
    assert transition.proposed_action == "wait" and transition.shielded


def test_accumulator_rejects_missing_physical_ticks_and_overlapping_decisions():
    from latency_meta_mdp.meta.transitions import DecisionStageAccumulator

    collector = DecisionStageAccumulator(gamma=1)
    collector.begin(formal_tick=3, state="a", action="wait")
    with pytest.raises(RuntimeError):
        collector.begin(formal_tick=4, state="b", action="wait")
    with pytest.raises(ValueError):
        collector.add_reward(formal_tick=4, reward=0)
    collector.add_reward(formal_tick=3, reward=1)
    with pytest.raises(ValueError):
        collector.finish(next_formal_tick=5, next_state="b")
