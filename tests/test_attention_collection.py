import numpy as np
from attention_fixtures import small_config

from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.behaviors import AttentionRuleScheduler
from latency_meta_mdp.meta.information.deployment import AttentionScheduler
from latency_meta_mdp.meta.information.memory_schema import ActionSpec, AttentionPacket


def test_attention_epsilon_probability_and_reset():
    model = AttentionStageQ(small_config())
    scheduler = AttentionScheduler(model, epsilon=0.2, seed=1)
    scheduler.reset(episode_id="a")
    packet = AttentionPacket(10, "G", decision_key=(10, 1))
    legal = packet.action_spec.legal("G", 50)
    result = scheduler.decide(packet, legal)
    greedy = int(np.flatnonzero(legal)[0])
    assert result.probability == 0.2 / legal.sum() + 0.8 * (result.action == greedy)
    assert scheduler.diagnostics[-1]["greedy_action"] == greedy
    scheduler.reset(episode_id="b")
    assert not scheduler.memory.events and not scheduler.diagnostics


def test_grid_ready_window_refresh_and_sleep_clipping():
    spec = ActionSpec()
    scheduler = AttentionRuleScheduler("grid", action_spec=spec, rho=0.5, refresh="every_4_ticks")
    scheduler.reset(episode_id="grid")

    def act(t, stage, remaining, version=0):
        p = AttentionPacket(
            t,
            stage,
            decision_key=(t, 1),
            buffer_mask=np.arange(50) < remaining,
            buffer_version=version,
            forecast_tick=t if stage == "P" else -1,
            forecast_tokens=np.zeros((2, 196, 384)) if stage == "P" else None,
            forecast_buffer_version=version if stage == "P" else -1,
        )
        return scheduler.decide(p, spec.legal(stage, remaining, forecast_fresh=stage == "P")).action

    # Ready after pending at tick 7: R=43 gives target 7+round(.5*23)=19.
    assert act(7, "G", 43) == spec.id_for("observe")
    assert act(7, "P", 43) == spec.id_for("sleep:4")
    assert act(11, "G", 39) == spec.id_for("observe")
    assert act(11, "P", 39) == spec.id_for("sleep:4")
    assert act(15, "P", 35) == spec.id_for("sleep:4")
    assert act(19, "P", 31) == spec.id_for("launch")
    assert act(45, "G", 20, 1) == spec.id_for("observe")
    assert act(45, "P", 20, 1) == spec.id_for("launch")


def test_e25_and_lazy_preserve_reserve_with_dense_sleeps():
    spec = ActionSpec()
    p = AttentionPacket(0, "G", decision_key=(0, 1), plan_age=23)
    for kind, target in [("periodic", 2), ("lazy", 6)]:
        s = AttentionRuleScheduler(kind, action_spec=spec)
        s.reset(episode_id=kind)
        assert s.decide(p, spec.legal("G", 50)).action == spec.id_for(f"sleep:{target}")
