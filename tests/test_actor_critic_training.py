import numpy as np
import pytest
import torch
from attention_fixtures import make_batch, rich_replay, small_config

from latency_meta_mdp.meta.information.actor_critic import AttentionValue, initialize_from_encoder
from latency_meta_mdp.meta.information.actor_learning import (
    balanced_probabilities,
    fit,
    initial_state,
    load_state,
)
from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.fqi import model_hash


def test_scalar_value_encoder_copy_keeps_actor_independent():
    batch, config = make_batch(2)
    actor = AttentionStageQ(config)
    before = model_hash(actor)
    value = initialize_from_encoder(actor, value=True)
    assert isinstance(value, AttentionValue)
    assert value(batch).shape == (2,)
    assert not {p.data_ptr() for p in actor.parameters()} & {
        p.data_ptr() for p in value.parameters()
    }
    with torch.no_grad():
        next(value.parameters()).add_(1)
    assert model_hash(actor) == before
    restored = AttentionStageQ(config)
    restored.load_state_dict(actor.state_dict(), strict=True)
    torch.testing.assert_close(restored(batch), actor(batch), rtol=0, atol=0)


def test_balancing_equal_delay_then_mode_not_longest_episode():
    episodes = [
        dict(regime="zero", behavior="lazy"),
        dict(regime="zero", behavior="active"),
        dict(regime="family", behavior="active"),
    ]
    weights = balanced_probabilities([1, 3, 2], episodes)
    np.testing.assert_allclose(weights, [0.25, 1 / 12, 1 / 12, 1 / 12, 0.25, 0.25])


def test_iql_resume_preserves_optimizers_rng_and_target(tmp_path):
    config = small_config()
    replay = rich_replay(config)
    state = initial_state(config, seed=7)
    opts = dict(
        global_batch=4,
        microbatch=2,
        updates=2,
        checkpoint_every=1,
        learning_rate=1.0e-4,
        seed=11,
        tau=0.7,
        ema=0.005,
        cpu_threads=1,
    )
    fit("iql", state, replay, tmp_path / "full", options=opts, devices=["cpu"])
    fit("iql", state, replay, tmp_path / "resume", options=dict(opts, updates=1), devices=["cpu"])
    saved = load_state(tmp_path / "resume" / "latest")
    assert saved["step"] == 1 and saved["optimizers"]
    fit("iql", saved, replay, tmp_path / "resume", options=opts, devices=["cpu"])
    full, resumed = [load_state(tmp_path / x / "latest") for x in ("full", "resume")]
    for name in full["models"]:
        for key in full["models"][name]:
            torch.testing.assert_close(
                full["models"][name][key], resumed["models"][name][key], rtol=0, atol=0
            )
    assert full["step"] == resumed["step"] == 2


def test_bad_world_batch_rejected_before_training(tmp_path):
    config = small_config()
    with pytest.raises(ValueError, match="divisible"):
        fit(
            "iql",
            initial_state(config),
            rich_replay(config),
            tmp_path,
            options=dict(global_batch=3, microbatch=1, updates=1),
            devices=["cpu", "cpu"],
        )


def test_two_ranks_match_single_rank_with_uneven_terminal_minibatch(tmp_path):
    config = small_config()
    replay = rich_replay(config)
    actor = AttentionStageQ(config)
    value = initialize_from_encoder(actor, value=True)
    state = dict(
        config=config,
        phase="critic",
        step=0,
        epoch=0,
        optimizers={},
        models=dict(actor=actor.state_dict(), v=value.state_dict()),
    )
    auxiliary = dict(returns=np.array([0.3, 0.2, 0.1], np.float32))
    options = dict(global_batch=4, microbatch=1, epochs=1, cpu_threads=1, learning_rate=1e-4)
    single = fit(
        "critic",
        state,
        replay,
        tmp_path / "single",
        options=options,
        devices=["cpu"],
        auxiliary=auxiliary,
    )
    multi = fit(
        "critic",
        state,
        replay,
        tmp_path / "multi",
        options=options,
        devices=["cpu", "cpu"],
        auxiliary=auxiliary,
    )
    for key in single["models"]["v"]:
        torch.testing.assert_close(
            single["models"]["v"][key], multi["models"]["v"][key], atol=2e-7, rtol=2e-5
        )
    for result in (single, multi):
        for key in state["models"]["actor"]:
            torch.testing.assert_close(
                state["models"]["actor"][key], result["models"]["actor"][key], rtol=0, atol=0
            )


def test_new_training_phase_resets_fit_binding_and_does_not_inherit_adam(tmp_path):
    config = small_config()
    actor = AttentionStageQ(config)
    state = dict(
        config=config,
        phase="actor",
        step=700,
        epoch=3,
        fit_binding="offline-teacher",
        optimizers={"actor": {"invalid": "must discard"}},
        models=dict(
            actor=actor.state_dict(), v=initialize_from_encoder(actor, value=True).state_dict()
        ),
    )
    fitted = fit(
        "critic",
        state,
        rich_replay(config),
        tmp_path,
        options=dict(global_batch=4, microbatch=2, epochs=1),
        devices=["cpu"],
        auxiliary=dict(returns=np.array([0.1, 0.1, 0.1], np.float32)),
    )
    assert fitted["step"] == 1 and set(fitted["optimizers"]) == {"v"}
