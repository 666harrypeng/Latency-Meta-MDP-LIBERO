import numpy as np
import pytest
import torch
from attention_fixtures import small_config

from latency_meta_mdp.meta.information.actor_deployment import ActorScheduler
from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.collection import resolve_scheduler
from latency_meta_mdp.meta.information.cycle import build_cycle_manifest
from latency_meta_mdp.meta.information.memory_schema import AttentionPacket


def test_actor_sampling_records_full_probability_without_epsilon_mixture():
    config = small_config()
    model = AttentionStageQ(config)
    with torch.no_grad():
        model.heads[0].bias[0] = np.log(3.0)
    actor = ActorScheduler(model, deterministic=False, seed=42)
    actor.reset(episode_id="sampling")
    packet = AttentionPacket(10, "G", decision_key=(10, 1))
    mask = np.zeros(11, bool)
    mask[:2] = True
    decision = actor.decide(packet, mask)
    assert decision.probability == pytest.approx(0.75 if decision.action == 0 else 0.25)
    assert len(actor.diagnostics[0]["logits"]) == 11
    assert actor.diagnostics[0]["legal"] == mask.tolist()
    assert resolve_scheduler("actor", 42, actor) is actor
    with pytest.raises(ValueError):
        resolve_scheduler("fqi", 42, actor)


def test_actor_generation_separates_train_delay_rng_and_fixed_greedy_validation(tmp_path):
    config = small_config()
    audit = dict(
        local_inventory_pass=True,
        candidate_splits=config["splits"],
        excluded_scenes=config["excluded_scenes"],
        sources=[],
    )
    kwargs = dict(root=tmp_path, code_hash="code", policy=dict(algorithm="actor"))
    generation = dict(
        name="ppo-1",
        trial=1,
        scenes=config["splits"]["train"][:32],
        behaviors=["actor"] * 32,
        regimes=["zero", "nominal", "family"],
        epsilon=0.0,
    )
    train = build_cycle_manifest(
        config, audit, phase="online", baselines=False, generation=generation, **kwargs
    )
    assert len(train["episodes"]) == 96
    assert all(not ep["deterministic"] and ep["delay_seed"] != 0 for ep in train["episodes"])
    assert all(ep["delay_seed"] != ep["scheduler_seed"] for ep in train["episodes"])
    evaluation = build_cycle_manifest(config, audit, phase="monitor", baselines=False, **kwargs)
    assert len(evaluation["episodes"]) == 9
    assert all(
        ep["behavior"] == "actor" and ep["deterministic"] and ep["delay_seed"] == 0
        for ep in evaluation["episodes"]
    )
