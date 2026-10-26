import copy
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.meta.information.attention_replay import DecisionSampler


def replay_groups(startup=1000, old=800, latest=200):
    return SimpleNamespace(
        group_indices=dict(
            startup=np.arange(startup),
            old=np.arange(startup, startup + old),
            latest=np.arange(startup + old, startup + old + latest),
        )
    )


def test_curriculum_uses_absolute_cycle_and_reaches_exact_endpoints():
    from latency_meta_mdp.meta.information.attention_campaign import curriculum_sampling

    spec = dict(
        total_cycles=100,
        start=0.1,
        end=0.5,
        curvature=1.5,
        latest_fraction_cap=0.25,
        latest_online_fraction=0.5,
        oversampling_cap=4,
    )
    ratios = [curriculum_sampling(spec, i)["online_fraction"] for i in range(1, 101)]
    assert ratios[0] == pytest.approx(0.1)
    assert ratios[-1] == pytest.approx(0.5)
    assert all(b > a for a, b in zip(ratios, ratios[1:]))
    assert np.diff(ratios)[-1] > np.diff(ratios)[0]
    assert ratios[3] < 0.11
    with pytest.raises(ValueError):
        curriculum_sampling(spec, 101)


def test_latest_preference_caps_and_actual_draw_accounting():
    spec = dict(
        online_fraction=0.5,
        latest_fraction_cap=0.25,
        latest_online_fraction=0.5,
        oversampling_cap=4,
    )
    sampler = DecisionSampler(replay_groups(), seed=8, sampling=spec)
    weights = dict(zip(sampler.names, sampler.probabilities))
    assert weights == pytest.approx(dict(startup=0.5, old=0.25, latest=0.25))
    draws = sampler.sample(20_000)
    report = sampler.describe(draws)
    assert sum(g["draws"] for g in report["groups"].values()) == 20_000
    assert report["groups"]["latest"]["unique"] <= 200
    assert report["effective_online_fraction"] == pytest.approx(0.5)
    scarce = DecisionSampler(replay_groups(latest=1), seed=8, sampling=spec)
    weights = dict(zip(scarce.names, scarce.probabilities))
    assert weights["latest"] <= 4 * weights["startup"] / 1000 + 1e-12
    assert weights["old"] / 800 <= 4 * weights["startup"] / 1000 + 1e-12
    assert weights["startup"] == pytest.approx(0.5)
    assert weights["old"] > 0.49


def test_sampling_transition_keeps_rng_but_strict_resume_rejects_change():
    original = dict(online_fraction=0.1, latest_fraction_cap=0.25, oversampling_cap=4)
    previous = DecisionSampler(replay_groups(), seed=2, sampling=original)
    previous.sample(19)
    state = copy.deepcopy(previous.state_dict())
    new = DecisionSampler(replay_groups(), seed=999, sampling=dict(original, online_fraction=0.2))
    with pytest.raises(ValueError, match="sampling"):
        new.load_state_dict(state)
    new.load_state_dict(state, allow_sampling_change=True)
    assert new.state_dict()["rng"] == state["rng"]
    assert new.visits == 19
    assert new.sampling["online_fraction"] == 0.2
    assert state["sampling"] == original


def test_campaign_offset_reuses_previous_online_and_bounded_resume(tmp_path):
    from pathlib import Path

    from latency_meta_mdp.meta.information.attention_campaign import attention_cycle_loop
    from latency_meta_mdp.meta.information.campaign import CampaignJournal

    recipe = dict(
        startup_sources=[dict(path="starter", group="startup")],
        initial_online_sources=[dict(path="cycle1", group="latest")],
        cycle_offset=1,
        reward_regression_policy="report_only",
        cycles=[dict(generation={"name": str(i)}) for i in range(3)],
    )
    calls = []

    def collect(index, definition, parent):
        calls.append((index, str(parent)))
        return dict(artifacts={}, source=dict(path=f"cycle{index}", group="latest"))

    def fit(index, definition, parent, sources):
        assert [s["group"] for s in sources] == ["startup"] + ["old"] * (index - 1) + ["latest"]
        return dict(
            artifacts={},
            latest=f"checkpoint{index}",
            candidates=[],
            optimizer_updates=2000,
            validation_episodes=9,
        )

    kwargs = dict(
        journal=CampaignJournal(tmp_path, binding={}),
        recipe=recipe,
        initial=Path("checkpoint1"),
        initial_metrics=None,
        collect=collect,
        fit=fit,
        stop_after_cycles=2,
    )
    result = attention_cycle_loop(**kwargs)
    assert calls == [(2, "checkpoint1"), (3, "checkpoint2")]
    assert result["status"] == "bounded_complete"
    assert attention_cycle_loop(**kwargs) == result
    assert len(calls) == 2


def test_campaign_resource_forwarding_keeps_scientific_recipe(tmp_path, monkeypatch):
    import json

    from latency_meta_mdp.meta.information import attention_campaign, campaign

    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(dict(schema_version=2, global_batch=512)))
    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(attention_campaign, "run_attention_campaign", run)
    assert (
        campaign.run_campaign(
            "config",
            tmp_path,
            root=tmp_path,
            recipe_path=path,
            audit_path="audit",
            gpus="2,3",
            training_gpus="2",
            rollout_workers_per_gpu=3,
            validation_workers_per_gpu=2,
            validation_gpus="3",
            cpu_threads=5,
            learner_cpu_threads=8,
            validation_cpu_threads=4,
            stop_after_cycles=3,
            compact_replay=True,
        )
        == "ok"
    )
    assert captured["rollout_workers_per_gpu"] == 3
    assert captured["validation_workers_per_gpu"] == 2
    assert captured["learner_cpu_threads"] == 8
    assert captured["validation_gpus"] == "3"
    assert json.loads(path.read_text())["global_batch"] == 512
