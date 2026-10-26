import numpy as np
import pytest
import torch


def test_gentle_sampler_keeps_new_data_small_and_caps_per_decision_mass():
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay, DecisionSampler

    replay = DecisionReplay(
        [[{}] * 1000, [{}] * 10, [{}] * 10], groups=["startup", "old", "latest"]
    )
    sampler = DecisionSampler(
        replay,
        seed=7,
        sampling={
            "online_fraction": 0.2,
            "latest_fraction_cap": 0.1,
            "oversampling_cap": 4.0,
        },
    )
    ids = sampler.sample(200000)
    # The source-size cap gives 80/(1000+80), rather than the requested 20%.
    assert (ids >= 1000).mean() == pytest.approx(80 / 1080, abs=0.002)
    assert (ids >= 1010).mean() == pytest.approx(40 / 1080, abs=0.002)
    state = sampler.state_dict()
    restored = DecisionSampler(replay, seed=0, sampling=sampler.sampling)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.sample(128), sampler.sample(128))
    with pytest.raises(ValueError, match="sampling"):
        DecisionSampler(replay, seed=0).load_state_dict(state)


def test_gentle_sampler_moves_excess_latest_mass_to_old_data():
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay, DecisionSampler

    replay = DecisionReplay([[{}] * 100, [{}] * 10, [{}] * 90], groups=["startup", "old", "latest"])
    sampler = DecisionSampler(
        replay,
        seed=7,
        sampling={
            "online_fraction": 0.3,
            "latest_fraction_cap": 0.1,
            "oversampling_cap": 4.0,
        },
    )
    np.testing.assert_allclose(sampler.probabilities, [0.7, 0.2, 0.1])


def test_scaled_loss_preserves_td_and_conservative_balance():
    from latency_meta_mdp.meta.information.attention_fqi import regression_terms

    q = torch.tensor([[0.2, 0.4, 1e6], [0.0, -0.2, 1e6]])
    actions = torch.tensor([0, 1])
    targets = torch.tensor([0.6, -0.1])
    legal = torch.tensor([[True, True, False]] * 2)
    td, conservative = regression_terms(q, actions, targets, legal)
    scaled = regression_terms(
        q * 100, actions, targets * 100, legal, huber_beta=100.0, conservative_temperature=100.0
    )
    torch.testing.assert_close(scaled[0], td * 100)
    torch.testing.assert_close(scaled[1], conservative * 100)
    torch.testing.assert_close(td, torch.tensor([0.08, 0.005]))
    shifted = regression_terms(q + 10, actions, targets, legal)
    torch.testing.assert_close(shifted[1], conservative)


def test_latest_cap_does_not_overweight_a_small_old_cohort():
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay, DecisionSampler

    sizes = np.array([1000, 10, 990])
    replay = DecisionReplay([[{}] * n for n in sizes], groups=["startup", "old", "latest"])
    sampler = DecisionSampler(
        replay,
        seed=7,
        sampling=dict(online_fraction=0.2, latest_fraction_cap=0.1, oversampling_cap=4),
    )
    mass = sampler.probabilities / sizes
    assert np.all(mass[1:] <= 4 * mass[0] + 1e-12)
    assert sampler.probabilities[2] == pytest.approx(0.1)


def test_truncated_smoke_preserves_evidence_without_complete_budget_claim():
    from latency_meta_mdp.meta.information.budget_reward import reward_view
    from latency_meta_mdp.meta.information.collection import attach_budget_report

    rows, operations, events, spec = _reward_fixture()
    events = events[:-1]  # The last parcel is still active at the diagnostic boundary.
    result = dict(truncated=True, parcel_events=events)
    attach_budget_report(result, rows, operations, spec)
    assert "budget" not in result
    assert result["budget_accounting"]["status"] == "incomplete_truncated_diagnostic"
    assert result["parcel_events"] == events
    with pytest.raises(ValueError, match="incomplete parcel ledger"):
        reward_view(rows, operations, events, spec)
    result["truncated"] = False
    with pytest.raises(ValueError, match="incomplete parcel ledger"):
        attach_budget_report(result, rows, operations, spec)


def _reward_fixture():
    from types import SimpleNamespace

    rows = [
        dict(packet=SimpleNamespace(tick=t), end_tick=end, forced=f, task_reward=r)
        for t, end, f, r in [
            (0, 2, True, 0.1),
            (2, 2, False, 0),
            (2, 4, True, 0.1),
            (4, 6, False, 0),
        ]
    ]
    operations = [
        dict(operation_id=str(i), component=c, owner=o, seconds=9999.0)
        for i, (c, o) in enumerate(
            [("vla", -1), ("meta_q", 1), ("jepa", 2), ("vla", 3), ("video", 3)]
        )
    ]
    events = [dict(parcel_id=i, tick=0, event="spawn") for i in range(4)]
    events += [
        dict(parcel_id=0, tick=1, event="success"),
        dict(parcel_id=1, tick=3, event="success"),
        dict(parcel_id=2, tick=5, event="miss"),
        dict(parcel_id=3, tick=5, event="timeout"),
    ]
    spec = dict(
        version=1,
        delivery_reward=10.0,
        drop_penalty=5.0,
        cost_weight=0.5,
        prices=dict(vla=1.0, dino=0.1, jepa=0.2, decoder=0.1, meta_q=0.02, bootstrap=1.0),
    )
    return rows, operations, events, spec


def test_event_reward_keeps_prefix_separate_and_folds_forced_work():
    from latency_meta_mdp.meta.information.budget_reward import reward_view

    rows, operations, events, spec = _reward_fixture()
    result = reward_view(rows, operations, events, spec)
    # First delivery and bootstrap precede any free decision.
    assert result["prefix_reward"] == pytest.approx(9.5)
    np.testing.assert_allclose(result["decision_rewards"], [9.89, -5.5])
    assert result["episode_reward"] == pytest.approx(13.89)
    assert result["deliveries"] == 2 and result["drops"] == 1 and result["timeouts"] == 1
    assert result["compute_cost"] == pytest.approx(2.22)
    # Wall time and logical delay do not set the fixed invocation price.
    for op in operations:
        op["seconds"] = 0.0
    assert reward_view(rows, operations, events, spec) == result


def test_optional_reward_prices_deployment_decisions_independent_of_collector():
    from types import SimpleNamespace

    from latency_meta_mdp.meta.information.budget_reward import reward_view

    rows, operations, events, spec = _reward_fixture()
    for row in rows:
        row["forced"] = False
        row["packet"].action_spec = SimpleNamespace(version=3)
    operations = [op for op in operations if op["component"] != "meta_q"]
    rule = operations + [
        dict(operation_id=f"decision-{i}", component="rule_scheduler", owner=i)
        for i in range(len(rows))
    ]
    learned = [
        {**op, "component": "meta_q" if op["component"] == "rule_scheduler" else op["component"]}
        for op in rule
    ]
    offline = reward_view(rows, rule, events, spec)
    online = reward_view(rows, learned, events, spec)
    assert offline["decision_rewards"] == pytest.approx(online["decision_rewards"])
    assert offline["compute_cost"] == pytest.approx(online["compute_cost"])
    assert offline["imputed_meta_calls"] == len(rows)
    assert "meta_q" not in offline["module_calls"]  # Do not invent observed calls.
    assert online["module_calls"]["meta_q"] == len(rows)
    assert online["imputed_meta_calls"] == 0
    with pytest.raises(ValueError, match="decision service"):
        reward_view(rows, rule[:-1], events, spec)


def test_event_reward_rejects_duplicate_outcomes_and_unknown_operations():
    from latency_meta_mdp.meta.information.budget_reward import reward_view

    rows, ops, events, spec = _reward_fixture()
    with pytest.raises(ValueError, match="outcome"):
        reward_view(rows, ops, events + [events[-1]], spec)
    with pytest.raises(ValueError, match="component"):
        reward_view(
            rows, ops + [dict(operation_id="new", component="new_model", owner=1)], events, spec
        )
    with pytest.raises(ValueError, match="owner"):
        reward_view(rows, [dict(operation_id="bad", component="vla", owner=10)], events, spec)


def test_event_reward_detects_wrong_tick_alignment_and_missing_ledger():
    from latency_meta_mdp.meta.information.budget_reward import reward_view

    rows, ops, events, spec = _reward_fixture()
    rows[2]["task_reward"] = 0.0
    with pytest.raises(ValueError, match="delivery"):
        reward_view(rows, ops, events, spec)
    with pytest.raises(ValueError, match="ledger"):
        reward_view(rows, ops, [], spec)


def test_budget_config_and_source_compatibility_are_explicit(tmp_path):
    import copy

    import yaml

    from latency_meta_mdp.meta.information.attention_replay import validate_sources
    from latency_meta_mdp.meta.information.config import identity_hash, load_config, sha256

    old = load_config("configs/experiments/conveyor_sort/meta_attention_v2.yaml")
    old_path = tmp_path / "source.yaml"
    old_path.write_text(yaml.safe_dump(old))
    new = copy.deepcopy(old)
    new["output_dir"] = "outputs/training/new-budget"
    new["training"].update(
        budget_lambda=0.5,
        reward=dict(
            version=1,
            delivery_reward=10,
            drop_penalty=5,
            cost_weight=0.5,
            prices=dict(vla=1, dino=0.1, jepa=0.1, decoder=0.1, meta_q=0.01, bootstrap=0),
        ),
        huber_beta=100,
        conservative_temperature=100,
        sampling=dict(online_fraction=0.1, latest_fraction_cap=0.1, oversampling_cap=4),
        replay_source_configs=[dict(path=str(old_path), sha256=sha256(old_path))],
    )
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(new))
    assert load_config(path) == new
    manifest = dict(
        binding=dict(
            schema_version=2,
            config_hash=identity_hash(old),
            actions=old["actions"],
            assets=old["assets"],
            mode="bound",
            scope="starter_training",
        ),
        episodes=[dict(episode_id="one", scene=10000, split="train")],
    )
    validate_sources(new, [manifest], root=tmp_path)
    # Merely naming a hash is insufficient: immutable input contracts must agree.
    changed = copy.deepcopy(new)
    changed["memory"]["visual_frames"] += 1
    with pytest.raises(ValueError):
        validate_sources(changed, [manifest], root=tmp_path)
    old_path.write_text(old_path.read_text() + "\n# changed file\n")
    with pytest.raises(ValueError):
        validate_sources(new, [manifest], root=tmp_path)
    new["training"]["budget_lambda"] = 0
    path.write_text(yaml.safe_dump(new))
    with pytest.raises(ValueError):
        load_config(path)


def test_budget_fitting_uses_negative_derived_reward_and_separate_losses(tmp_path):
    from attention_fixtures import small_config, tiny_replay

    from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.attention_replay import DecisionSampler

    config = small_config()
    config["training"]["reward"] = {"version": 1}
    replay = tiny_replay(config)
    replay[0]["training_reward"] = -5.5
    model = AttentionStageQ(config)
    metrics = fit_attention_iteration(
        model,
        replay,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        sampler=DecisionSampler(replay, seed=1),
        optimizer_updates=2,
        global_batch=2,
        microbatch=1,
        huber_beta=100,
        conservative_temperature=100,
        conservative_weight=0.01,
    )
    assert metrics["target_min"] == metrics["target_max"] == -5.5
    assert metrics["loss"] == pytest.approx(
        metrics["td_loss"] + metrics["weighted_conservative_loss"]
    )
    assert metrics["diagnostics"]["startup/G"]["count"] == 4
    assert metrics["target_hash_before"] == metrics["target_hash_after"]
    del replay[0]["training_reward"]
    with pytest.raises((KeyError, ValueError)):
        fit_attention_iteration(
            model,
            replay,
            optimizer=torch.optim.Adam(model.parameters()),
            sampler=DecisionSampler(replay, seed=1),
            optimizer_updates=1,
            global_batch=2,
            microbatch=1,
        )


def test_budget_warm_start_preserves_representation_and_resets_learning_state(tmp_path):
    import copy
    import json

    from attention_fixtures import make_batch

    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.attention_training import warm_start_budget
    from latency_meta_mdp.meta.information.deployment import deployment_binding, load_scheduler
    from latency_meta_mdp.meta.information.training import _save

    batch, original = make_batch()
    model = AttentionStageQ(original).eval()
    parent = tmp_path / "parent"
    parent.mkdir()
    _save(
        parent,
        model,
        binding={
            **deployment_binding(original),
            "normalization": "fixed-physical-scales-and-learned-LayerNorm",
        },
        iteration=0,
        metrics=None,
    )
    new = copy.deepcopy(original)
    new["training"]["reward"] = dict(
        version=1,
        delivery_reward=10,
        drop_penalty=5,
        cost_weight=0.5,
        prices=dict(vla=1, dino=0.1, jepa=0.1, decoder=0.1, meta_q=0.01, bootstrap=0),
    )
    new["training"]["budget_lambda"] = 0.5
    destination = warm_start_budget(new, original, parent / "iteration-0000", tmp_path / "budget")
    converted = load_scheduler(destination, deployment_binding(new)).model.eval()
    with torch.no_grad():
        torch.testing.assert_close(converted(batch), model(batch) * 100, rtol=2e-5, atol=1e-5)
    state = torch.load(destination / "state.pt", weights_only=True)
    assert "optimizer" not in state and "sampler" not in state
    receipt = json.loads((destination / "complete.json").read_text())
    assert receipt["binding"]["warm_start"]["q_scale"] == 100
    assert (
        warm_start_budget(new, original, parent / "iteration-0000", tmp_path / "budget")
        == destination
    )
    bad = copy.deepcopy(new)
    bad["actions"]["starvation_mode"] = "allow"
    with pytest.raises(ValueError):
        warm_start_budget(bad, original, parent / "iteration-0000", tmp_path / "bad")


def test_outcome_at_end_boundary_belongs_to_action_before_boundary():
    from types import SimpleNamespace

    from latency_meta_mdp.meta.information.budget_reward import reward_view

    _, _, _, spec = _reward_fixture()
    rows = [
        dict(packet=SimpleNamespace(tick=10), end_tick=20, forced=False, task_reward=0.1),
        dict(packet=SimpleNamespace(tick=20), end_tick=30, forced=False, task_reward=0.1),
    ]
    events = [dict(parcel_id=i, tick=0, event="spawn") for i in range(2)]
    events += [
        dict(parcel_id=0, tick=20, event="success"),
        dict(parcel_id=1, tick=30, event="success"),
    ]
    assert reward_view(rows, [], events, spec)["decision_rewards"] == [10, 10]


@pytest.mark.parametrize(
    "selection", ["mean_nominal_family_incumbent_on_ties", "mean_budget_all_regimes"]
)
def test_validation_reports_cost_without_replacing_delivery_metric(selection):
    from latency_meta_mdp.meta.information.validation import validation_metrics

    rows = []
    for scene in (1, 2, 3):
        for regime in ("zero", "nominal", "family"):
            rows.append(
                dict(
                    episode=dict(
                        scene=scene, regime=regime, behavior="fqi", epsilon=0, split="validation"
                    ),
                    report=dict(
                        task_reward=0.5,
                        terminated=True,
                        truncated=False,
                        starvation_ticks=0,
                        parcels=dict(spawned=10, successes=5, misses=2, timeouts=3, active=0),
                        budget=dict(
                            reward_spec_hash="same",
                            episode_reward=38,
                            compute_cost=4,
                            deliveries=5,
                            drops=2,
                            timeouts=3,
                            module_calls={"vla": 4},
                        ),
                    ),
                )
            )
    result = validation_metrics(rows, scenes=[1, 2, 3], selection=selection)
    assert result["selection_score"] == (38 if selection == "mean_budget_all_regimes" else 0.5)
    assert result["zero/budget_return"] == 38
    assert result["family/compute_cost"] == 4
    assert result["nominal/calls/vla"] == 4
    rows[0]["report"]["budget"]["reward_spec_hash"] = "different"
    with pytest.raises(ValueError):
        validation_metrics(rows, scenes=[1, 2, 3])


def test_budget_selection_accepts_negative_returns_and_breaks_ties_by_delivery_then_cost(tmp_path):
    from latency_meta_mdp.meta.information.validation import record_validation

    metrics = {"selection_score": -5.0}
    for regime in ("zero", "nominal", "family"):
        metrics.update(
            {
                f"{regime}/reward": 0.3,
                f"{regime}/budget_return": -5.0,
                f"{regime}/compute_cost": 100.0,
            }
        )
    protocol = {"selection": "mean_budget_all_regimes"}
    for iteration in range(4):
        path = tmp_path / f"iteration-{iteration:04d}"
        path.mkdir()
        (path / "complete.json").write_text("{}")
        if iteration == 1:
            metrics["zero/reward"] = 0.4
        if iteration == 2:
            metrics["zero/compute_cost"] = 90.0
        state = record_validation(
            tmp_path,
            iteration=iteration,
            metrics=dict(metrics),
            protocol=protocol,
            evaluation_sha256="eval",
        )
        assert state["best_iteration"] == min(iteration, 2)
    with pytest.raises(ValueError, match="selection"):
        record_validation(
            tmp_path,
            iteration=3,
            metrics=dict(metrics, selection_score=1),
            protocol=protocol,
            evaluation_sha256="eval",
        )


def test_wandb_metrics_export_decomposed_losses_and_source_diagnostics():
    from latency_meta_mdp.meta.online_tracking import fit_metrics

    event = dict(
        phase="fit",
        iteration=1,
        optimizer_update=1,
        loss=3.0,
        grad_norm=0.2,
        td_loss=2.0,
        conservative_loss=100.0,
        weighted_conservative_loss=1.0,
    )
    result = fit_metrics(event, episodes=1, epochs=1)
    assert result["train/td_loss"] == 2
    assert result["train/weighted_conservative_loss"] == 1
    result = fit_metrics(
        dict(
            phase="iteration_complete",
            iteration=1,
            metrics=dict(diagnostics={"latest/P": {"taken_q_mean": 3.0}}),
        ),
        episodes=1,
        epochs=1,
    )
    assert result["fqi/diagnostics/latest/P/taken_q_mean"] == 3
