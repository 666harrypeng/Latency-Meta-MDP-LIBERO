"""Scratch identity, learned O-stage regression, and portable protocol-3 recovery."""

import copy
import json

import numpy as np
import pytest
import torch
from test_optional_prediction import optional_config


def test_optional_validation_reports_hold_and_request_weighted_forecast_use():
    from latency_meta_mdp.meta.information.validation import validation_metrics

    rows = []
    for scene in (1, 2, 3):
        for regime in ("zero", "nominal", "family"):
            launches = 0 if regime == "zero" else scene
            predicted = int(launches > 1)
            rows.append(
                dict(
                    episode=dict(
                        scene=scene, regime=regime, behavior="fqi", epsilon=0, split="validation"
                    ),
                    report=dict(
                        task_reward=0.0,
                        terminated=True,
                        truncated=False,
                        starvation_ticks=100,
                        parcels=dict(spawned=10, successes=0, misses=10, timeouts=0, active=0),
                        scheduling=dict(
                            hold_ticks=100,
                            observe_requests=launches,
                            predict_requests=predicted,
                            launch_requests=launches,
                            predicted_launches=predicted,
                            current_only_launches=launches - predicted,
                        ),
                    ),
                )
            )
    metrics = validation_metrics(rows, scenes=[1, 2, 3], starvation_mode="allow_hold")
    assert metrics["zero/hold_ticks"] == 100
    assert metrics["zero/launch_requests"] == 0
    assert metrics["zero/forecast_launch_fraction"] == 0
    assert metrics["zero/reward"] == 0
    assert metrics["family/launch_requests"] == 2
    assert metrics["family/current_only_launches"] == pytest.approx(4 / 3)
    assert metrics["family/forecast_launch_fraction"] == pytest.approx(2 / 6)
    del rows[0]["report"]["scheduling"]
    with pytest.raises(ValueError, match="mixed scheduling"):
        validation_metrics(rows, scenes=[1, 2, 3], starvation_mode="allow_hold")


def test_optional_scratch_is_seeded_and_rejects_old_warm_start(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import model as model_module
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.attention_training import (
        initialize_attention,
        warm_start_budget,
    )
    from latency_meta_mdp.meta.information.deployment import deployment_binding, load_scheduler
    from latency_meta_mdp.meta.information.fqi import model_hash

    config = optional_config()
    # Skip unavailable frozen normalization assets, not the actual Meta initializer.
    monkeypatch.setattr(model_module, "build_model", lambda c, root: AttentionStageQ(c))
    checkpoints = [
        initialize_attention(config, tmp_path / str(i), root=tmp_path, seed=17) for i in range(2)
    ]
    receipt = json.loads((checkpoints[0] / "complete.json").read_text())
    identity = receipt["binding"]["initialization"]
    assert identity == dict(
        kind="scratch",
        seed=17,
        initial_model_hash=receipt["model_hash"],
        experiment_id=config["experiment_id"],
    )
    restored = [load_scheduler(p, deployment_binding(config)).model for p in checkpoints]
    assert model_hash(restored[0]) == model_hash(restored[1])
    state = torch.load(checkpoints[0] / "state.pt", weights_only=True)
    assert "optimizer" not in state and "sampler" not in state
    assert len(restored[0].heads) == 3
    with pytest.raises(ValueError, match="seed|binding"):
        initialize_attention(config, tmp_path / "0", root=tmp_path, seed=18)
    with pytest.raises(ValueError, match="scratch"):
        warm_start_budget(config, config, tmp_path / "never-read", tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()
    other = copy.deepcopy(config)
    other["experiment_id"] = "different-budget-experiment"
    with pytest.raises(ValueError):
        load_scheduler(checkpoints[0], deployment_binding(other))


def test_optional_campaign_forbids_external_initialization_before_reading_files(tmp_path):
    from latency_meta_mdp.meta.information.attention_campaign import validate_recipe

    for override in (
        {"initial_checkpoint": "old/S05"},
        {"cycle_offset": 1},
        {"initial_online_sources": [{"path": "old/data"}]},
    ):
        recipe = dict(initial_checkpoint=None, cycle_offset=0, initial_online_sources=[])
        recipe.update(override)
        with pytest.raises(ValueError, match="scratch"):
            validate_recipe(optional_config(), recipe, {}, root=tmp_path)


@pytest.mark.parametrize("microbatch", [1, 2])
def test_optional_fit_preserves_O_rewards_and_exact_resume(tmp_path, microbatch):
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay, contexts_from_raw
    from latency_meta_mdp.meta.information.deployment import deployment_binding, load_scheduler
    from latency_meta_mdp.meta.information.fqi import model_hash
    from latency_meta_mdp.meta.information.memory_schema import ActionSpec, OptionalPredictionPacket
    from latency_meta_mdp.meta.information.records import StageRecorder
    from latency_meta_mdp.meta.information.training import train_iterations

    config = optional_config()
    config["training"]["reward"] = {"version": 1}  # Derived reward is mandatory at fitting.
    spec = ActionSpec(**config["actions"])
    recorder = StageRecorder("one", action_spec=spec)
    recorder.start(
        OptionalPredictionPacket(0, "G", action_spec=spec, decision_key=(0, 1)),
        8,
        forced=False,
        probability=1,
    )
    recorder.start(
        OptionalPredictionPacket(0, "O", action_spec=spec, decision_key=(0, 2)),
        10,
        forced=False,
        probability=1,
    )
    recorder.finish(10, terminated=True)
    rows = contexts_from_raw(
        dict(episode_id="one", rows=recorder.rows, events=[]), spec, config=config["memory"]
    )
    for row, value in zip(rows, [-0.1, 95.0]):
        row["training_reward"] = value
    replay = DecisionReplay([rows])
    torch.manual_seed(17)
    initial = AttentionStageQ(config)
    binding = dict(
        deployment_binding(config),
        seed=17,
        normalization="fixed-physical-scales-and-learned-LayerNorm",
        initialization=dict(
            kind="scratch",
            seed=17,
            initial_model_hash=model_hash(initial),
            experiment_id=config["experiment_id"],
        ),
    )

    def fit(model, path, **kwargs):
        return train_iterations(
            model,
            replay,
            path,
            binding=binding,
            total_iterations=2,
            epochs=1,
            learning_rate=1e-4,
            conservative_weight=0.01,
            validate_initial=False,
            attention_options=dict(global_batch=2, microbatch=microbatch, optimizer_updates=3),
            **kwargs,
        )

    full, split = copy.deepcopy(initial), copy.deepcopy(initial)
    fit(full, tmp_path / "full")
    fit(split, tmp_path / "split", stop_after_iterations=1)
    fit(split, tmp_path / "split")
    assert model_hash(full) == model_hash(split)
    saved = torch.load(tmp_path / "split/latest/state.pt", weights_only=True)
    expected = torch.load(tmp_path / "full/latest/state.pt", weights_only=True)
    assert saved["sampler"]["visits"] == 12
    assert saved["sampler"] == expected["sampler"]
    for key, value in saved["optimizer"]["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(
                tensor, expected["optimizer"]["state"][key][name], rtol=0, atol=0
            )
    np.testing.assert_array_equal(saved["torch_rng"], expected["torch_rng"])
    receipt = json.loads((tmp_path / "split/latest/complete.json").read_text())
    assert receipt["metrics"]["target_max"] >= 95
    assert receipt["metrics"]["diagnostics"]["startup/O"]["count"] > 0
    deployed = load_scheduler(tmp_path / "split/latest", deployment_binding(config))
    assert deployed.version == model_hash(split)


def test_new_experiment_rejects_old_wandb_writer_identity(tmp_path, monkeypatch):
    import sys

    from latency_meta_mdp.meta import campaign_tracking

    root = tmp_path / "new-experiment"
    telemetry = root / "telemetry-main"
    telemetry.mkdir(parents=True)
    identity = dict(id="old-C10", binding=dict(project="old-budget", experiment="old"))
    (telemetry / "wandb-run.json").write_text(json.dumps(identity))
    config = tmp_path / "new.yaml"
    config.write_text("schema_version: 3\nexperiment_id: optional-test-v1\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "track",
            "--root",
            str(root),
            "--operations",
            str(tmp_path / "ops"),
            "--config",
            str(config),
            "--project",
            "new-budget",
            "--name",
            "new",
            "--once",
        ],
    )
    with pytest.raises(ValueError, match="identity/config changed"):
        campaign_tracking.main()
    assert json.loads((telemetry / "wandb-run.json").read_text()) == identity


def test_old_parent_rejected_before_assets_or_replay(tmp_path, monkeypatch):
    import yaml

    from latency_meta_mdp.meta.information import preflight
    from latency_meta_mdp.meta.information.attention_training import run_attention_training

    config = optional_config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    parent = tmp_path / "old-checkpoint"
    parent.mkdir()
    (parent / "complete.json").write_text(json.dumps(dict(binding=dict(mode="bound"))))
    monkeypatch.setattr(preflight, "verify_assets", lambda *a: pytest.fail("old parent admitted"))
    with pytest.raises(ValueError, match="binding"):
        run_attention_training(
            path,
            [],
            tmp_path / "new",
            root=tmp_path,
            checkpoint=parent,
            total_iterations=1,
            optimizer_updates=1,
            global_batch=2,
        )


def test_validation_keeps_individual_scene_outcomes():
    from latency_meta_mdp.meta.information.validation import validation_metrics
    rows = [dict(episode=dict(scene=s, regime=r, behavior='fqi', epsilon=0, split='validation'),
                 report=dict(task_reward=s / 10, terminated=True, truncated=False,
                             starvation_ticks=0,
                             parcels=dict(spawned=10, successes=s, misses=9-s,
                                          timeouts=1, active=0)))
            for s in (1, 2, 3) for r in ('zero', 'nominal', 'family')]
    m = validation_metrics(rows, scenes=[1, 2, 3])
    assert m['zero/scene-1/delivery_ratio'] == .1
    assert m['family/scene-3/timeouts'] == 1
