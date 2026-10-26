"""Versioned scheduling authority, independent of expensive frozen model inference."""

import copy

import numpy as np
import pytest
import torch
import yaml
from attention_fixtures import make_batch, small_config

from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.config import load_config
from latency_meta_mdp.meta.information.memory_schema import ActionSpec


def optional_config():
    config = small_config()
    config.update(schema_version=3, mode="optional_prediction", experiment_id="optional-test-v1")
    config["actions"].update(version=3, starvation_mode="allow_hold", reserve=0)
    return config


def test_optional_periodic_rule_can_wait_beyond_action_chunk():
    from latency_meta_mdp.meta.information.behaviors import AttentionRuleScheduler
    from latency_meta_mdp.meta.information.memory_schema import OptionalPredictionPacket

    spec = ActionSpec(**optional_config()["actions"])
    rule = AttentionRuleScheduler("periodic", action_spec=spec, period=75)
    packet = OptionalPredictionPacket(
        60, "G", plan_age=60, buffer_mask=np.zeros(50, bool), action_spec=spec, decision_key=(60, 0)
    )
    decision = rule.decide(packet, spec.legal("G", 0))
    assert decision.action < len(spec.sleep_ticks)
    with pytest.raises(ValueError, match="reserve"):
        AttentionRuleScheduler(
            "periodic", action_spec=ActionSpec(**small_config()["actions"]), period=75
        )


@pytest.mark.parametrize("protocol", [2, 3])
def test_clean_bootstrap_assets_required_only_by_legacy_protocol(tmp_path, protocol):
    config = optional_config() if protocol == 3 else small_config()
    assets = config["assets"]
    prefix = assets["paths"]["clean"] + "/"
    assets["files"] = {p: sha for p, sha in assets["files"].items() if not p.startswith(prefix)}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    if protocol == 3:
        assert load_config(path)["assets"]["files"] == assets["files"]
    else:
        with pytest.raises(ValueError, match="outside the asset binding"):
            load_config(path)


def test_optional_replay_roundtrip_preserves_free_observe_rows_and_readiness(tmp_path):
    from latency_meta_mdp.meta.information.attention_replay import contexts_from_raw
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import load_episode, save_episode
    from latency_meta_mdp.meta.information.packed_replay import (
        load_packed_raw,
        prepare_packed_episode,
    )

    result, recorder, _ = run_optional_episode(
        {
            (0, "G"): "observe",
            (0, "O"): "sleep:12",
            (12, "G"): "observe",
            (12, "O"): "predict",
            (12, "P"): "launch",
        },
        steps=60,
    )
    path = tmp_path / "episode"
    binding = {"mode": "optional_prediction", "actions": optional_config()["actions"]}
    save_episode(path, recorder, report=result, binding=binding)
    raw = load_episode(path, binding=binding, raw=True)
    entry = prepare_packed_episode(
        path,
        tmp_path / "cache",
        binding=binding,
        expected_receipt=sha256(path / "complete.json"),
        raw=True,
    )
    packed = load_packed_raw(entry)
    for decoded in (raw, packed):
        rows = contexts_from_raw(decoded, recorder.action_spec, config=optional_config()["memory"])
        assert len(rows) == len(recorder.rows)
        observed = [r for r in rows if r["packet"].stage == "O"]
        assert [r["packet"].prediction_ready for r in observed] == [False, True]
        assert [bool(r["legal"][9]) for r in observed] == [False, True]
        assert all(not r["forced"] for r in decoded["rows"])
        assert all(r["legal"][r["action"]] for r in rows)
        assert sum(r["task_reward"] for r in rows) == pytest.approx(result["task_reward"])


def test_optional_sources_reject_old_schema_and_accept_new_identity():
    from latency_meta_mdp.meta.information.attention_replay import validate_sources
    from latency_meta_mdp.meta.information.config import identity_hash

    config = optional_config()
    manifest = dict(
        binding=dict(
            schema_version=3,
            mode=config["mode"],
            actions=config["actions"],
            assets=config["assets"],
            config_hash=identity_hash(config),
            scope="starter_training",
        ),
        episodes=[dict(episode_id="one", scene=config["splits"]["train"][0], split="train")],
    )
    validate_sources(config, [manifest])
    for version in (1, 2):
        manifest["binding"]["schema_version"] = version
        with pytest.raises(ValueError):
            validate_sources(config, [manifest])


def test_optional_reward_attributes_same_tick_costs_and_hold_outcomes_once():
    from types import SimpleNamespace

    from test_attention_budget_training import _reward_fixture

    from latency_meta_mdp.meta.information.budget_reward import reward_view

    spec = _reward_fixture()[-1]
    # G->O->P at tick 0, followed by physical hold, then a current-only launch.
    rows = [
        dict(packet=SimpleNamespace(tick=t), end_tick=end, forced=False, task_reward=r)
        for t, end, r in [(0, 0, 0), (0, 0, 0), (0, 5, 0.5), (5, 5, 0), (5, 12, 0.5)]
    ]
    operations = [
        dict(operation_id=str(i), component=c, owner=i)
        for i, c in enumerate(["dino", "jepa", "meta_q", "dino", "vla"])
    ]
    events = [dict(parcel_id=i, tick=0, event="spawn") for i in range(11)]
    events += [dict(parcel_id=i, tick=i + 1, event="success") for i in range(10)]
    events += [dict(parcel_id=10, tick=11, event="miss")]
    view = reward_view(rows, operations, events, spec)
    cost = sum(spec["prices"][op["component"]] for op in operations)
    assert view["prefix_reward"] == 0
    assert len(view["decision_rewards"]) == 5
    assert view["episode_reward"] == pytest.approx(100 - 5 - spec["cost_weight"] * cost)
    assert sum(view["decision_rewards"]) == pytest.approx(view["episode_reward"])


def test_optional_rule_collectors_cover_both_launch_paths_and_never_force_predict():
    from latency_meta_mdp.meta.information.behaviors import AttentionRuleScheduler
    from latency_meta_mdp.meta.information.memory_schema import OptionalPredictionPacket

    spec = ActionSpec(**optional_config()["actions"])
    packet = OptionalPredictionPacket(0, "O", action_spec=spec, decision_key=(0, 1))
    for kind in ("periodic", "active", "lazy", "exploration"):
        scheduler = AttentionRuleScheduler(kind, action_spec=spec, seed=9)
        scheduler.reset(episode_id="one")
        legal = spec.legal("O", 0)
        decisions = [scheduler.decide(packet, legal) for _ in range(200)]
        assert all(legal[d.action] for d in decisions)
        if kind == "exploration":
            legal = spec.legal("O", 0, prediction_ready=True)
            seen = {scheduler.decide(packet, legal).action for _ in range(200)}
            assert {9, 10}.issubset(seen)


def test_optional_cohort_and_validation_use_new_protocol_without_bootstrap(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import runtime_identity
    from latency_meta_mdp.meta.information.cohort import build_manifest
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest
    from latency_meta_mdp.meta.information.monitor import IterationValidator

    config = optional_config()
    audit = dict(
        local_inventory_pass=True,
        candidate_splits=config["splits"],
        excluded_scenes=config["excluded_scenes"],
        sources=[],
    )
    manifest = build_manifest(config, audit, root=tmp_path, code_hash="test")
    assert manifest["binding"]["schema_version"] == 3
    assert manifest["binding"]["actions"] == config["actions"]
    evaluation = build_cycle_manifest(
        config,
        audit,
        root=tmp_path,
        code_hash="test",
        policy={},
        phase="monitor",
        baselines=False,
    )
    assert len(evaluation["episodes"]) == 9
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}")
    monkeypatch.setattr(runtime_identity, "source_identity", lambda root: {"code_hash": "test"})
    validator = IterationValidator(
        config_path,
        tmp_path / "evaluation",
        root=tmp_path,
        audit_path=audit_path,
        bootstrap_dir=tmp_path / "nonexistent-bootstrap",
    )
    assert validator.protocol["starvation_mode"] == "allow_hold"
    assert validator.protocol["initial_buffer"] == "empty_hold"


@pytest.mark.parametrize("remaining", [0, 1, 20, 50])
def test_optional_actions_allow_wait_and_current_only_launch_without_forcing_prediction(remaining):
    spec = ActionSpec(version=3, starvation_mode="allow_hold", reserve=0)
    assert np.flatnonzero(spec.legal("G", remaining)).tolist() == list(range(9))
    assert np.flatnonzero(spec.legal("O", remaining)).tolist() == [*range(8), 10]
    assert np.flatnonzero(spec.legal("O", remaining, prediction_ready=True)).tolist() == [
        *range(8),
        9,
        10,
    ]
    assert np.flatnonzero(spec.legal("P", remaining, forecast_fresh=True)).tolist() == [
        *range(8),
        10,
    ]
    with pytest.raises(ValueError, match="forecast"):
        spec.legal("P", remaining)


def test_optional_config_is_versioned_and_cannot_reintroduce_starvation_guards(tmp_path):
    config = optional_config()
    path = tmp_path / "optional.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["schema_version"] == 3
    for field, value in [("version", 2), ("starvation_mode", "guarded"), ("reserve", 20)]:
        invalid = copy.deepcopy(config)
        invalid["actions"][field] = value
        path.write_text(yaml.safe_dump(invalid))
        with pytest.raises(ValueError):
            load_config(path)


def test_optional_observe_value_is_learnable_without_forecast_and_old_model_rejects_it():
    batch, old_config = make_batch(missing_forecast=True)
    batch.tensors["stage"].fill_(1)
    with pytest.raises(ValueError, match="forced"):
        AttentionStageQ(old_config)(batch)
    model = AttentionStageQ(optional_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad()
        values = model(batch)
        assert torch.isfinite(values).all()
        (values[:, 10] - 1).square().mean().backward()
        optimizer.step()
    assert model(batch)[:, 10].min().item() > 0
    assert model.heads[1].weight.grad.abs().sum().item() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.history.parameters())
    batch.tensors["forecast"].fill_(1000)
    assert torch.isfinite(model(batch)).all()


def run_optional_episode(schedule, *, delay=4, steps=24):
    from pathlib import Path

    from test_action_conditioned_jepa_rollout import _normalization
    from test_conveyor_policy_evaluation import ConveyorFixture
    from test_information_perception import Decoder, Encoder, Predictor

    from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
    from latency_meta_mdp.meta.information.behaviors import Decision
    from latency_meta_mdp.meta.information.perception import PerceptionAccess
    from latency_meta_mdp.meta.information.records import StageRecorder
    from latency_meta_mdp.runtime.information_execution import run_information_episode
    from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config

    spec = ActionSpec(version=3, starvation_mode="allow_hold", reserve=0)
    recorder = StageRecorder("optional", action_spec=spec)
    access = PerceptionAccess(
        FrozenForecastEngine(Predictor(), Decoder(), device="cpu"),
        Encoder(),
        _normalization(),
        recorder,
    )
    contexts = []

    class Scripted:
        component = "meta_q"

        def reset(self, *, episode_id):
            pass

        def decide(self, packet, legal_mask):
            action = spec.id_for(schedule.get((packet.tick, packet.stage), "sleep:16"))
            assert legal_mask[action]
            return Decision(action)

    def policy(obs, context):
        contexts.append(context)
        return np.full((50, 7), 0.25)

    def bootstrap(obs):
        pytest.fail("new protocol must never call the clean bootstrap")

    report = run_information_episode(
        runtime=ConveyorFixture(),
        policy=policy,
        bootstrap_policy=bootstrap,
        perception=access,
        scheduler=Scripted(),
        recorder=recorder,
        client_config=load_rtc_client_config(
            Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")
        ),
        delay_sampler=lambda: delay,
        maximum_steps=steps,
        mode="optional_prediction",
        action_spec=spec,
    )
    return report, recorder, contexts


@pytest.mark.parametrize("delay", [0, 4, 20])
def test_current_only_launch_from_tick_zero_does_not_invoke_predictor_or_decoder(delay):
    report, recorder, contexts = run_optional_episode(
        {(0, "G"): "observe", (0, "O"): "launch"}, delay=delay
    )
    assert len(contexts) == 1
    assert report["scheduling"]["current_only_launches"] == 1
    assert report["scheduling"]["predicted_launches"] == 0
    assert report["scheduling"]["hold_ticks"] == delay
    assert contexts[0].origin_tick == 0
    assert contexts[0].forecast.rgb is None
    assert contexts[0].forecast.proprio is None
    components = [op["component"] for op in recorder.operations]
    assert not {"jepa", "decoder", "bootstrap"}.intersection(components)
    assert components.count("vla") == 1
    assert report["starvation_ticks"] == delay
    np.testing.assert_array_equal(report["executed_actions"][:delay, :6], 0)
    np.testing.assert_array_equal(report["executed_actions"][delay:], 0.25)
    assert recorder.prefix_reward == 0
    assert any(r["packet"].stage == "O" and not r["forced"] for r in recorder.rows)


def test_prediction_followed_by_sleep_cannot_leak_to_later_direct_launch():
    report, recorder, contexts = run_optional_episode(
        {
            (0, "G"): "observe",
            (0, "O"): "sleep:12",
            (12, "G"): "observe",
            (12, "O"): "predict",
            (12, "P"): "sleep:4",
            (16, "G"): "observe",
            (16, "O"): "launch",
        }
    )
    assert len(contexts) == 1 and contexts[0].origin_tick == 16
    assert contexts[0].forecast.rgb is None
    assert report["starvation_ticks"] == 20
    components = [op["component"] for op in recorder.operations]
    assert components.count("jepa") == 1
    assert components.count("decoder") == 0
    later = [r["packet"] for r in recorder.rows if r["packet"].tick == 16]
    assert all(p.forecast_tokens is None for p in later)


def test_predicted_launch_decodes_once_and_keeps_observation_source_identity():
    _, recorder, contexts = run_optional_episode(
        {
            (0, "G"): "sleep:12",
            (12, "G"): "observe",
            (12, "O"): "predict",
            (12, "P"): "launch",
        }
    )
    assert len(contexts) == 1
    context = contexts[0]
    assert context.forecast.rgb is not None
    assert context.forecast.source_tick == context.origin_tick == 12
    assert context.forecast.buffer_version == context.buffer_version
    components = [op["component"] for op in recorder.operations]
    assert components.count("jepa") == components.count("decoder") == components.count("vla") == 1


def test_voluntary_wait_advances_the_world_without_free_perception_or_inference():
    report, recorder, contexts = run_optional_episode({}, steps=60)
    assert contexts == []
    assert report["executed_steps"] == report["starvation_ticks"] == 60
    np.testing.assert_array_equal(report["executed_actions"][:, :6], 0)
    assert report["parcels"]["successes"] == 2  # Fixture events test reward ownership during hold.
    assert recorder.prefix_reward == 0
    assert sum(r["task_reward"] for r in recorder.rows) == pytest.approx(0.2)
    components = {op["component"] for op in recorder.operations}
    assert not {"dino", "jepa", "decoder", "vla", "bootstrap"}.intersection(components)


@pytest.mark.parametrize("scope", ["starter_training", "evaluation"])
def test_collection_entrypoint_saves_optional_episode_without_bootstrap(
    tmp_path, monkeypatch, scope
):
    pytest.importorskip("jax")
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace

    from test_action_conditioned_jepa_rollout import _normalization
    from test_conveyor_policy_evaluation import ConveyorFixture
    from test_information_perception import Decoder, Encoder, Predictor

    from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
    from latency_meta_mdp.envs.conveyor import meta_audit, task
    from latency_meta_mdp.io import policy_video
    from latency_meta_mdp.meta.information import collection
    from latency_meta_mdp.runtime import policy_execution

    runtime = ConveyorFixture()
    runtime.world.goal = SimpleNamespace(events=[])
    monkeypatch.setattr(task, "make_conveyor_runtime", lambda *args, **kwargs: runtime)
    video = SimpleNamespace(write=lambda obs: None)
    monkeypatch.setattr(policy_video, "DualCameraVideoWriter", lambda path: nullcontext(video))
    monkeypatch.setattr(
        meta_audit,
        "ProgressAudit",
        lambda *a, **kw: SimpleNamespace(events=[], observe=lambda world, tick: None),
    )
    monkeypatch.setattr(
        policy_execution,
        "InProcessRtcOpenpiPolicy",
        lambda *a, **kw: lambda obs, context: {"actions": np.zeros((50, 7), np.float32)},
    )
    services = SimpleNamespace(
        policy=SimpleNamespace(),
        encoder=Encoder(),
        engine=FrozenForecastEngine(Predictor(), Decoder(), device="cpu"),
        normalization=_normalization(),
    )
    report = collection.run_episode(
        Path.cwd(),
        optional_config(),
        tmp_path,
        services,
        code_hash="test",
        maximum_steps=60,
        regime="zero",
        behavior="active",
        seed=10000,
        scheduler_seed=0,
        bootstrap_dir=None,
        scope=scope,
    )
    assert runtime.closed
    assert report["terminated"]
    assert report["bootstrap_cache"] is None
    assert report["scope"] == scope
    assert (tmp_path / "episode/records.json").is_file()
    if scope == "starter_training":
        assert (tmp_path / "episode/complete.json").is_file()
    else:
        assert not (tmp_path / "episode/visual.npz").exists()
    assert not (tmp_path / "bootstrap").exists()


def test_legacy_diagnostic_rejects_optional_mode_before_loading_models(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import preflight
    from latency_meta_mdp.meta.information.collection import run_smoke

    config = tmp_path / "optional.yaml"
    config.write_text(yaml.safe_dump(optional_config()))
    monkeypatch.setattr(preflight, "preflight", lambda *a, **kw: pytest.fail("models requested"))
    with pytest.raises(ValueError, match="diagnostic"):
        run_smoke(config, tmp_path / "run", root=tmp_path, maximum_steps=60, diagnose=True)


def test_matched_periodic_baselines_only_change_forecast_choice():
    from latency_meta_mdp.meta.information.behaviors import AttentionRuleScheduler
    from latency_meta_mdp.meta.information.memory_schema import OptionalPredictionPacket

    spec = ActionSpec(**optional_config()["actions"])
    current = AttentionRuleScheduler("periodic", action_spec=spec, forecast_mode="current_only")
    predicted = AttentionRuleScheduler("periodic", action_spec=spec, forecast_mode="predict")
    for stage, age in [("G", 0), ("G", 25), ("O", 25)]:
        packet = OptionalPredictionPacket(
            32, stage, action_spec=spec, decision_key=(32, 1), prediction_ready=True, plan_age=age
        )
        mask = spec.legal(stage, 0, prediction_ready=True)
        left, right = current.decide(packet, mask), predicted.decide(packet, mask)
        if stage == "G":
            assert left.action == right.action
        else:
            assert left.action == spec.id_for("launch")
            assert right.action == spec.id_for("predict")
    with pytest.raises(ValueError, match="optional"):
        AttentionRuleScheduler("periodic", action_spec=ActionSpec(), forecast_mode="current_only")
    with pytest.raises(ValueError, match="forecast"):
        AttentionRuleScheduler("periodic", action_spec=spec, forecast_mode="unknown")
