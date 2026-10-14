import copy

import pytest
import torch
from attention_fixtures import small_config, tiny_replay


def train(model, replay, path, **kwargs):
    from latency_meta_mdp.meta.information.training import train_iterations

    return train_iterations(
        model,
        replay,
        path,
        binding=dict(mode="bound", seed=7, schema_version=2),
        total_iterations=2,
        epochs=1,
        learning_rate=0.0001,
        conservative_weight=0.01,
        attention_options=dict(global_batch=2, microbatch=1, optimizer_updates=2),
        **kwargs,
    )


def test_attention_resume_restores_optimizer_sampler_and_target_iteration(tmp_path):
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ

    torch.manual_seed(12)
    config = small_config()
    original = AttentionStageQ(config)
    full, split = copy.deepcopy(original), copy.deepcopy(original)
    replay = tiny_replay(config)
    train(full, replay, tmp_path / "full")
    train(split, replay, tmp_path / "split", stop_after_iterations=1)
    restored = AttentionStageQ(config)
    train(restored, replay, tmp_path / "split")
    for k, v in full.state_dict().items():
        torch.testing.assert_close(v, restored.state_dict()[k], atol=0, rtol=0)
    state = torch.load(tmp_path / "split/latest/state.pt", weights_only=True)
    assert state["sampler"]["visits"] == 8
    assert state["optimizer"]["state"]
    assert state["model_config"]["model"]["architecture_id"] == "attention_memory_v2"


def test_attention_validation_failure_resumes_without_extra_updates(tmp_path):
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.config import sha256

    config = small_config()
    seen = []

    def validate(path, receipt):
        seen.append(receipt["iteration"])
        if seen == [1]:
            raise RuntimeError("validation interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        train(
            AttentionStageQ(config),
            tiny_replay(config),
            tmp_path,
            validate_initial=False,
            validate_checkpoint=validate,
        )
    digest = sha256(tmp_path / "iteration-0001/complete.json")
    train(
        AttentionStageQ(config),
        tiny_replay(config),
        tmp_path,
        validate_initial=False,
        validate_checkpoint=validate,
    )
    assert seen == [1, 1, 2]
    assert sha256(tmp_path / "iteration-0001/complete.json") == digest


@pytest.mark.parametrize("updates,batch,devices", [(None, 64, "0"), (0, 64, "0"), (1, 7, "0,1")])
def test_bad_budget_rejected_before_assets_or_replay(tmp_path, updates, batch, devices):
    from latency_meta_mdp.meta.information.attention_training import run_attention_training

    with pytest.raises(ValueError, match="optimizer_updates|global.batch"):
        run_attention_training(
            "configs/experiments/conveyor_sort/meta_attention_v2.yaml",
            [],
            tmp_path,
            root=tmp_path,
            total_iterations=2,
            optimizer_updates=updates,
            global_batch=batch,
            training_gpus=devices,
        )


def test_periodic_latest_resumes_fixed_target_and_adam_after_lost_work(tmp_path):
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.training import train_iterations

    torch.manual_seed(19)
    initial = AttentionStageQ(small_config())
    replay = tiny_replay(initial.config)
    # Include bootstrapping: refreshing the target at resume would change results.
    replay.episodes[0][0]["terminated"] = False
    replay.episodes[0][0]["next_context"] = replay.episodes[0][0]["context"]
    replay.episodes[0][0]["next_legal"] = replay.episodes[0][0]["legal"]
    options = dict(
        binding=dict(mode="bound", seed=3),
        total_iterations=1,
        epochs=1,
        learning_rate=0.001,
        conservative_weight=0.01,
        attention_options=dict(global_batch=2, microbatch=1, optimizer_updates=4),
        checkpoint_every_updates=2,
    )
    full = copy.deepcopy(initial)
    train_iterations(full, replay, tmp_path / "full", **options)

    def interrupt(event):
        if event["phase"] == "fit" and event["optimizer_update"] == 3:
            raise RuntimeError("time limit")

    with pytest.raises(RuntimeError, match="time limit"):
        train_iterations(
            copy.deepcopy(initial), replay, tmp_path / "split", progress=interrupt, **options
        )
    saved = torch.load(tmp_path / "split/training-0001-latest.pt", weights_only=True)
    assert saved["next_update"] == 2
    restored = copy.deepcopy(initial)
    events = []
    train_iterations(restored, replay, tmp_path / "split", progress=events.append, **options)
    assert [e["optimizer_update"] for e in events if e["phase"] == "fit"] == [3, 4]
    for k, v in full.state_dict().items():
        torch.testing.assert_close(v, restored.state_dict()[k], atol=0, rtol=0)
    resumed_state = torch.load(tmp_path / "split/latest/state.pt", weights_only=True)
    reference_state = torch.load(tmp_path / "full/latest/state.pt", weights_only=True)
    assert resumed_state["sampler"] == reference_state["sampler"]
    for key, value in reference_state["optimizer"]["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(
                tensor, resumed_state["optimizer"]["state"][key][field], atol=0, rtol=0
            )
    assert not (tmp_path / "split/training-0001-latest.pt").exists()


def test_validation_cadence_survives_unvalidated_checkpoint_resume(tmp_path):
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ

    config = small_config()
    seen = []
    options = dict(
        validate_initial=True,
        validation_every=2,
        validate_checkpoint=lambda path, receipt: seen.append(receipt["iteration"]),
    )
    train(
        AttentionStageQ(config), tiny_replay(config), tmp_path, stop_after_iterations=1, **options
    )
    assert seen == [0]
    train(AttentionStageQ(config), tiny_replay(config), tmp_path, **options)
    assert seen == [0, 2]
