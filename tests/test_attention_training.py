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
