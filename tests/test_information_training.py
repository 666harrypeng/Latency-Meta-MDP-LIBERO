import copy
import json

import numpy as np
import pytest
import torch

from latency_meta_mdp.meta.information.model import StageQ
from latency_meta_mdp.meta.information.schema import ActorPacket
from latency_meta_mdp.meta.information.training import train_iterations


def episodes():
    return [
        [
            dict(
                packet=ActorPacket(10, "G"),
                action=5,
                task_reward=0.1,
                cost=0.0,
                duration=1,
                terminated=True,
                next_packet=None,
            )
        ]
    ]


def run(model, data, output, binding, **kwargs):
    return train_iterations(
        model,
        data,
        output,
        binding=binding,
        total_iterations=2,
        epochs=1,
        learning_rate=0.001,
        conservative_weight=0.01,
        **kwargs,
    )


def test_checkpoint_resume_matches_continuous_and_rejects_changed_binding(tmp_path):
    torch.manual_seed(12)
    initial = StageQ()
    continuous = copy.deepcopy(initial)
    split = copy.deepcopy(initial)
    binding = {"mode": "bound", "data": "test", "seed": 0}
    run(continuous, episodes(), tmp_path / "full", binding)
    first = run(split, episodes(), tmp_path / "split", binding, stop_after_iterations=1)
    assert first["completed_iterations"] == 1
    # A crash before publication is not a resumable checkpoint.
    (tmp_path / "split" / ".pending-partial").mkdir()
    restarted = StageQ()
    run(restarted, episodes(), tmp_path / "split", binding)
    for key, value in continuous.state_dict().items():
        torch.testing.assert_close(value, restarted.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="binding"):
        run(StageQ(), episodes(), tmp_path / "split", {**binding, "data": "different"})
    checkpoint = tmp_path / "split" / "iteration-0002" / "state.pt"
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        run(StageQ(), episodes(), tmp_path / "split", binding)


def test_each_iteration_keeps_its_target_fixed_and_emits_progress(tmp_path):
    progress = []
    result = run(StageQ(), episodes(), tmp_path, {"mode": "bound"}, progress=progress.append)
    assert result["completed_iterations"] == 2
    assert result["metrics"]["target_hash_before"] == result["metrics"]["target_hash_after"]
    assert progress and any(x["phase"] == "fit" for x in progress)
    assert np.isfinite(result["metrics"]["loss"])
    receipt = json.loads((tmp_path / "iteration-0002" / "complete.json").read_text())
    assert receipt["binding"]["mode"] == "bound"


def test_streaming_replay_requires_complete_training_binding(tmp_path):
    from latency_meta_mdp.meta.information.cohort import complete_attempt
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.records import StageRecorder
    from latency_meta_mdp.meta.information.training import VerifiedReplay

    binding = {
        "assets": {"identity": {"vla": "pinned"}},
        "mode": "bound",
        "code_hash": "collection",
        "scope": "starter_training",
    }
    episode = {
        "episode_id": "1-nominal-lazy",
        "scene": 1,
        "regime": "nominal",
        "behavior": "lazy",
        "scheduler_seed": 100,
        "policy_seed": 0,
        "split": "train",
    }
    manifest = {"binding": binding, "episodes": [episode]}
    (tmp_path / "cohort.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete"):
        VerifiedReplay(tmp_path, manifest)
    attempt = tmp_path / "episodes/1-nominal-lazy/attempt-000"
    attempt.mkdir(parents=True)
    recorder = StageRecorder("test")
    recorder.start(ActorPacket(10, "G"), 5, forced=False, probability=1)
    recorder.add_reward(10, 0)
    recorder.finish(11, terminated=True)
    from latency_meta_mdp.io.paths import repository_root
    from latency_meta_mdp.runtime.evaluate_conveyor import conveyor_delay_sampler

    _, delay = conveyor_delay_sampler(
        repository_root(), regime="nominal", scene_seed=1, policy_seed=0
    )
    inner = {
        "assets": {"vla": "pinned"},
        "mode": "bound",
        "scene": 1,
        "delay": delay,
        "scope": "starter_training",
        "code_hash": "collection",
        "scheduler_seed": 100,
        "behavior": "lazy",
        "policy_seed": 0,
    }
    save_episode(attempt / "episode", recorder, report={}, binding=inner)
    (attempt / "summary.json").write_text(
        json.dumps(
            dict(
                terminated=True,
                truncated=False,
                starvation_ticks=0,
                scope="starter_training",
                task_reward=0,
                parcels=dict(spawned=10, successes=0, misses=10, timeouts=0, active=0),
            )
        )
    )
    complete_attempt(attempt, binding={"cohort": binding, "episode": episode})
    replay = VerifiedReplay(tmp_path, manifest)
    assert len(replay) == 1 and len(next(iter(replay))) == 1
    wrong = copy.deepcopy(manifest)
    wrong["episodes"][0]["split"] = "test"
    with pytest.raises(ValueError, match="train"):
        VerifiedReplay(tmp_path, wrong)
    # Self-consistent receipt hashes do not authorize different latency semantics.
    receipt = attempt / "episode/complete.json"
    original_receipt = json.loads(receipt.read_text())
    changed = copy.deepcopy(original_receipt)
    changed["binding"]["delay"]["seed_components"][0] += 1
    receipt.write_text(json.dumps(changed))
    complete_attempt(attempt, binding={"cohort": binding, "episode": episode})
    with pytest.raises(ValueError, match="binding"):
        VerifiedReplay(tmp_path, manifest)
    receipt.write_text(json.dumps(original_receipt))
    complete_attempt(attempt, binding={"cohort": binding, "episode": episode})
    # Even an internally re-signed wrong scope may not enter training.
    inner["scope"] = "diagnostic"
    receipt = attempt / "episode/complete.json"
    contents = json.loads(receipt.read_text())
    contents["binding"] = inner
    receipt.write_text(json.dumps(contents))
    complete_attempt(attempt, binding={"cohort": binding, "episode": episode})
    with pytest.raises(ValueError, match="binding"):
        VerifiedReplay(tmp_path, manifest)


def test_warm_start_new_generation_retains_parent_and_rejects_parent_change(tmp_path):
    from latency_meta_mdp.meta.information.online_training import start_generation

    model = StageQ()
    with torch.no_grad():
        model.heads[0].bias.fill_(0.2)
    parent = tmp_path / "parent"
    binding = dict(
        mode="bound",
        algorithm="fqi",
        schema_version=1,
        assets={"vla": "fixed"},
        normalization="fixed-physical-scales-and-learned-LayerNorm",
    )
    run(model, episodes(), parent, binding, stop_after_iterations=1)
    new = tmp_path / "new"
    loaded = start_generation(parent / "iteration-0001", new, binding, device="cpu")
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v, rtol=0, atol=0)
    assert all(p.requires_grad for p in loaded.parameters())
    assert not (new / "iteration-0000").exists()
    run(loaded, episodes(), new, binding, stop_after_iterations=1)
    with pytest.raises(ValueError, match="parent"):
        start_generation(parent / "iteration-0000", new, binding, device="cpu")


def test_shuffled_mixed_replay_resume_matches_continuous(tmp_path):
    from latency_meta_mdp.meta.information.online_training import MixedReplay

    torch.manual_seed(44)
    initial = StageQ()
    data = MixedReplay(episodes(), [[{**episodes()[0][0], "task_reward": 0.2}]])
    continuous, split = copy.deepcopy(initial), copy.deepcopy(initial)
    np.random.seed(7)
    run(continuous, data, tmp_path / "full", {"mode": "bound"})
    np.random.seed(7)
    run(split, data, tmp_path / "split", {"mode": "bound"}, stop_after_iterations=1)
    resumed = StageQ()
    run(resumed, data, tmp_path / "split", {"mode": "bound"})
    for k, v in continuous.state_dict().items():
        torch.testing.assert_close(v, resumed.state_dict()[k], rtol=0, atol=0)


def test_generation_skips_initial_validation_but_retries_completed_fit(tmp_path):
    from latency_meta_mdp.meta.information.training import train_iterations

    seen = []

    def validate(checkpoint, receipt):
        seen.append(receipt["iteration"])
        if len(seen) == 1:
            raise RuntimeError("validation interrupted")

    kwargs = dict(
        binding={"mode": "bound"},
        total_iterations=1,
        epochs=1,
        learning_rate=0.001,
        conservative_weight=0.01,
        validate_initial=False,
        validate_checkpoint=validate,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        train_iterations(StageQ(), episodes(), tmp_path, **kwargs)
    receipt = (tmp_path / "iteration-0001/complete.json").read_bytes()
    train_iterations(StageQ(), episodes(), tmp_path, **kwargs)
    assert seen == [1, 1]
    assert (tmp_path / "iteration-0001/complete.json").read_bytes() == receipt
