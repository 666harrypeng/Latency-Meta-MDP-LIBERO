import copy
from pathlib import Path

import pytest
import torch
from attention_fixtures import small_config, tiny_replay

from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.attention_replay import DecisionSampler
from latency_meta_mdp.meta.information.distributed_fqi import fit_attention_distributed


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("warm_optimizer", [False, True])
def test_global_decision_batch_single_multi_cpu_equivalence(world, warm_optimizer):
    config = small_config()
    torch.manual_seed(7)
    first = AttentionStageQ(config)
    replay = tiny_replay(config)
    initial_optimizer = torch.optim.Adam(first.parameters(), lr=1e-4)
    if warm_optimizer:
        from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration

        fit_attention_iteration(
            first,
            replay,
            optimizer=initial_optimizer,
            sampler=DecisionSampler(replay, seed=7),
            optimizer_updates=2,
            global_batch=8,
            microbatch=2,
        )
    second = copy.deepcopy(first)
    states = []
    for model, devices in [(first, ["cpu"]), (second, ["cpu"] * world)]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer.state_dict()))
        sampler = DecisionSampler(replay, seed=42)
        metrics = fit_attention_distributed(
            model,
            replay,
            devices=devices,
            optimizer=optimizer,
            sampler=sampler,
            optimizer_updates=2,
            global_batch=8,
            microbatch=2,
            cpu_threads=1,
        )
        assert metrics["optimizer_updates"] == 2
        assert metrics["transition_visits"] == 16
        assert metrics["target_hash_before"] == metrics["target_hash_after"]
        assert all(
            state["step"].item() == (4 if warm_optimizer else 2)
            for state in optimizer.state.values()
        )
        states.append(sampler.state_dict())
    assert states[0] == states[1]
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.integration
def test_attention_real_gpu_world_sizes(tmp_path):
    import json
    import os
    from time import perf_counter

    from attention_fixtures import rich_replay

    from latency_meta_mdp.meta.information.config import load_config

    selected = os.environ.get("META_TEST_GPUS", "").split(",")
    if len(selected) < 4 or any(not g.isdigit() for g in selected):
        pytest.skip("set META_TEST_GPUS to four exclusively available physical GPUs")
    config = load_config("configs/experiments/conveyor_sort/meta_attention_v2.yaml")
    torch.manual_seed(7)
    initial = AttentionStageQ(config)
    replay = rich_replay(config)
    from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration

    warm_optimizer = torch.optim.Adam(initial.parameters(), lr=1e-4)
    fit_attention_iteration(
        initial,
        replay,
        optimizer=warm_optimizer,
        sampler=DecisionSampler(replay, seed=7),
        optimizer_updates=2,
        global_batch=8,
        microbatch=2,
    )
    reference = None
    report = []
    for world in (1, 2, 4):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected[:world])
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(copy.deepcopy(warm_optimizer.state_dict()))
        sampler = DecisionSampler(replay, seed=42)
        started = perf_counter()
        metrics = fit_attention_distributed(
            model,
            replay,
            devices=[f"cuda:{i}" for i in range(world)],
            optimizer=optimizer,
            sampler=sampler,
            optimizer_updates=3,
            global_batch=8,
            microbatch=2,
            cpu_threads=2,
        )
        assert all(v["step"].item() == 5 for v in optimizer.state.values())
        if reference is None:
            reference = model
        else:
            for a, b in zip(reference.parameters(), model.parameters(), strict=True):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
        report.append(dict(world_size=world, wall_seconds=perf_counter() - started, **metrics))
    destination = Path(os.environ.get("META_TEST_REPORT", str(tmp_path / "gpu-parity.json")))
    destination.write_text(json.dumps(report, indent=2) + "\n")


def test_periodic_checkpoint_resumes_on_different_learner_world(tmp_path):
    from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration

    torch.manual_seed(4)
    initial = AttentionStageQ(small_config())
    replay = tiny_replay(initial.config)
    options = dict(optimizer_updates=4, global_batch=8, microbatch=2)
    checkpoint = dict(
        path=str(tmp_path / "latest.pt"), binding=dict(replay="fixture"), every_updates=2
    )

    def interrupt(event):
        if event["phase"] == "fit" and event["optimizer_update"] == 3:
            raise RuntimeError("time limit")

    reference = copy.deepcopy(initial)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4)
    fit_attention_iteration(
        reference,
        replay,
        optimizer=reference_optimizer,
        sampler=DecisionSampler(replay, seed=7),
        **options,
    )
    interrupted = copy.deepcopy(initial)
    with pytest.raises(RuntimeError, match="time limit"):
        fit_attention_iteration(
            interrupted,
            replay,
            optimizer=torch.optim.Adam(interrupted.parameters(), lr=1e-4),
            sampler=DecisionSampler(replay, seed=7),
            checkpoint=checkpoint,
            progress=interrupt,
            **options,
        )
    restored = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    metrics = fit_attention_distributed(
        restored,
        replay,
        devices=["cpu"] * 2,
        optimizer=optimizer,
        sampler=DecisionSampler(replay, seed=0),
        checkpoint=checkpoint,
        cpu_threads=1,
        **options,
    )
    assert metrics["resumed_updates"] == metrics["performed_updates"] == 2
    for a, b in zip(reference.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    assert all(v["step"].item() == 4 for v in optimizer.state.values())
    with pytest.raises(ValueError, match="binding"):
        fit_attention_iteration(
            restored,
            replay,
            optimizer=optimizer,
            sampler=DecisionSampler(replay, seed=0),
            checkpoint=dict(checkpoint, binding=dict(replay="different")),
            **options,
        )
