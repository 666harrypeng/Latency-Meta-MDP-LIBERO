"""Compiled execution retains FQI sampling, targets and resumable state."""

import copy

import pytest
import torch
from attention_fixtures import rich_replay, small_config

from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration
from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.attention_replay import DecisionSampler


@pytest.fixture(autouse=True)
def learner_kernels():
    """Match distributed_fqi's kernel policy, including single-GPU execution."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def test_compiled_fqi_requires_cuda():
    config = small_config()
    model = AttentionStageQ(config)
    replay = rich_replay(config)
    with pytest.raises(ValueError, match="CUDA"):
        fit_attention_iteration(
            model,
            replay,
            optimizer=torch.optim.Adam(model.parameters()),
            sampler=DecisionSampler(replay, seed=17),
            optimizer_updates=2,
            global_batch=4,
            microbatch=2,
            compile_model=True,
        )


def test_execution_options_do_not_rebind_scientific_checkpoint(tmp_path):
    from latency_meta_mdp.meta.information.training import train_iterations

    config = small_config()
    replay = rich_replay(config)
    model = AttentionStageQ(config)
    requested = []

    def cpu_fitter(*args, compile_model=False, **kwargs):
        requested.append(compile_model)
        return fit_attention_iteration(*args, **kwargs)

    for compiled, micro, stop in ((False, 2, 1), (True, "full", None)):
        train_iterations(
            model,
            replay,
            tmp_path,
            binding={},
            total_iterations=2,
            epochs=1,
            learning_rate=1e-4,
            conservative_weight=0.01,
            fitter=cpu_fitter,
            attention_options=dict(
                global_batch=4, microbatch=micro, optimizer_updates=2, compile_model=compiled
            ),
            stop_after_iterations=stop,
        )
    assert requested == [False, True]
    state = torch.load(tmp_path / "latest/state.pt", weights_only=True)
    assert state["sampler"]["visits"] == 16


@pytest.mark.parametrize("bf16", [False, True])
def test_compiled_fqi_accumulation_tail_and_checkpoint_resume(tmp_path, bf16):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Inductor/CUDA Graph execution")
    config = small_config()
    replay = rich_replay(config)
    torch.manual_seed(71)
    initial = AttentionStageQ(config).cuda()
    for head in initial.heads:
        torch.nn.init.normal_(head.weight, std=0.01)
    results = []
    for compiled, interrupt in ((False, False), (True, False), (True, True)):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sampler = DecisionSampler(replay, seed=17)
        checkpoint = dict(
            path=str(tmp_path / f"state-{len(results)}.pt"), binding={}, every_updates=2
        )

        def fit(callback=None):
            return fit_attention_iteration(
                model,
                replay,
                optimizer=optimizer,
                sampler=sampler,
                optimizer_updates=4,
                global_batch=4,
                microbatch=2,
                compile_model=compiled,
                bf16=bf16,
                checkpoint=checkpoint,
                progress=callback,
            )

        def pause(event):
            if event["phase"] == "fit" and event["optimizer_update"] == 2:
                raise InterruptedError("after a complete optimizer update")

        if interrupt:
            with pytest.raises(InterruptedError):
                fit(pause)
            model = copy.deepcopy(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            sampler = DecisionSampler(replay, seed=999)
        metrics = fit()
        assert metrics["target_hash_before"] == metrics["target_hash_after"]
        assert metrics["unique_decisions"] == 3  # Includes a short target microbatch.
        assert metrics["resumed_updates"] == (2 if interrupt else 0)
        assert metrics["compile_model"] == compiled
        results.append(
            (
                copy.deepcopy(model.state_dict()),
                optimizer.state_dict(),
                sampler.state_dict(),
                metrics,
            )
        )
    for state, optimizer, sampler, metrics in results[1:]:
        assert state.keys() == results[0][0].keys()  # No _orig_mod checkpoint prefix.
        if bf16:
            # BF16 fused reductions can change the sign of near-zero gradients;
            # Adam amplifies those coordinates. Bound aggregate drift rather
            # than claiming bitwise equivalence of the two kernel paths.
            delta = torch.cat([(state[k] - v).flatten() for k, v in results[0][0].items()])
            assert delta.square().mean().sqrt() < 5e-5
        else:
            for key, expected in results[0][0].items():
                torch.testing.assert_close(state[key], expected, atol=2e-6, rtol=1e-4)
        assert sampler == results[0][2]
        assert metrics["sampling"] == results[0][3]["sampling"]
        assert metrics["loss"] == pytest.approx(
            results[0][3]["loss"], rel=0.01 if bf16 else 1e-4, abs=1e-5
        )
        if bf16:
            for name in ("exp_avg", "exp_avg_sq"):
                expected = torch.cat([v[name].flatten() for v in results[0][1]["state"].values()])
                actual = torch.cat([v[name].flatten() for v in optimizer["state"].values()])
                assert (actual - expected).norm() / expected.norm() < 0.02
        for key, values in results[0][1]["state"].items():
            for name, expected in values.items():
                if not bf16 or name == "step":
                    torch.testing.assert_close(
                        optimizer["state"][key][name], expected, atol=2e-6, rtol=1e-4
                    )
    for key, expected in results[1][0].items():
        torch.testing.assert_close(results[2][0][key], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("protocol", [2, 3])
def test_compiled_forward_and_gradient_match_before_adam(bf16, protocol):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from attention_fixtures import make_batch

    from latency_meta_mdp.meta.information.actor_inputs import transfer_batch

    batch, config = make_batch(batch_size=3 if protocol == 3 else 2)
    if protocol == 3:
        from test_optional_prediction import optional_config

        config = optional_config()
        batch.tensors["stage"][:] = torch.tensor([0, 1, 2])
    batch = transfer_batch(batch, "cuda")
    torch.manual_seed(71)
    original = AttentionStageQ(config).cuda()
    for head in original.heads:
        torch.nn.init.normal_(head.weight, std=0.01)
    results = []
    for compiled in (False, True):
        model = copy.deepcopy(original)
        forward = (
            torch.compile(
                model,
                dynamic=False,
                options={"triton.cudagraphs": True, "emulate_precision_casts": True},
            )
            if compiled
            else model
        )
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            values = forward(batch)
            loss = values.float().square().mean()
        loss.backward()
        gradient = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        results.append((values.detach().clone(), gradient.clone()))
    if bf16:
        # Relative error of a single near-zero Q value is ill-conditioned.
        # Bound vector error and max error relative to the output scale instead.
        expected, actual = results[0][0].float(), results[1][0].float()
        assert (actual - expected).norm() / expected.norm() < 0.015
        assert (actual - expected).abs().max() / expected.abs().max() < 0.015
    else:
        torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-4, atol=2e-6)
    first, second = results[0][1], results[1][1]
    assert torch.nn.functional.cosine_similarity(first, second, dim=0) > (
        0.999 if bf16 else 0.99999
    )
    assert (first - second).norm() / first.norm() < (0.015 if bf16 else 1e-4)


def test_compiled_forward_preserves_stage_validation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from attention_fixtures import make_batch

    from latency_meta_mdp.meta.information.actor_inputs import transfer_batch

    batch, config = make_batch()
    batch = transfer_batch(batch, "cuda")
    model = AttentionStageQ(config).cuda()
    forward = torch.compile(model, mode="reduce-overhead", dynamic=False)
    batch.tensors["stage"].fill_(1)
    with pytest.raises(ValueError, match="forced"):
        forward(batch)
