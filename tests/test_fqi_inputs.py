"""FQI transport keeps the exact causal inputs and sample stream."""

import numpy as np
import pytest
import torch
from attention_fixtures import rich_replay, small_config

from latency_meta_mdp.meta.information.memory import collate_contexts


def test_prefetched_fqi_inputs_equal_dense_for_targets_and_regression():
    from latency_meta_mdp.meta.information.fqi_inputs import FQIInputs

    config = small_config()
    replay = rich_replay(config)
    order = np.array([2, 0, 0, 1, 2])
    for following in (False, True):
        with FQIInputs(replay, config["memory"], microbatch=2, device="cpu") as inputs:
            batches = list(inputs.batches(order, following=following))
        assert [i for ids, _, _ in batches for i in ids] == order.tolist()
        for ids, rows, batch in batches:
            expected = collate_contexts(
                [
                    r["next_context"]
                    if following and r["next_context"] is not None
                    else r["context"]
                    for r in rows
                ],
                config=config["memory"],
            )
            assert batch.refs == expected.refs
            for key in expected.tensors:
                torch.testing.assert_close(
                    batch.tensors[key], expected.tensors[key], rtol=0, atol=0
                )


@pytest.fixture
def deterministic_kernels():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bf16", [False, True])
def test_fqi_prefetch_matches_dense_optimizer_and_checkpoint_resume(
    tmp_path, monkeypatch, device, bf16, deterministic_kernels
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import copy
    from contextlib import nullcontext

    from latency_meta_mdp.meta.information import attention_fqi as fqi
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.attention_replay import DecisionSampler
    from latency_meta_mdp.meta.information.fqi_inputs import FQIInputs

    config = small_config()
    replay = rich_replay(config)
    torch.manual_seed(19)
    initial = AttentionStageQ(config).to(device)

    class DenseInputs:
        def __init__(self, replay, config, *, microbatch, device):
            self.replay, self.config = replay, config
            self.microbatch, self.device = microbatch, device

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def batches(self, order, *, following):
            for start in range(0, len(order), self.microbatch):
                ids = order[start : start + self.microbatch]
                rows = [self.replay[int(i)] for i in ids]
                contexts = [
                    r["next_context"]
                    if following and r["next_context"] is not None
                    else r["context"]
                    for r in rows
                ]
                yield ids, rows, collate_contexts(contexts, config=self.config, device=self.device)

    results = []
    for pipeline, interrupt in ((DenseInputs, False), (FQIInputs, False), (FQIInputs, True)):
        monkeypatch.setattr(fqi, "FQIInputs", pipeline)
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sampler = DecisionSampler(replay, seed=17)
        checkpoint = dict(path=str(tmp_path / f"{len(results)}.pt"), binding={}, every_updates=2)

        def progress(event):
            if interrupt and event["phase"] == "fit" and event["optimizer_update"] == 2:
                raise InterruptedError("test interruption after saved update")

        with pytest.raises(InterruptedError) if interrupt else nullcontext():
            metrics = fqi.fit_attention_iteration(
                model,
                replay,
                optimizer=optimizer,
                sampler=sampler,
                optimizer_updates=4,
                global_batch=4,
                microbatch=2,
                checkpoint=checkpoint,
                progress=progress,
                bf16=bf16,
            )
        if interrupt:
            # Recreate the process state; the periodic checkpoint must restore it all.
            model = copy.deepcopy(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            sampler = DecisionSampler(replay, seed=999)
            metrics = fqi.fit_attention_iteration(
                model,
                replay,
                optimizer=optimizer,
                sampler=sampler,
                optimizer_updates=4,
                global_batch=4,
                microbatch=2,
                checkpoint=checkpoint,
                bf16=bf16,
            )
            assert metrics["resumed_updates"] == 2
        assert metrics["target_hash_before"] == metrics["target_hash_after"]
        results.append((model, optimizer, sampler, metrics))
    for model, optimizer, sampler, metrics in results[1:]:
        for key, tensor in results[0][0].state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
        for key, state in results[0][1].state_dict()["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(
                    optimizer.state_dict()["state"][key][field], value, rtol=0, atol=0
                )
        assert sampler.state_dict() == results[0][2].state_dict()
        assert metrics["diagnostics"] == results[0][3]["diagnostics"]


def test_cpu_prefetch_overlaps_collation_and_preserves_order(monkeypatch):
    from threading import Barrier

    from latency_meta_mdp.meta.information import fqi_inputs

    config = small_config()
    replay = rich_replay(config)
    barrier = Barrier(2)
    original = fqi_inputs.collate_contexts
    calls = []

    def collate(*args, **kwargs):
        calls.append(True)
        if len(calls) <= 2:
            barrier.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(fqi_inputs, "collate_contexts", collate)
    with fqi_inputs.FQIInputs(replay, config["memory"], microbatch=1, device="cpu") as inputs:
        batches = list(inputs.batches(np.array([2, 0, 1]), following=False))
    assert [int(ids[0]) for ids, _, _ in batches] == [2, 0, 1]


def test_direct_pinned_collation_preserves_values_and_storage_dtype():
    if not torch.cuda.is_available():
        pytest.skip("Pinned host allocation requires a CUDA runtime")
    config = small_config()
    replay = rich_replay(config)
    contexts = [replay[i]["context"] for i in (2, 0, 2)]
    expected = collate_contexts(contexts, config=config["memory"], compact_visual=True)
    actual = collate_contexts(
        contexts, config=config["memory"], compact_visual=True, pin_memory=True
    )
    assert actual.refs == expected.refs
    for key in expected.tensors:
        assert actual.tensors[key].is_pinned()
        torch.testing.assert_close(actual.tensors[key], expected.tensors[key], rtol=0, atol=0)


@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_pinned_copy_preserves_readonly_strided_mmap_frames(tmp_path, mixed_dtype):
    from types import MappingProxyType

    from latency_meta_mdp.meta.information.actor_inputs import transfer_batch

    if not torch.cuda.is_available():
        pytest.skip("Pinned host allocation requires a CUDA runtime")
    config = small_config()
    replay = rich_replay(config)
    contexts = [replay[i]["context"] for i in (2, 0, 2)]
    events = {}
    for context in contexts:
        for ref in context.visual_refs:
            if ref is not None:
                events[id(ref[0])] = ref[0]
        if context.forecast_ref is not None:
            events[id(context.forecast_ref)] = context.forecast_ref
    original = []
    for i, event in enumerate(events.values()):
        dtype = np.float32 if mixed_dtype and i % 2 else np.float16
        value = event.payload["visual"].astype(dtype)
        path = tmp_path / f"frames-{i}.npy"
        np.save(path, np.repeat(value, 2, axis=-1))
        frame = np.load(path, mmap_mode="r")[..., ::2]
        object.__setattr__(event, "payload", MappingProxyType({**event.payload, "visual": frame}))
        original.append((frame, value.copy()))
    expected = collate_contexts(contexts, config=config["memory"])
    compact = collate_contexts(
        contexts, config=config["memory"], compact_visual=True, pin_memory=True
    )
    assert compact.tensors["visual_pool"].is_pinned()
    assert compact.tensors["visual_pool"].dtype == (torch.float32 if mixed_dtype else torch.float16)
    actual = transfer_batch(compact, "cpu")
    for key, value in expected.tensors.items():
        torch.testing.assert_close(actual.tensors[key], value, rtol=0, atol=0)
    for frame, value in original:
        assert not frame.flags.writeable
        np.testing.assert_array_equal(frame, value)


def test_cuda_prefetch_honors_consumer_stream_and_keeps_tail():
    from latency_meta_mdp.meta.information.fqi_inputs import FQIInputs

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = small_config()
    replay = rich_replay(config)
    order = np.array([2, 0, 0, 1, 2])
    stream = torch.cuda.Stream()
    received = []
    with torch.cuda.stream(stream):
        with FQIInputs(replay, config["memory"], microbatch=2, device="cuda") as inputs:
            for ids, rows, batch in inputs.batches(order, following=False):
                received.append((ids, rows, {k: t.clone() for k, t in batch.tensors.items()}))
    stream.synchronize()
    assert [int(i) for ids, _, _ in received for i in ids] == order.tolist()
    for _, rows, tensors in received:
        expected = collate_contexts([r["context"] for r in rows], config=config["memory"])
        for key, value in expected.tensors.items():
            torch.testing.assert_close(tensors[key].cpu(), value, rtol=0, atol=0)
