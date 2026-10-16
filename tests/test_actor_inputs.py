"""Optimized transport must preserve every public field and causal mask."""

import numpy as np
import pytest
import torch
from attention_fixtures import rich_replay, small_config

from latency_meta_mdp.meta.information.memory import collate_contexts


def test_compact_visual_transport_preserves_values_masks_and_duplicate_decisions():
    from latency_meta_mdp.meta.information.actor_inputs import transfer_batch

    config = small_config()
    replay = rich_replay(config)
    contexts = [replay[i]["context"] for i in [0, 1, 0, 2]]
    dense = collate_contexts(contexts, config=config["memory"])
    compact = collate_contexts(contexts, config=config["memory"], compact_visual=True)
    assert compact.tensors["visual_pool"].shape[0] == 4  # three observed frames + forecast
    restored = transfer_batch(compact, "cpu")
    assert restored.refs == dense.refs
    assert set(restored.tensors) == set(dense.tensors)
    for name in dense.tensors:
        torch.testing.assert_close(restored.tensors[name], dense.tensors[name], rtol=0, atol=0)


def test_compact_missing_history_has_zero_inputs_without_cross_episode_alias():
    from latency_meta_mdp.meta.information.actor_inputs import transfer_batch
    from latency_meta_mdp.meta.information.memory import MemoryStore
    from latency_meta_mdp.meta.information.memory_schema import AttentionPacket, DecisionRef

    config = small_config()
    a, b = rich_replay(config), rich_replay(config)
    event = b[0]["context"].visual_refs[0][0]
    # Matching event names and timestamps are not global frame identities.
    event.payload["visual"].setflags(write=True)
    event.payload["visual"][:] = 7
    store = MemoryStore(config=config["memory"])
    store.reset("empty")
    empty = store.context(DecisionRef("empty", 0), AttentionPacket(0, "G"))
    contexts = [a[0]["context"], b[0]["context"], empty]
    compact = collate_contexts(contexts, config=config["memory"], compact_visual=True)
    restored = transfer_batch(compact, "cpu")
    assert not restored.tensors["visual_mask"][2].any()
    assert torch.count_nonzero(restored.tensors["visual"][2]) == 0
    assert not torch.equal(restored.tensors["visual"][0], restored.tensors["visual"][1])
    dense = collate_contexts(contexts, config=config["memory"])
    for name in dense.tensors:
        torch.testing.assert_close(restored.tensors[name], dense.tensors[name], rtol=0, atol=0)


def test_prefetch_keeps_sample_order_successors_and_tail():
    from latency_meta_mdp.meta.information.actor_inputs import InputPipeline

    config = small_config()
    replay = rich_replay(config)
    with InputPipeline(replay, config["memory"], microbatch=2, prefetch=True) as pipeline:
        batches = list(pipeline.batches(np.array([2, 0, 1]), successors=True))
    assert [x.indices.tolist() for x in batches] == [[2, 0], [1]]
    assert [x.future for x in batches] == [[1], [0]]
    assert [x.context.refs for x in batches] == [
        (replay[2]["context"].decision, replay[0]["context"].decision),
        (replay[1]["context"].decision,),
    ]
    assert batches[0].following.refs == (replay[0]["next_context"].decision,)
    assert [r["action"] for x in batches for r in x.rows] == [8, 0, 8]


def test_prefetch_propagates_missing_successor_and_invalid_action():
    from latency_meta_mdp.meta.information.actor_inputs import InputPipeline

    config = small_config()
    replay = rich_replay(config)
    replay[0]["next_context"] = None
    with InputPipeline(replay, config["memory"], microbatch=1, prefetch=True) as pipeline:
        with pytest.raises(ValueError, match="successor"):
            list(pipeline.batches(np.array([0]), successors=True))
    replay[0]["action"] = 10
    replay[0]["legal"][10] = False
    with InputPipeline(replay, config["memory"], microbatch=1, prefetch=False) as pipeline:
        with pytest.raises(ValueError, match="illegal"):
            list(pipeline.batches(np.array([0]), successors=False))


def test_half_storage_is_restored_to_exact_float32_before_model_input():
    from latency_meta_mdp.meta.information.actor_inputs import transfer_batch
    from latency_meta_mdp.meta.information.memory_schema import MemoryBatch

    config = small_config()
    contexts = [rich_replay(config)[0]["context"]]
    compact = collate_contexts(contexts, config=config["memory"], compact_visual=True)
    compact.tensors["visual_pool"] = compact.tensors["visual_pool"].half().float()
    expected = transfer_batch(compact, "cpu")
    half = MemoryBatch(
        dict(compact.tensors, visual_pool=compact.tensors["visual_pool"].half()), compact.refs
    )
    result = transfer_batch(half, "cpu")
    for key in ("visual", "forecast"):
        assert result.tensors[key].dtype == torch.float32
        torch.testing.assert_close(result.tensors[key], expected.tensors[key], rtol=0, atol=0)
