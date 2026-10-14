import numpy as np
import pytest

from latency_meta_mdp.meta.information.memory_schema import (
    AttentionPacket,
    DecisionRef,
    PublicEvent,
)


def observation(name, tick, ordinal, ticks):
    return PublicEvent(
        name,
        "observe",
        (tick, ordinal),
        tuple(ticks),
        dict(
            visual=np.stack([np.full((2, 196, 384), t, np.float32) for t in ticks]),
            proprio=np.stack([np.full(16, t, np.float32) for t in ticks]),
        ),
    )


def test_same_tick_future_events_cannot_enter_context_and_reset_removes_history():
    from latency_meta_mdp.meta.information.memory import MemoryStore, collate_contexts

    store = MemoryStore()
    store.reset("one")
    store.ingest(observation("obs", 10, 2, [2, 6, 10]))
    before = AttentionPacket(10, "G", decision_key=(10, 1))
    after = AttentionPacket(10, "P", decision_key=(10, 4))
    ctx = [store.context(DecisionRef("one", i), p) for i, p in enumerate([before, after])]
    batch = collate_contexts(ctx, device="cpu")
    assert not batch.tensors["visual_mask"][0].any()
    assert batch.tensors["visual_mask"][1, :3].tolist() == [True, True, True]
    assert batch.tensors["visual_ticks"][1, :3].tolist() == [2, 6, 10]
    store.reset("two")
    empty = store.context(DecisionRef("two", 0), after)
    assert all(ref is None for ref in empty.visual_refs)
    with pytest.raises(ValueError, match="episode"):
        store.context(DecisionRef("one", 0), after)


def test_overlap_deduplicates_time_identity_and_keeps_latest_bundle_dense():
    from latency_meta_mdp.meta.information.memory import MemoryStore

    store = MemoryStore()
    store.reset("one")
    store.ingest(observation("a", 10, 1, [2, 6, 10]))
    store.ingest(observation("b", 14, 2, [6, 10, 14]))
    context = store.context(DecisionRef("one", 0), AttentionPacket(14, "P", decision_key=(14, 3)))
    refs = [r for r in context.visual_refs if r is not None]
    ticks = [event.represented_ticks[i] for event, i in refs]
    assert ticks == [6, 10, 14, 2]
    assert refs[0][0].event_id == "a"  # earliest public acquisition retained
    assert len(set(ticks)) == 4


def test_streaming_and_reconstructed_history_have_identical_decision_inputs():
    import torch

    from latency_meta_mdp.meta.information.memory import MemoryStore, collate_contexts

    events = [observation("a", 10, 1, [2, 6, 10]), observation("b", 14, 3, [6, 10, 14])]
    live, replay = MemoryStore(), MemoryStore()
    live.reset("one")
    replay.reset("one")
    for event in events:
        replay.ingest(event)
    for i, event in enumerate(events):
        live.ingest(event)
        packet = AttentionPacket(
            event.available_at[0],
            "P",
            decision_key=(event.available_at[0], event.available_at[1] + 1),
        )
        ref = DecisionRef("one", i)
        a = collate_contexts([live.context(ref, packet)], device="cpu")
        b = collate_contexts([replay.context(ref, packet)], device="cpu")
        for name in a.tensors:
            torch.testing.assert_close(a.tensors[name], b.tensors[name])
