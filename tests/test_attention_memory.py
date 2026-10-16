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


def test_lag_lookup_matches_reference_with_late_frames_and_historical_queries():
    from latency_meta_mdp.meta.information.memory import MemoryStore

    store = MemoryStore()
    store.reset("one")
    events = [
        observation(str(i), 20 * i + 10, 1, [20 * i, 20 * i + 5, 20 * i + 10]) for i in range(24)
    ]
    # Older represented time can first become visible in a later observation.
    events.append(observation("late", 480, 1, [3, 479, 480]))
    for event in events:
        store.ingest(event)
    for tick, ordinal in [(480, 2), (10, 1), (10, 2), (100, 0), (210, 2), (480, 1)]:
        packet = AttentionPacket(tick, "G", decision_key=(tick, ordinal))
        context = store.context(DecisionRef("one", 0), packet)
        visible = [e for e in events if e.available_at < packet.decision_key]
        frames = {}
        for event in visible:
            for i, source_tick in enumerate(event.represented_ticks):
                frames.setdefault(source_tick, (event, i))
        dense = list(visible[-1].represented_ticks[-3:]) if visible else []
        selected = set(dense)
        expected = [frames[t] for t in dense] + [None] * (3 - len(dense))
        recent = [t for t in sorted(frames, reverse=True) if t not in selected][:5]
        selected.update(recent)
        expected += [frames[t] for t in recent] + [None] * (5 - len(recent))
        for lag in store.config["lag_seconds"]:
            candidates = [t for t in frames if t <= tick - lag / 0.02 and t not in selected]
            chosen = max(candidates) if candidates else None
            expected.append(frames[chosen] if chosen is not None else None)
            selected.add(chosen)

        def identity(refs):
            return [None if r is None else (r[0].event_id, r[1]) for r in refs]

        assert identity(context.visual_refs) == identity(expected)
