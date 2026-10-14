import copy

import numpy as np

from latency_meta_mdp.meta.information.config import load_config
from latency_meta_mdp.meta.information.memory import MemoryStore, collate_contexts
from latency_meta_mdp.meta.information.memory_schema import (
    AttentionPacket,
    DecisionRef,
    PublicEvent,
)


def small_config():
    config = copy.deepcopy(load_config("configs/experiments/conveyor_sort/meta_attention_v2.yaml"))
    config["model"].update(width=32, ffn_width=64)
    config["memory"].update(
        visual_frames=3, lag_seconds=[], execution_records=2, scheduler_events=4
    )
    return config


def make_batch(batch_size=2, missing_visual=False, missing_forecast=False):
    config = small_config()
    rng = np.random.default_rng(4)
    contexts = []
    for i in range(batch_size):
        store = MemoryStore(config=config["memory"])
        store.reset(str(i))
        visual = rng.normal(size=(3, 2, 196, 384)).astype(np.float32)
        if not missing_visual:
            store.ingest(
                PublicEvent(
                    "obs",
                    "observe",
                    (10, 1),
                    (2, 6, 10),
                    dict(visual=visual, proprio=np.zeros((3, 16))),
                )
            )
        if not missing_forecast:
            store.ingest(
                PublicEvent(
                    "pred",
                    "predict",
                    (10, 2),
                    (14,),
                    dict(
                        visual=visual[-1],
                        proprio=np.zeros(16),
                        source_tick=10,
                        query_ticks=4,
                        buffer_version=0,
                        prefix_hash="fixture",
                    ),
                )
            )
        packet = AttentionPacket(
            10,
            "G" if missing_forecast else "P",
            decision_key=(10, 3),
            current_proprio=rng.normal(size=16).astype(np.float32),
        )
        contexts.append(store.context(DecisionRef(str(i), 0), packet))
    return collate_contexts(contexts, config=config["memory"]), config


def tiny_replay(config):
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay

    store = MemoryStore(config=config["memory"])
    store.reset("tiny")
    context = store.context(DecisionRef("tiny", 0), AttentionPacket(10, "G", decision_key=(10, 1)))
    return DecisionReplay(
        [
            [
                dict(
                    context=context,
                    next_context=None,
                    action=8,
                    task_reward=0.1,
                    terminated=True,
                    legal=np.array([True] * 9 + [False, False]),
                    next_legal=np.zeros(11, bool),
                )
            ]
        ]
    )


def rich_replay(config):
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay

    rng = np.random.default_rng(117)
    store = MemoryStore(config=config["memory"])
    store.reset("rich")
    events = []
    for t in range(20, 101, 20):
        events.append(
            PublicEvent(
                f"obs-{t}",
                "observe",
                (t, 1),
                (t - 8, t - 4, t),
                dict(
                    visual=rng.normal(size=(3, 2, 196, 384)).astype(np.float32),
                    proprio=rng.normal(size=(3, 16)).astype(np.float32),
                ),
            )
        )
    for t in range(36, 100):
        events.append(
            PublicEvent(
                f"exec-{t}",
                "execute",
                (t + 1, 0),
                (t, t + 1),
                dict(
                    action=rng.uniform(-1, 1, 7).astype(np.float32),
                    proprio=rng.normal(size=16).astype(np.float32),
                ),
            )
        )
    events.append(
        PublicEvent(
            "predict",
            "predict",
            (100, 2),
            (120,),
            dict(
                visual=rng.normal(size=(2, 196, 384)).astype(np.float32),
                proprio=np.zeros(16),
                source_tick=100,
                query_ticks=20,
                buffer_version=0,
                prefix_hash="fixture",
            ),
        )
    )
    for event in sorted(events, key=lambda e: e.available_at):
        store.ingest(event)
    contexts = []
    for i in range(3):
        packet = AttentionPacket(
            100 + i,
            "P" if i == 0 else "G",
            decision_key=(100 + i, 3),
            current_proprio=rng.normal(size=16).astype(np.float32),
        )
        contexts.append(store.context(DecisionRef("rich", i), packet))
    rows = []
    for i, context in enumerate(contexts):
        legal = np.zeros(11, bool)
        legal[:8] = True
        legal[10 if i == 0 else 8] = True
        following = np.zeros(11, bool)
        following[:9] = i < 2
        rows.append(
            dict(
                context=context,
                next_context=contexts[i + 1] if i < 2 else None,
                action=0 if i == 0 else 8,
                task_reward=0.1 if i == 2 else 0.0,
                terminated=i == 2,
                legal=legal,
                next_legal=following,
            )
        )
    return DecisionReplay([rows])
