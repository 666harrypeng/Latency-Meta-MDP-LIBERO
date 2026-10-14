"""Causal, reconstructible history shared by rollout and replay."""

from bisect import bisect_left

import numpy as np
import torch

from latency_meta_mdp.meta.information.memory_schema import (
    ActionSpec,
    MemoryBatch,
    MemoryContext,
    PublicEvent,
)
from latency_meta_mdp.meta.information.schema import STAGES

DEFAULT_MEMORY = dict(
    visual_frames=16,
    recent_frames=3,
    execution_records=64,
    scheduler_events=32,
    spatial_tokens=16,
    lag_seconds=[0.25, 0.5, 1, 2, 4, 8, 16, 32],
    tick_seconds=0.02,
    time_scale=0.02,
    episode_ticks=2200,
)
EVENT_KINDS = (
    "observe",
    "predict",
    "execute",
    "request",
    "arrival",
    "decision",
    "hold_start",
    "hold_end",
)


class MemoryStore:
    def __init__(self, action_spec=None, *, config=None):
        self.action_spec = action_spec or ActionSpec()
        self.config = dict(DEFAULT_MEMORY if config is None else config)
        self.reset("")

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.events = []
        self._ids = set()
        self._frames = {}
        self._by_kind = {name: [] for name in ("observe", "predict", "execute", "schedule")}
        self._keys = {name: [] for name in self._by_kind}

    def ingest(self, event):
        if not isinstance(event, PublicEvent):
            raise TypeError("memory only accepts public events")
        if event.event_id in self._ids or (
            self.events and event.available_at <= self.events[-1].available_at
        ):
            raise ValueError("duplicate or out-of-order memory event")
        self.events.append(event)
        self._ids.add(event.event_id)
        kinds = [event.kind] if event.kind in self._by_kind else []
        if event.kind != "execute":
            kinds.append("schedule")
        for kind in kinds:
            self._by_kind[kind].append(event)
            self._keys[kind].append(event.available_at)
        if event.kind == "observe":
            for i, tick in enumerate(event.represented_ticks):
                self._frames.setdefault(tick, (event, i))

    def context(self, ref, state):
        if ref.episode_id != self.episode_id:
            raise ValueError("memory episode differs")
        key = state.decision_key

        def visible(kind, count):
            end = bisect_left(self._keys[kind], key)
            return self._by_kind[kind][max(0, end - count) : end]

        frames = {t: ref for t, ref in self._frames.items() if ref[0].available_at < key}
        observations = visible("observe", 1)
        latest = observations[-1].represented_ticks if observations else ()
        executions = visible("execute", self.config["execution_records"])
        schedule = visible("schedule", self.config["scheduler_events"])
        forecasts = visible("predict", 1)
        forecast = forecasts[-1] if forecasts else None
        dense = list(latest[-3:])
        selected = set(dense)
        refs = [frames[t] for t in dense] + [None] * (3 - len(dense))
        recent_count = self.config["visual_frames"] - 3 - len(self.config["lag_seconds"])
        recent = [t for t in sorted(frames, reverse=True) if t not in selected][:recent_count]
        selected.update(recent)
        refs += [frames[t] for t in recent] + [None] * (recent_count - len(recent))
        for lag in self.config["lag_seconds"]:
            before = key[0] - lag / self.config["tick_seconds"]
            candidates = [t for t in frames if t <= before and t not in selected]
            tick = max(candidates) if candidates else None
            refs.append(frames[tick] if tick is not None else None)
            if tick is not None:
                selected.add(tick)
        if forecast is not None and (
            forecast.available_at[0] != state.tick
            or forecast.payload["buffer_version"] != state.buffer_version
        ):
            forecast = None
        return MemoryContext(
            ref, key, state, tuple(refs), tuple(executions), tuple(schedule), forecast
        )


def critical_features(packet):
    delays = np.asarray(packet.completed_delays, dtype=np.float32)
    return np.concatenate(
        (
            packet.current_proprio,
            np.eye(3, dtype=np.float32)[STAGES.index(packet.stage)],
            np.array(
                [
                    packet.remaining / 50,
                    packet.estimated_delay / 20,
                    (packet.tick - packet.current_tick) / 2200 if packet.current_tick >= 0 else 0,
                    (packet.tick - packet.forecast_tick) / 2200 if packet.forecast_tick >= 0 else 0,
                    packet.tick / 2200,
                    packet.episode_remaining / 2200,
                    packet.plan_age / 50,
                    packet.current_tokens is not None,
                    packet.forecast_tokens is not None,
                    packet.forecast_fresh,
                    packet.action_spec.starvation_mode == "allow_hold",
                    packet.remaining == 0,
                    packet.elapsed_ticks / 50,
                    delays[-1] / 20 if len(delays) else 0,
                    delays.mean() / 20 if len(delays) else 0,
                    delays.std() / 20 if len(delays) else 0,
                    delays.min() / 20 if len(delays) else 0,
                    delays.max() / 20 if len(delays) else 0,
                    min(len(delays), 10) / 10,
                    packet.previous_action / max(1, packet.num_actions - 1),
                    len(delays) > 0,
                ],
                np.float32,
            ),
        )
    ).astype(np.float32)


def collate_contexts(contexts, source_store=None, *, device="cpu", config=None):
    """Pad raw public inputs; trainable features are exclusively model-owned."""
    if not contexts:
        raise ValueError("cannot collate empty decision batch")
    config = DEFAULT_MEMORY if config is None else config
    b, n, e, r = (
        len(contexts),
        config["visual_frames"],
        config["execution_records"],
        config["scheduler_events"],
    )
    shapes = dict(
        critical=(40,),
        tick=(),
        stage=(),
        buffer=(50, 7),
        buffer_mask=(50,),
        buffer_ticks=(50,),
        visual=(n, 2, 196, 384),
        visual_proprio=(n, 16),
        visual_mask=(n,),
        visual_ticks=(n,),
        execution_action=(e, 7),
        execution_proprio=(e, 16),
        execution_mask=(e,),
        execution_ticks=(e, 2),
        event_features=(r, 16),
        event_mask=(r,),
        event_ticks=(r,),
        forecast=(2, 196, 384),
        forecast_proprio=(16,),
        forecast_metadata=(8,),
        forecast_mask=(),
        forecast_tick=(),
    )
    arrays = {
        k: np.zeros(
            (b, *shape), bool if k.endswith("mask") else (np.int64 if k == "stage" else np.float32)
        )
        for k, shape in shapes.items()
    }
    for row, context in enumerate(contexts):
        packet = context.state
        arrays["critical"][row] = critical_features(packet)
        arrays["tick"][row], arrays["stage"][row] = packet.tick, STAGES.index(packet.stage)
        arrays["buffer"][row], arrays["buffer_mask"][row] = packet.buffer, packet.buffer_mask
        arrays["buffer_ticks"][row] = packet.tick + np.arange(50)
        for j, ref in enumerate(context.visual_refs):
            if ref is None:
                continue
            event, i = ref
            arrays["visual"][row, j] = event.payload["visual"][i]
            arrays["visual_proprio"][row, j] = event.payload["proprio"][i]
            arrays["visual_mask"][row, j] = True
            arrays["visual_ticks"][row, j] = event.represented_ticks[i]
        for j, event in enumerate(context.execution_refs):
            arrays["execution_action"][row, j] = event.payload["action"]
            arrays["execution_proprio"][row, j] = event.payload["proprio"]
            arrays["execution_mask"][row, j] = True
            arrays["execution_ticks"][row, j] = event.represented_ticks
        for j, event in enumerate(context.scheduler_refs):
            features = arrays["event_features"][row, j]
            features[EVENT_KINDS.index(event.kind)] = 1
            if event.kind in ("arrival", "request"):
                features[8] = (
                    event.payload.get("realized_delay", event.payload.get("estimated_delay")) / 20
                )
            if event.kind == "decision":
                features[9] = event.payload["action"] / max(1, packet.num_actions - 1)
                features[10 + STAGES.index(event.payload["stage"])] = 1
            arrays["event_mask"][row, j] = True
            arrays["event_ticks"][row, j] = event.available_at[0]
        forecast = context.forecast_ref
        if forecast is not None:
            arrays["forecast"][row] = forecast.payload["visual"]
            arrays["forecast_proprio"][row] = forecast.payload["proprio"]
            arrays["forecast_tick"][row] = forecast.represented_ticks[0]
            arrays["forecast_mask"][row] = True
            arrays["forecast_metadata"][row, :3] = [forecast.payload["query_ticks"] / 20, 1, 0]
    return MemoryBatch(
        {k: torch.as_tensor(v, device=device) for k, v in arrays.items()},
        tuple(c.decision for c in contexts),
    )


def build_context_index(events, decisions, spec):
    """Return causal source indices for immutable decision snapshots."""
    store = MemoryStore(config=spec)
    store.reset(decisions[0][0].episode_id if decisions else "")
    for event in events:
        store.ingest(event)
    indices = {id(event): i for i, event in enumerate(events)}
    result = []
    for ref, state in decisions:
        context = store.context(ref, state)
        result.append(
            dict(
                visual=[
                    None if x is None else (indices[id(x[0])], x[1]) for x in context.visual_refs
                ],
                execution=[indices[id(x)] for x in context.execution_refs],
                scheduler=[indices[id(x)] for x in context.scheduler_refs],
                forecast=None
                if context.forecast_ref is None
                else indices[id(context.forecast_ref)],
            )
        )
    return result
