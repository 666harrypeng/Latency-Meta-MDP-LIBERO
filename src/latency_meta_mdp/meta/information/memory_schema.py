"""Versioned public events, actions and batch contracts for attention Meta."""

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from latency_meta_mdp.meta.information.schema import SLEEP_TICKS, STAGES, ActorPacket, _array

EventKey = tuple[int, int]


@dataclass(frozen=True)
class ActionSpec:
    version: int = 2
    sleep_ticks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    starvation_mode: str = "guarded"
    reserve: int = 20

    def __post_init__(self):
        ticks = tuple(self.sleep_ticks)
        if (
            self.version not in (2, 3)
            or not ticks
            or any(type(t) is not int or t <= 0 for t in ticks)
            or tuple(sorted(set(ticks))) != ticks
        ):
            raise ValueError("invalid action vocabulary")
        if self.starvation_mode not in ("guarded", "allow_hold"):
            raise ValueError("unknown starvation protocol")
        if type(self.reserve) is not int or not 0 <= self.reserve <= 20:
            raise ValueError("invalid action reserve")
        if self.version == 3 and (self.starvation_mode != "allow_hold" or self.reserve != 0):
            raise ValueError("optional prediction requires voluntary hold without a reserve")
        object.__setattr__(self, "sleep_ticks", ticks)

    @property
    def names(self):
        return tuple(f"sleep:{t}" for t in self.sleep_ticks) + ("observe", "predict", "launch")

    @property
    def num_actions(self):
        return len(self.names)

    def id_for(self, name):
        return self.names.index(name)

    def name_for(self, action):
        if type(action) is not int or not 0 <= action < self.num_actions:
            raise ValueError("invalid action ID")
        return self.names[action]

    def from_v1(self, action):
        if type(action) is not int or not 0 <= action < 8:
            raise ValueError("invalid v1 action")
        name = (
            f"sleep:{SLEEP_TICKS[action]}"
            if action < 5
            else ("observe", "predict", "launch")[action - 5]
        )
        return self.id_for(name)

    def legal(self, stage, remaining, *, forecast_fresh=False, prediction_ready=False):
        minimum = self.reserve if self.starvation_mode == "guarded" else 0
        if stage not in STAGES or type(remaining) is not int or not minimum <= remaining <= 50:
            raise ValueError("invalid stage or insufficient action coverage")
        mask = np.zeros(self.num_actions, bool)
        if stage != "O" or self.version == 3:
            mask[: len(self.sleep_ticks)] = [
                self.starvation_mode == "allow_hold" or remaining >= self.reserve + t
                for t in self.sleep_ticks
            ]
        if stage == "P" and not forecast_fresh:
            raise ValueError("prediction stage requires a fresh forecast")
        if stage == "O" and self.version == 3:
            mask[self.id_for("predict")] = prediction_ready
            mask[self.id_for("launch")] = True
        else:
            mask[self.id_for(dict(G="observe", O="predict", P="launch")[stage])] = True
        return mask


def validate_batch(*, global_batch, world_size, microbatch, accumulation):
    if any(
        type(x) is not int or x <= 0 for x in (global_batch, world_size, microbatch, accumulation)
    ):
        raise ValueError("batch dimensions must be positive integers")
    if global_batch % world_size or global_batch != world_size * microbatch * accumulation:
        raise ValueError("global batch must equal world size * microbatch * accumulation")


_EVENT_FIELDS = {
    "observe": {"visual", "proprio"},
    "predict": {"visual", "proprio", "source_tick", "query_ticks", "buffer_version", "prefix_hash"},
    "execute": {"action", "proprio"},
    "request": {"request_id", "estimated_delay", "buffer_version"},
    "arrival": {"request_id", "realized_delay", "buffer_version"},
    "decision": {"action", "probability", "stage"},
    "hold_start": set(),
    "hold_end": set(),
}


def event_key(value):
    if len(value) != 2 or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("invalid causal event key")
    return tuple(value)


@dataclass(frozen=True)
class AttentionPacket(ActorPacket):
    action_spec: ActionSpec = field(default_factory=ActionSpec)
    decision_key: EventKey = (0, 0)
    episode_remaining: int = 2200

    def __post_init__(self):
        super().__post_init__()
        if event_key(self.decision_key)[0] != self.tick or self.episode_remaining < 0:
            raise ValueError("invalid decision clock/remaining episode time")

    @property
    def num_actions(self):
        return self.action_spec.num_actions


@dataclass(frozen=True)
class OptionalPredictionPacket(AttentionPacket):
    """V3 publishes predictor input readiness without changing historical packet fields."""

    prediction_ready: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.action_spec.version != 3 or type(self.prediction_ready) is not bool:
            raise ValueError("invalid optional prediction packet")


@dataclass(frozen=True)
class PublicEvent:
    event_id: str
    kind: str
    available_at: EventKey
    represented_ticks: tuple[int, ...]
    payload: dict

    def __post_init__(self):
        key = event_key(self.available_at)
        ticks = tuple(self.represented_ticks)
        if (
            not isinstance(self.event_id, str)
            or not self.event_id
            or self.kind not in _EVENT_FIELDS
        ):
            raise ValueError("invalid public event identity")
        if set(self.payload) != _EVENT_FIELDS[self.kind]:
            raise ValueError("unknown or missing public event fields")
        if not ticks or any(type(t) is not int or t < 0 for t in ticks):
            raise ValueError("invalid represented ticks")
        if self.kind != "predict" and max(ticks) > key[0]:
            raise ValueError("future actual observation or execution is not public")
        payload = dict(self.payload)
        if self.kind == "observe":
            if tuple(sorted(set(ticks))) != ticks:
                raise ValueError("observe source ticks must be ordered and unique")
            payload["visual"] = _array(payload["visual"], (len(ticks), 2, 196, 384))
            payload["proprio"] = _array(payload["proprio"], (len(ticks), 16))
        elif self.kind == "predict":
            payload["visual"] = _array(payload["visual"], (2, 196, 384))
            payload["proprio"] = _array(payload["proprio"], (16,))
            source, query = payload["source_tick"], payload["query_ticks"]
            if (
                type(source) is not int
                or not 0 <= source <= key[0]
                or type(query) is not int
                or not 0 <= query <= 20
                or ticks != (source + query,)
                or not isinstance(payload["prefix_hash"], str)
            ):
                raise ValueError("invalid forecast source/target time")
        elif self.kind == "execute":
            if len(ticks) != 2 or ticks[1] != ticks[0] + 1:
                raise ValueError("execution must identify action and post-state ticks")
            payload["action"] = _array(payload["action"], (7,))
            payload["proprio"] = _array(payload["proprio"], (16,))
        elif self.kind in ("request", "arrival"):
            name = "estimated_delay" if self.kind == "request" else "realized_delay"
            if type(payload[name]) is not int or not 0 <= payload[name] <= 20:
                raise ValueError("invalid public delay")
        elif self.kind == "decision":
            if (
                type(payload["action"]) is not int
                or payload["action"] < 0
                or payload["stage"] not in STAGES
                or not 0 < payload["probability"] <= 1
            ):
                raise ValueError("invalid public decision")
        object.__setattr__(self, "available_at", key)
        object.__setattr__(self, "represented_ticks", ticks)
        object.__setattr__(self, "payload", MappingProxyType(payload))

    def __reduce__(self):
        return type(self), (
            self.event_id,
            self.kind,
            self.available_at,
            self.represented_ticks,
            dict(self.payload),
        )


@dataclass(frozen=True)
class DecisionRef:
    episode_id: str
    event_index: int


@dataclass(frozen=True)
class MemoryContext:
    decision: DecisionRef
    key: EventKey
    state: object
    visual_refs: tuple
    execution_refs: tuple
    scheduler_refs: tuple
    forecast_ref: PublicEvent | None


@dataclass(frozen=True)
class MemoryBatch:
    tensors: dict
    refs: tuple[DecisionRef, ...]
