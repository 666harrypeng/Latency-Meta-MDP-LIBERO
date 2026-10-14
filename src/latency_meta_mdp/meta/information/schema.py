"""Public scheduler inputs and the finite information-stage action contract."""

from dataclasses import dataclass, field, fields

import numpy as np

SCHEMA_VERSION = 1
SLEEP_TICKS = (1, 2, 4, 8, 16)
OBSERVE, PREDICT, LAUNCH = 5, 6, 7
STAGES = ("G", "O", "P")


def legal_actions(stage, remaining, reserve=20, mode="bound", forecast_fresh=False):
    if stage not in STAGES or mode not in ("bound", "split"):
        raise ValueError("invalid information stage or mode")
    if any(type(x) is not int for x in (remaining, reserve)) or not 0 <= reserve <= 20:
        raise ValueError("invalid action coverage")
    if not reserve <= remaining <= 50:
        raise ValueError("insufficient action coverage")
    mask = np.zeros(8, bool)
    mask[:5] = [remaining >= reserve + d for d in SLEEP_TICKS]
    if stage == "G":
        mask[OBSERVE] = True
    elif stage == "O":
        mask[PREDICT] = True
        if mode == "bound":
            mask[:5] = False
    elif forecast_fresh:
        mask[LAUNCH] = True
    else:
        raise ValueError("prediction stage requires a fresh forecast")
    return mask


def _array(value, shape, dtype=np.float32):
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"public array must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def _zeros(shape):
    return field(default_factory=lambda: np.zeros(shape, np.float32))


@dataclass(frozen=True)
class ActorPacket:
    tick: int
    stage: str
    current_tokens: np.ndarray | None = None
    forecast_tokens: np.ndarray | None = None
    current_tick: int = -1
    forecast_tick: int = -1
    forecast_buffer_version: int = -1
    buffer_version: int = 0
    estimated_delay: int = 20
    plan_age: int = 0
    current_proprio: np.ndarray = _zeros(16)
    observed_proprio: np.ndarray = _zeros(16)
    forecast_proprio: np.ndarray = _zeros(16)
    buffer: np.ndarray = _zeros((50, 7))
    buffer_mask: np.ndarray = field(default_factory=lambda: np.ones(50, bool))
    completed_delays: tuple = ()
    previous_action: int = -1
    elapsed_ticks: int = 0

    def __post_init__(self):
        if type(self.tick) is not int or self.tick < 0 or self.stage not in STAGES:
            raise ValueError("invalid public clock/stage")
        for name in ("current", "forecast"):
            value, tick = getattr(self, f"{name}_tokens"), getattr(self, f"{name}_tick")
            if type(tick) is not int or tick > self.tick:
                raise ValueError("future observation is not public")
            if (value is None) != (tick == -1) or tick < -1:
                raise ValueError("visual source timestamp/availability mismatch")
            if value is not None:
                object.__setattr__(self, f"{name}_tokens", _array(value, (2, 196, 384)))
        for name in ("current_proprio", "observed_proprio", "forecast_proprio"):
            object.__setattr__(self, name, _array(getattr(self, name), (16,)))
        object.__setattr__(self, "buffer", _array(self.buffer, (50, 7)))
        if np.asarray(self.buffer_mask).dtype != np.bool_:
            raise ValueError("buffer validity must be boolean")
        mask = _array(self.buffer_mask, (50,), bool)
        if np.any(mask[1:] & ~mask[:-1]):
            raise ValueError("buffer validity must be a contiguous prefix")
        object.__setattr__(self, "buffer_mask", mask)
        delays = tuple(self.completed_delays)
        if len(delays) > 32 or any(type(d) is not int or not 0 <= d <= 20 for d in delays):
            raise ValueError("completed delay history is invalid")
        object.__setattr__(self, "completed_delays", delays)
        if type(self.estimated_delay) is not int or not 0 <= self.estimated_delay <= 20:
            raise ValueError("invalid forecast query")
        if (
            not -1 <= self.previous_action < self.num_actions
            or self.elapsed_ticks < 0
            or self.plan_age < 0
        ):
            raise ValueError("invalid scheduler history")

    @property
    def num_actions(self):
        return 8

    @property
    def forecast_fresh(self):
        return (
            self.forecast_tokens is not None
            and self.forecast_tick == self.tick
            and self.forecast_buffer_version == self.buffer_version
        )

    @property
    def remaining(self):
        return int(self.buffer_mask.sum())


ACTOR_FIELDS = frozenset(f.name for f in fields(ActorPacket))
