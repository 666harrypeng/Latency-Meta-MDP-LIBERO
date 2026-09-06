"""Decision-stage semi-MDP bookkeeping over contiguous physical control ticks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionTransition:
    state: Any
    action: str
    proposed_action: str
    shielded: bool
    next_state: Any
    start_tick: int
    end_tick: int
    reward: float
    undiscounted_reward: float
    discount: float
    terminated: bool
    truncated: bool

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def bootstrap_discount(self) -> float:
        return 0.0 if self.terminated else self.discount


class DecisionStageAccumulator:
    """Bookkeeping only; callers supply public state and a fixed policy environment."""

    def __init__(self, *, gamma: float = 1.0):
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0,1]")
        self.gamma = float(gamma)
        self._current = None
        self._rewards = []

    @property
    def active(self) -> bool:
        return self._current is not None

    def begin(
        self,
        *,
        formal_tick: int,
        state: Any,
        action: str,
        proposed_action: str | None = None,
        shielded: bool = False,
    ) -> None:
        if self.active:
            raise RuntimeError("finish the preceding decision interval before starting another")
        if type(formal_tick) is not int or formal_tick < 0 or action not in {"wait", "launch"}:
            raise ValueError("decision identity is invalid")
        proposed = action if proposed_action is None else proposed_action
        if proposed not in {"wait", "launch"} or type(shielded) is not bool:
            raise ValueError("proposed/shielded decision is invalid")
        self._current = (formal_tick, state, action, proposed, shielded)
        self._rewards = []

    def add_reward(self, *, formal_tick: int, reward: float) -> None:
        if not self.active:
            raise RuntimeError("no decision interval is active")
        if formal_tick != self._current[0] + len(self._rewards) or not math.isfinite(reward):
            raise ValueError("rewards must cover every physical tick exactly once")
        self._rewards.append(float(reward))

    def finish(
        self,
        *,
        next_formal_tick: int,
        next_state: Any,
        terminated: bool = False,
        truncated: bool = False,
    ) -> DecisionTransition:
        if not self.active:
            raise RuntimeError("no decision interval is active")
        tick, state, action, proposed, shielded = self._current
        duration = len(self._rewards)
        if (
            duration == 0
            or next_formal_tick != tick + duration
            or type(terminated) is not bool
            or type(truncated) is not bool
        ):
            raise ValueError("decision holding time does not match collected physical rewards")
        result = DecisionTransition(
            state=state,
            action=action,
            proposed_action=proposed,
            shielded=shielded,
            next_state=next_state,
            start_tick=tick,
            end_tick=next_formal_tick,
            reward=sum(self.gamma**i * value for i, value in enumerate(self._rewards)),
            undiscounted_reward=sum(self._rewards),
            discount=self.gamma**duration,
            terminated=terminated,
            truncated=truncated,
        )
        self._current = None
        self._rewards = []
        return result
