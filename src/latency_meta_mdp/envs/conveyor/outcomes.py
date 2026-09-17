"""Per-parcel terminal events and a supply-plus-drain episode clock."""

from dataclasses import dataclass

import numpy as np


class ParcelLedger:
    def __init__(self, *, supply_ticks, drain_ticks):
        if any(type(t) is not int or t <= 0 for t in (supply_ticks, drain_ticks)):
            raise ValueError("supply and drain durations must be positive formal ticks")
        self.supply_ticks, self.drain_ticks = supply_ticks, drain_ticks
        self.statuses, self.events = {}, []
        self.ended_at = None
        self.last_tick = 0

    def spawn(self, parcel_id, tick):
        if (
            self.ended_at is not None
            or not self.last_tick <= tick < self.supply_ticks
            or parcel_id in self.statuses
        ):
            raise ValueError("invalid or duplicate parcel birth")
        self.last_tick = tick
        self.statuses[parcel_id] = "active"
        self.events.append(dict(parcel_id=parcel_id, tick=tick, event="spawn"))

    def finish(self, parcel_id, status, tick):
        if (
            self.ended_at is not None
            or self.statuses.get(parcel_id) != "active"
            or status not in {"success", "miss", "timeout"}
            or tick < self.last_tick
        ):
            raise ValueError("invalid, duplicate or noncausal parcel termination")
        self.last_tick = tick
        self.statuses[parcel_id] = status
        self.events.append(dict(parcel_id=parcel_id, tick=tick, event=status))

    def advance(self, tick):
        if tick < self.last_tick:
            raise ValueError("parcel clock cannot run backwards")
        self.last_tick = tick
        if self.ended_at is not None:
            return True
        if tick >= self.supply_ticks + self.drain_ticks:
            for parcel_id, status in list(self.statuses.items()):
                if status == "active":
                    self.finish(parcel_id, "timeout", tick)
        if tick >= self.supply_ticks and "active" not in self.statuses.values():
            self.ended_at = tick
        return self.ended_at is not None

    def summary(self):
        counts = {
            s: list(self.statuses.values()).count(s)
            for s in ("success", "miss", "timeout", "active")
        }
        return {
            "spawned": len(self.statuses),
            "successes": counts["success"],
            "misses": counts["miss"],
            "timeouts": counts["timeout"],
            "active": counts["active"],
            "success_rate": counts["success"] / len(self.statuses) if self.statuses else None,
            "end_tick": self.ended_at,
        }


@dataclass
class DeliveryProgress:
    last_tick: int = -2
    grasp_count: int = 0
    ready_width: float | None = None
    releasing: bool = False
    completed: bool = False


class GoalDeliveryTracker:
    protocol_id = "grasp_goal_open_release_v1"

    def __init__(
        self,
        *,
        center,
        radius,
        ball_radius,
        resting_height,
        min_lift,
        grasp_ticks,
        release_width_m,
        release_opening_delta_m,
    ):
        self.center, self.radius = np.asarray(center), radius + ball_radius
        self.minimum_z = resting_height + min_lift
        self.grasp_ticks = grasp_ticks
        self.release_width = release_width_m
        self.opening_delta = release_opening_delta_m
        self.history = {}
        self.events = []

    def observe(self, parcel_id, tick, position, *, grasped, opening, gripper_width):
        if not np.isfinite(gripper_width):
            raise ValueError("delivery requires a measured finite gripper width")
        state = self.history.setdefault(parcel_id, DeliveryProgress())
        if state.completed:
            return False
        if state.last_tick != tick - 1:
            state = self.history[parcel_id] = DeliveryProgress()
        state.last_tick = tick
        state.grasp_count = state.grasp_count + 1 if grasped else 0
        overlap = np.linalg.norm(np.asarray(position) - self.center) <= self.radius
        if not overlap:
            state.ready_width, state.releasing = None, False
            return False
        if not opening:
            state.releasing = False
            if not grasped:
                # Slipping while closed cannot become a delivery by opening later.
                state.ready_width = None
            elif (
                state.grasp_count >= self.grasp_ticks and position[2] >= self.minimum_z
            ) or state.ready_width is not None:
                if state.ready_width is None:
                    self.events.append(dict(parcel_id=parcel_id, tick=tick, event="held_in_goal"))
                state.ready_width = gripper_width
            return False
        if state.ready_width is None:
            return False
        if not state.releasing:
            self.events.append(dict(parcel_id=parcel_id, tick=tick, event="release_requested"))
            state.releasing = True
        if (
            not grasped
            and gripper_width >= self.release_width
            and gripper_width >= state.ready_width + self.opening_delta
        ):
            state.completed = True
            self.events.append(dict(parcel_id=parcel_id, tick=tick, event="released_in_goal"))
            return True
        return False
