"""Conserved raw stages and a learner-neutral free-decision training view."""

import math
from contextlib import contextmanager
from time import perf_counter

from latency_meta_mdp.meta.information.schema import STAGES


class StageRecorder:
    def __init__(self, episode_id, *, action_spec=None):
        self.episode_id = episode_id
        self.action_spec = action_spec
        self.events = []
        self.rows, self.operations = [], []
        self._ids = set()
        self._operation_counter = 0
        self._timers = []
        self._last_reward_tick = None
        self.closed = False
        self.prefix_reward = 0.0
        self.prefix_cost = 0.0

    def start(self, packet, action, *, forced, probability):
        count = 8 if self.action_spec is None else self.action_spec.num_actions
        if self.closed or type(action) is not int or not 0 <= action < count:
            raise ValueError("invalid stage action")
        if type(forced) is not bool or not math.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("invalid stage behavior probability")
        if self.rows:
            previous = self.rows[-1]
            last = previous["packet"]
            if packet.tick < last.tick or (
                packet.tick == last.tick and STAGES.index(packet.stage) <= STAGES.index(last.stage)
            ):
                raise ValueError("stage must advance without same-tick cycles")
            previous["end_tick"] = packet.tick
            previous["next_packet"] = packet
        self.rows.append(
            dict(
                packet=packet,
                action=action,
                forced=forced,
                probability=float(probability),
                task_reward=0.0,
                cost=0.0,
                end_tick=None,
                next_packet=None,
                terminated=False,
                truncated=False,
            )
        )

    def add_cost(self, operation_id, component, seconds, **metadata):
        if self.closed or operation_id in self._ids:
            raise ValueError("duplicate or closed operation")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("operation cost must be finite and nonnegative")
        self._ids.add(operation_id)
        owner = len(self.rows) - 1
        self.operations.append(
            dict(
                operation_id=operation_id,
                owner=owner,
                component=component,
                seconds=float(seconds),
                **metadata,
            )
        )
        if owner < 0:
            self.prefix_cost += seconds
        else:
            self.rows[owner]["cost"] += seconds

    @contextmanager
    def measure(self, component, **metadata):
        started = perf_counter()
        operation_id = f"{self.episode_id}:{self._operation_counter}"
        self._operation_counter += 1
        timer = {"children": 0.0}
        self._timers.append(timer)
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self._timers.pop()
            if self._timers:
                self._timers[-1]["children"] += elapsed
            self.add_cost(
                operation_id, component, max(0.0, elapsed - timer["children"]), **metadata
            )

    def add_reward(self, tick, delivered):
        if (
            self.closed
            or type(tick) is not int
            or type(delivered) is not int
            or delivered < 0
            or (self._last_reward_tick is not None and tick != self._last_reward_tick + 1)
        ):
            raise ValueError("invalid or repeated physical boundary")
        self._last_reward_tick = tick
        if self.rows:
            if tick < self.rows[-1]["packet"].tick:
                raise ValueError("physical boundary predates its owner")
            self.rows[-1]["task_reward"] += delivered / 10
        else:
            self.prefix_reward += delivered / 10

    def finish(self, tick, *, terminated):
        if self.closed or type(terminated) is not bool:
            raise ValueError("episode already closed or terminal flag invalid")
        if self._last_reward_tick is not None and tick != self._last_reward_tick + 1:
            raise ValueError("episode finish must follow final physical boundary")
        if self.rows:
            if tick < self.rows[-1]["packet"].tick:
                raise ValueError("episode clock moved backwards")
            self.rows[-1].update(end_tick=tick, terminated=terminated, truncated=not terminated)
        self.closed = True

    def summary(self):
        import numpy as np

        components = {}
        for op in self.operations:
            components[op["component"]] = components.get(op["component"], 0) + op["seconds"]
        stats = {}
        for name in components:
            values = [op["seconds"] for op in self.operations if op["component"] == name]
            stats[name] = dict(
                count=len(values),
                total=components[name],
                mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)),
            )
        pipeline = (
            "vla",
            "dino",
            "jepa",
            "decoder",
            "meta_q",
            "feature_transfer",
            "forecast_prepare_overhead",
            "feature_cache",
        )
        return dict(
            task_reward=self.prefix_reward + sum(r["task_reward"] for r in self.rows),
            measured_seconds=sum(components.values()),
            components=components,
            component_statistics=stats,
            model_pipeline_components=list(pipeline),
            model_pipeline_seconds=sum(components.get(k, 0) for k in pipeline),
            prefix_reward=self.prefix_reward,
            prefix_cost=self.prefix_cost,
            raw_stages=len(self.rows),
        )


def decision_view(rows):
    if not rows or not rows[-1]["terminated"] or any(r["truncated"] for r in rows):
        raise ValueError("incomplete episode is not admitted to the training view")
    result = []
    for row in rows:
        if not row["forced"]:
            if result:
                result[-1]["next_packet"] = row["packet"]
            result.append(
                dict(
                    packet=row["packet"],
                    action=row["action"],
                    probability=row["probability"],
                    task_reward=0.0,
                    cost=0.0,
                    duration=0,
                    next_packet=None,
                    terminated=False,
                )
            )
        if result:
            if row["end_tick"] is None:
                raise ValueError("unfinished raw stage")
            out = result[-1]
            out["task_reward"] += row["task_reward"]
            out["cost"] += row["cost"]
            out["duration"] += row["end_tick"] - row["packet"].tick
            out["terminated"] = row["terminated"]
    return result
