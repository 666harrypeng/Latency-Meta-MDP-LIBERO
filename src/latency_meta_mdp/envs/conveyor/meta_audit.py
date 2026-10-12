"""Post-episode conservation and outcome audit; never a scheduler observation."""

import math

from latency_meta_mdp.meta.information.records import decision_view


def audit_episode(recorder, report):
    if not recorder.closed:
        raise ValueError("cannot audit an open episode")
    outcomes, last_tick = {}, -1
    for event in report["parcel_events"]:
        parcel, tick, status = event["parcel_id"], event["tick"], event["event"]
        if tick < last_tick:
            raise ValueError("noncausal parcel outcome ledger")
        last_tick = tick
        if status == "spawn" and parcel not in outcomes:
            outcomes[parcel] = "active"
        elif status in ("success", "miss", "timeout") and outcomes.get(parcel) == "active":
            outcomes[parcel] = status
        else:
            raise ValueError("duplicate or invalid parcel outcome")
    if report["terminated"] and "active" in outcomes.values():
        raise ValueError("terminal episode retains an unresolved outcome")
    summary = recorder.summary()
    if not math.isclose(
        summary["task_reward"], list(outcomes.values()).count("success") / 10, abs_tol=1e-9
    ):
        raise ValueError("delivery reward is not conserved")
    total_cost = recorder.prefix_cost + sum(r["cost"] for r in recorder.rows)
    if not math.isclose(total_cost, summary["measured_seconds"], abs_tol=1e-9):
        raise ValueError("operation cost is not conserved")
    transitions = decision_view(recorder.rows) if report["terminated"] and recorder.rows else []
    # Forced prefix before the first free decision is not learnable, but stays in raw totals.
    first_free = next(
        (i for i, r in enumerate(recorder.rows) if not r["forced"]), len(recorder.rows)
    )
    excluded = recorder.rows[:first_free]
    if transitions:
        for field, prefix in [
            ("cost", recorder.prefix_cost),
            ("task_reward", recorder.prefix_reward),
        ]:
            total = prefix + sum(r[field] for r in excluded) + sum(r[field] for r in transitions)
            expected = summary["measured_seconds"] if field == "cost" else summary["task_reward"]
            if not math.isclose(total, expected, abs_tol=1e-9):
                raise ValueError("free-decision folding is not conserved")
    return dict(
        task_reward=summary["task_reward"],
        outcomes=outcomes,
        free_decisions=len(transitions),
        free_decision_cost=sum(r["cost"] for r in transitions),
        failures=[
            dict(parcel_id=k, outcome=v, failure_mode="unknown")
            for k, v in outcomes.items()
            if v in ("miss", "timeout")
        ],
        failure_mode_scope="No inferred grasp-stage cause without an observed event trace",
    )


class ProgressAudit:
    """Current-world sidecar and observed progress events, without reward shaping."""

    def __init__(self, path, *, goal_center, goal_radius, minimum_z, grasp_ticks):
        from pathlib import Path

        import numpy as np

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.goal_center = np.asarray(goal_center)
        self.goal_radius, self.minimum_z, self.grasp_ticks = goal_radius, minimum_z, grasp_ticks
        self.history, self.events, self.last_tick = {}, [], -1

    def record(self, tick, parcels, statuses):
        import json

        import numpy as np

        if tick != self.last_tick + 1:
            raise ValueError("audit requires consecutive physical boundaries")
        self.last_tick = tick
        for parcel in parcels:
            pid, grasped = parcel["parcel_id"], bool(parcel["grasped"])
            position = np.asarray(parcel["position"])
            old = self.history.get(pid, dict(grasp=0, lifted=False, goal=False))
            count = old["grasp"] + 1 if grasped else 0
            lifted = bool(grasped and position[2] >= self.minimum_z)
            goal = bool(np.linalg.norm(position - self.goal_center) <= self.goal_radius)
            transitions = {
                "grasp_started": count == 1,
                "grasp_sustained": count == self.grasp_ticks,
                "grasp_lost": bool(old["grasp"] and not grasped),
                "lift_threshold_crossed": lifted and not old["lifted"],
                "goal_entered": goal and not old["goal"],
            }
            for event, happened in transitions.items():
                if happened:
                    self.events.append(dict(parcel_id=pid, tick=tick, event=event))
            self.history[pid] = dict(grasp=count, lifted=lifted, goal=goal)
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(tick=tick, parcels=parcels, statuses=statuses)) + "\n")

    def observe(self, world, tick):
        env = world.env
        rows = world.trace(tick)["parcels"]
        for row in rows:
            row["grasped"] = bool(
                env._check_grasp(env.robots[0].gripper, env.parcels[row["parcel_id"]])
            )
        self.record(tick, rows, world.ledger.statuses)
