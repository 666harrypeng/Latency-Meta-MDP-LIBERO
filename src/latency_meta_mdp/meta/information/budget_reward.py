"""Versioned event reward views; source recordings and actor inputs stay unchanged."""

import math
from bisect import bisect_right
from collections import Counter

# These are instrumentation or preparation intervals, not additional model calls.
# Preparation must be included in calibrated service prices rather than double charged.
UNPRICED_COMPONENTS = frozenset(
    {
        "audit",
        "video",
        "capture",
        "physics",
        "simulation_initialize",
        "simulation_step_and_capture",
        "source_archive",
        "feature_transfer",
        "feature_cache",
        "forecast_prepare_overhead",
        "rule",
        "rule_scheduler",
    }
)
MODEL_COMPONENTS = frozenset({"vla", "dino", "jepa", "decoder", "meta_q", "bootstrap"})


def validate_reward_spec(spec):
    if set(spec) != {"version", "delivery_reward", "drop_penalty", "cost_weight", "prices"}:
        raise ValueError("reward specification has missing or unknown fields")
    if spec["version"] != 1 or set(spec["prices"]) != MODEL_COMPONENTS:
        raise ValueError("unsupported reward version or module prices")
    values = [spec[k] for k in ("delivery_reward", "drop_penalty", "cost_weight")]
    values += list(spec["prices"].values())
    if any(
        isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v < 0
        for v in values
    ):
        raise ValueError("reward coefficients must be finite and nonnegative")
    if spec["delivery_reward"] <= 0 or spec["prices"]["vla"] <= 0:
        raise ValueError("positive delivery reward and VLA price required")


def reward_view(rows, operations, parcel_events, spec):
    """Assign outcome ticks and operation owners, then fold into free decisions.

    The old task_reward is retained as a delivery-alignment assertion, not mutated.
    Work before the first free decision stays in the excluded prefix. No privileged
    outcome fields are attached to a context or made visible to the actor.
    """
    validate_reward_spec(spec)
    if not rows or not parcel_events:
        raise ValueError("complete raw rows and parcel ledger required")
    ticks = [row["packet"].tick for row in rows]
    if ticks != sorted(ticks) or any(r["end_tick"] is None for r in rows):
        raise ValueError("invalid raw stage timeline")
    if any(r["end_tick"] != ticks[i + 1] for i, r in enumerate(rows[:-1])):
        raise ValueError("discontinuous raw stage timeline")
    finish = rows[-1]["end_tick"]
    # Slot zero is the pre-row prefix; subsequent slots correspond to raw stages.
    deliveries, drops = [0] * (len(rows) + 1), [0] * (len(rows) + 1)
    costs = [0.0] * (len(rows) + 1)
    outcomes, last_tick = {}, -1
    for event in parcel_events:
        pid, tick, kind = (event[k] for k in ("parcel_id", "tick", "event"))
        if type(tick) is not int or tick < last_tick or tick < 0 or tick > finish:
            raise ValueError("invalid parcel outcome timeline")
        last_tick = tick
        if kind == "spawn" and pid not in outcomes:
            outcomes[pid] = "active"
            continue
        if kind not in ("success", "miss", "timeout") or outcomes.get(pid) != "active":
            raise ValueError("duplicate or invalid parcel outcome")
        outcomes[pid] = kind
        if kind in ("success", "miss"):
            if tick <= 0 or tick > finish:
                raise ValueError("outcome outside executed decision intervals")
            # ConveyorWorld labels the resulting boundary t+1; StageRecorder
            # attributes its reward to the action executed at t. A request made
            # at the new boundary must not receive credit for the preceding step.
            owner = bisect_right(ticks, tick - 1)
            (deliveries if kind == "success" else drops)[owner] += 1
    if "active" in outcomes.values():
        raise ValueError("incomplete parcel ledger")
    for i, row in enumerate(rows):
        if not math.isclose(row["task_reward"], deliveries[i + 1] / 10, abs_tol=1e-8):
            raise ValueError("delivery ledger and raw reward ownership differ")
    optional = getattr(getattr(rows[0]["packet"], "action_spec", None), "version", 1) == 3
    seen, counts, decision_services = set(), Counter(), Counter()
    imputed_meta_calls = 0
    for op in operations:
        oid, owner, component = (op[k] for k in ("operation_id", "owner", "component"))
        if oid in seen:
            raise ValueError("duplicate operation")
        seen.add(oid)
        if type(owner) is not int or not -1 <= owner < len(rows):
            raise ValueError("invalid operation owner")
        if component not in MODEL_COMPONENTS | UNPRICED_COMPONENTS:
            raise ValueError(f"unclassified operation component: {component}")
        if component in MODEL_COMPONENTS:
            counts[component] += 1
            costs[owner + 1] += spec["prices"][component]
        if optional and component in ("meta_q", "rule_scheduler"):
            decision_services[owner] += 1
            if component == "rule_scheduler":
                # The data collector is a rule, but the trained deployment runs
                # Meta at every free decision. Preserve actual module counts and
                # price this explicit service equivalent only in the reward view.
                costs[owner + 1] += spec["prices"]["meta_q"]
                imputed_meta_calls += 1
    if optional and decision_services != Counter(
        {i: 1 for i, row in enumerate(rows) if not row["forced"]}
    ):
        raise ValueError("optional protocol requires one decision service per free decision")
    raw_rewards = [
        spec["delivery_reward"] * d - spec["drop_penalty"] * m - spec["cost_weight"] * c
        for d, m, c in zip(deliveries, drops, costs)
    ]
    decisions, prefix = [], raw_rewards[0]
    for row, reward in zip(rows, raw_rewards[1:], strict=True):
        if not row["forced"]:
            decisions.append(0.0)
        if decisions:
            decisions[-1] += reward
        else:
            prefix += reward
    result = dict(
        decision_rewards=decisions,
        prefix_reward=prefix,
        episode_reward=sum(raw_rewards),
        compute_cost=sum(costs),
        deliveries=sum(deliveries),
        drops=sum(drops),
        timeouts=sum(v == "timeout" for v in outcomes.values()),
        module_calls=dict(counts),
    )
    if optional:
        result.update(
            decision_cost_policy="deployment_meta_per_free_decision_v1",
            imputed_meta_calls=imputed_meta_calls,
        )
    return result
