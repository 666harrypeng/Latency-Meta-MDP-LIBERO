"""Fixed reference costs over actual model invocations, independent of logical delay."""

import json
import math
from bisect import bisect_right
from pathlib import Path

COMPONENTS = (
    "vla_conditioned",
    "vla_clean",
    "forecast",
    "decode",
    "meta_q",
    "rule",
    "bootstrap_clean",
)


def validate_cost_profile(profile):
    if (
        profile.get("schema") != 1
        or profile.get("unit") != "conditioned_vla_rtc_equivalent"
        or not math.isfinite(profile.get("reference_seconds", math.nan))
        or profile["reference_seconds"] <= 0
    ):
        raise ValueError("invalid reference cost profile")
    weights = profile.get("weights", {})
    if (
        set(weights) != set(COMPONENTS)
        or weights["vla_conditioned"] != 1
        or any(not math.isfinite(v) or v < 0 for v in weights.values())
        or not isinstance(profile.get("binding"), dict)
    ):
        raise ValueError("invalid model cost weights or binding")
    return profile


def load_cost_profile(path):
    return validate_cost_profile(json.loads(Path(path).read_text()))


def budget_multiplier_step(value, mean_cost, budget, step_size):
    if (
        not all(math.isfinite(v) for v in (value, mean_cost, budget, step_size))
        or min(value, mean_cost, step_size) < 0
        or budget <= 0
    ):
        raise ValueError("invalid budget feedback")
    return max(0.0, value + step_size * (mean_cost / budget - 1.0))


def episode_cost_components(result, profile):
    """Charge each call once, at its source decision (including uninstalled requests).

    Runtime may log forecast+decode together for rules, or separately for Meta.
    The reference weights are fixed: observed wall times are not training rewards.
    """
    validate_cost_profile(profile)
    identity = result["identity"]
    forecast_modes = ("native_rtc_forecast_rgb_v1", "native_rtc_forecast_only_rgb_v1")
    conditioned = identity.get("conditioning") in forecast_modes
    if identity.get("conditioning") not in (None, *forecast_modes):
        raise ValueError("cost accounting only supports native RTC and forecast RGB policies")
    bindings = {
        **profile["binding"],
        **profile.get("policy_bindings", {}).get("conditioned" if conditioned else "clean", {}),
    }
    if any(identity.get(k) != v for k, v in bindings.items()):
        raise ValueError("cost profile/result model binding differs")
    events = result["stage_events"]
    rows = result.get("decision_transitions") or None
    if rows is not None:
        starts = [r["start_tick"] for r in rows]
        if any(r["end_tick"] <= r["start_tick"] for r in rows) or any(
            a["end_tick"] != b["start_tick"] for a, b in zip(rows, rows[1:])
        ):
            raise ValueError("noncontiguous cost decision intervals")
    costs = [0.0] * len(rows) if rows is not None else None
    counts = dict.fromkeys(COMPONENTS, 0)
    initial = 0.0
    invocations = []

    def add(kind, tick):
        nonlocal initial
        counts[kind] += 1
        charge = profile["weights"][kind]
        invocations.append((kind, tick, charge))
        if tick is None or (rows is not None and tick < starts[0]):
            initial += charge
        elif rows is not None:
            index = bisect_right(starts, tick) - 1
            if index < 0 or tick >= rows[index]["end_tick"]:
                raise ValueError("model invocation outside decision intervals")
            costs[index] += charge

    def requests(stage):
        found = {}
        for e in events:
            if e["stage"] == stage:
                rid = e.get("request_id")
                if rid is None or rid in found:
                    raise ValueError(f"missing or duplicate request identity: {stage}")
                found[rid] = e
        return found

    launches, returns = requests("policy_launch"), requests("policy_return")
    if set(launches) != set(returns) or len(launches) != result["policy_calls"]:
        raise ValueError("incomplete model request accounting")
    combined, decodes = requests("forecast"), requests("forecast_decode")
    prepared = [e for e in events if e["stage"] == "forecast_prepare"]
    if combined and (prepared or decodes):
        raise ValueError("duplicate combined/separate forecast accounting")
    if len({e["formal_tick"] for e in prepared}) != len(prepared) or not (
        set(combined) | set(decodes)
    ) <= set(launches):
        raise ValueError("duplicate forecast or unmatched decoder request")
    for rid, e in {**combined, **decodes}.items():
        if e["formal_tick"] != launches[rid]["formal_tick"]:
            raise ValueError("forecast/request source boundary differs")
    if prepared:
        by_tick = {e["formal_tick"]: e for e in prepared}
        for e in decodes.values():
            source = by_tick.get(e["formal_tick"])
            if source is None or source["available"] != e["available"]:
                raise ValueError("decoder has no matching prepared forecast")
    if len(prepared) + len(combined) != result["forecast_calls"]:
        raise ValueError("forecast invocation count differs")
    decoded_count = sum(bool(e["available"]) for e in [*combined.values(), *decodes.values()])
    if decoded_count != result["forecast_decodes"]:
        raise ValueError("decoder invocation count differs")
    if conditioned and set(launches) != set(combined) | set(decodes):
        raise ValueError("conditioned policy lacks forecast invocation")
    for e in [*prepared, *combined.values()]:
        if e["available"]:
            add("forecast", e["formal_tick"])
    for e in [*combined.values(), *decodes.values()]:
        if e["available"]:
            add("decode", e["formal_tick"])
    for e in launches.values():
        add("vla_conditioned" if conditioned else "vla_clean", e["formal_tick"])
    decisions = [e for e in events if e["stage"] == "meta_decision"]
    if len({e["formal_tick"] for e in decisions}) != len(decisions):
        raise ValueError("duplicate Meta decision")
    if rows is not None and [e["formal_tick"] for e in decisions] != starts:
        raise ValueError("decision events/replay intervals differ")
    for e in decisions:
        add("meta_q" if e.get("q_values") is not None else "rule", e["formal_tick"])
    bootstrap = [e for e in events if e["stage"] == "bootstrap_launch"]
    if len(bootstrap) != result["bootstrap_calls"] or len(bootstrap) != sum(
        e["stage"] == "bootstrap_return" for e in events
    ):
        raise ValueError("bootstrap invocation count differs")
    for _ in bootstrap:
        add("bootstrap_clean", None)
    total = math.fsum(v for _, _, v in invocations)
    if rows is not None and not math.isclose(total, initial + math.fsum(costs)):
        raise ValueError("cost accounting does not conserve episode total")
    return {
        "episode_cost": total,
        "initial_cost": initial,
        "transition_costs": costs,
        "counts": counts,
        "auxiliary_wall_ns": {
            stage: sum(e.get("wall_duration_ns", 0) for e in events if e["stage"] == stage)
            for stage in ("forecast_history", "observation", "control_and_next_observation")
        },
    }
