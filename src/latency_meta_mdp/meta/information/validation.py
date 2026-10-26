"""Paired validation metrics and hash-bound best/latest weight retention."""

import json
import math
from pathlib import Path

from latency_meta_mdp.meta.information.config import sha256

DELAYED_SELECTION = "mean_nominal_family_incumbent_on_ties"
BUDGET_SELECTION = "mean_budget_all_regimes"


def validation_rank(metrics, selection=DELAYED_SELECTION):
    """One predeclared ordering for startup, per-fit and campaign-wide selection."""
    rewards = [metrics[f"{r}/reward"] for r in ("zero", "nominal", "family")]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rewards):
        raise ValueError("invalid validation reward")
    if selection == DELAYED_SELECTION:
        expected = sum(rewards[1:]) / 2
        rank = (round(60 * expected),)
    elif selection == BUDGET_SELECTION:
        returns = [metrics[f"{r}/budget_return"] for r in ("zero", "nominal", "family")]
        costs = [metrics[f"{r}/compute_cost"] for r in ("zero", "nominal", "family")]
        if any(not math.isfinite(v) for v in returns + costs) or min(costs) < 0:
            raise ValueError("invalid selection budget metrics")
        expected = sum(returns) / 3
        rank = (round(expected, 10), round(sum(rewards), 10), -round(sum(costs), 10))
    else:
        raise ValueError("unknown validation selection")
    if not math.isfinite(metrics["selection_score"]) or not math.isclose(
        metrics["selection_score"], expected, rel_tol=0, abs_tol=1e-10
    ):
        raise ValueError("selection score differs from declared protocol")
    return rank


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validation_metrics(rows, *, scenes, starvation_mode="guarded", selection=DELAYED_SELECTION):
    if starvation_mode not in ("guarded", "allow_hold"):
        raise ValueError("unknown validation starvation protocol")
    expected = {(s, r) for s in scenes for r in ("zero", "nominal", "family")}
    if len(set(scenes)) != 3 or len(rows) != 9:
        raise ValueError("validation requires three distinct paired scenes and nine episodes")
    seen, rewards = set(), {r: [] for r in ("zero", "nominal", "family")}
    for row in rows:
        ep, report = row["episode"], row["report"]
        key = (ep["scene"], ep["regime"])
        p = report["parcels"]
        if (
            key not in expected
            or key in seen
            or ep["behavior"] not in ("fqi", "actor")
            or (ep["behavior"] == "actor" and not ep.get("deterministic", False))
            or ep["epsilon"] != 0
            or ep["split"] != "validation"
            or not report["terminated"]
            or report["truncated"]
            or (report["starvation_ticks"] and starvation_mode != "allow_hold")
            or p["spawned"] != 10
            or p["active"]
            or sum(p[k] for k in ("successes", "misses", "timeouts")) != 10
            or not 0 <= p["successes"] <= 10
            or not math.isclose(report["task_reward"], p["successes"] / 10)
        ):
            raise ValueError("invalid or incomplete paired validation episode")
        seen.add(key)
        rewards[ep["regime"]].append(report["task_reward"])
    result = {f"{r}/reward": sum(values) / 3 for r, values in rewards.items()}
    result["selection_score"] = (result["nominal/reward"] + result["family/reward"]) / 2
    for row in rows:
        ep, report = row["episode"], row["report"]
        prefix = f"{ep['regime']}/scene-{ep['scene']}"
        for label, key in (("deliveries", "successes"), ("drops", "misses"),
                           ("timeouts", "timeouts")):
            result[f"{prefix}/{label}"] = report["parcels"][key]
        result[f"{prefix}/delivery_ratio"] = report["task_reward"]
        for key in ("compute_cost", "episode_reward"):
            if key in report.get("budget", {}):
                result[f"{prefix}/{key}"] = report["budget"][key]
        for component, count in report.get("budget", {}).get("module_calls", {}).items():
            result[f"{prefix}/calls/{component}"] = count
    if any("scheduling" in row["report"] for row in rows):
        if not all("scheduling" in row["report"] for row in rows):
            raise ValueError("mixed scheduling protocols in validation")
        for regime in rewards:
            values = [
                row["report"]["scheduling"] for row in rows if row["episode"]["regime"] == regime
            ]
            for key in (
                "hold_ticks",
                "observe_requests",
                "predict_requests",
                "launch_requests",
                "predicted_launches",
                "current_only_launches",
            ):
                result[f"{regime}/{key}"] = sum(v[key] for v in values) / 3
            launches = sum(v["launch_requests"] for v in values)
            result[f"{regime}/forecast_launch_fraction"] = (
                sum(v["predicted_launches"] for v in values) / launches if launches else 0.0
            )
    if any("budget" in row["report"] for row in rows):
        if not all("budget" in row["report"] for row in rows):
            raise ValueError("mixed reward accounting in validation")
        if len({row["report"]["budget"]["reward_spec_hash"] for row in rows}) != 1:
            raise ValueError("mixed budget specifications in validation")
        for row in rows:
            budget, parcels = row["report"]["budget"], row["report"]["parcels"]
            if any(
                budget[k] != parcels[p]
                for k, p in (
                    ("deliveries", "successes"),
                    ("drops", "misses"),
                    ("timeouts", "timeouts"),
                )
            ):
                raise ValueError("budget outcome accounting differs")
            if (
                not math.isfinite(budget["episode_reward"])
                or not math.isfinite(budget["compute_cost"])
                or budget["compute_cost"] < 0
            ):
                raise ValueError("invalid budget accounting")
        for regime in rewards:
            values = [row["report"]["budget"] for row in rows if row["episode"]["regime"] == regime]
            for label, key in (
                ("budget_return", "episode_reward"),
                ("compute_cost", "compute_cost"),
                ("drops", "drops"),
            ):
                result[f"{regime}/{label}"] = sum(v[key] for v in values) / 3
            for component in set().union(*(v["module_calls"] for v in values)):
                result[f"{regime}/calls/{component}"] = (
                    sum(v["module_calls"].get(component, 0) for v in values) / 3
                )
    if selection == BUDGET_SELECTION:
        if not all("budget" in row["report"] for row in rows):
            raise ValueError("budget selection requires complete budget accounting")
        result["selection_score"] = sum(result[f"{r}/budget_return"] for r in rewards) / 3
    validation_rank(result, selection)
    return result


def verified_validation_metrics(checkpoint, evaluation, cached, protocol):
    """Cached selection scores are derived data, never an authority on resume."""
    from latency_meta_mdp.meta.information.config import validate_binding

    validate_binding(
        cached["checkpoint_receipt_sha256"], sha256(Path(checkpoint) / "complete.json")
    )
    summary = Path(evaluation) / "rollout-summary.json"
    validate_binding(cached["evaluation_sha256"], sha256(summary))
    metrics = validation_metrics(
        json.loads(summary.read_text())["episodes"],
        scenes=protocol["scenes"],
        starvation_mode=protocol.get("starvation_mode", "guarded"),
        selection=protocol.get("selection", DELAYED_SELECTION),
    )
    validate_binding(cached["metrics"], metrics)
    return metrics


def retain_checkpoints(directory, *, best_iteration):
    """Keep all receipts, but only best/latest weights; publish intent before unlink."""
    directory = Path(directory)
    entries = sorted(directory.glob("iteration-*/complete.json"))
    if not entries:
        raise ValueError("no complete checkpoints")
    latest = json.loads(entries[-1].read_text())["iteration"]
    keep = {latest, best_iteration}
    for iteration in keep:
        folder = directory / f"iteration-{iteration:04d}"
        receipt = json.loads((folder / "complete.json").read_text())
        if receipt["files"] != {"state.pt": sha256(folder / "state.pt")}:
            raise ValueError("retained checkpoint hash differs")
    pruned = {}
    for path in entries:
        receipt = json.loads(path.read_text())
        if receipt["iteration"] not in keep:
            pruned[path.parent.name] = sha256(path)
    atomic_json(directory / "retention.json", dict(pruned=pruned))
    for role, iteration in (("best", best_iteration), ("latest", latest)):
        temp = directory / f".{role}.tmp"
        temp.unlink(missing_ok=True)
        temp.symlink_to(f"iteration-{iteration:04d}", target_is_directory=True)
        temp.replace(directory / role)
    for name in pruned:
        (directory / name / "state.pt").unlink(missing_ok=True)


def record_validation(directory, *, iteration, metrics, protocol, evaluation_sha256):
    from latency_meta_mdp.meta.information.config import validate_binding

    directory = Path(directory)
    path = directory / "validation-state.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else dict(protocol=protocol, completed={}, best_iteration=None)
    )
    validate_binding(protocol, state["protocol"])
    selection = protocol.get("selection", DELAYED_SELECTION)
    rank = validation_rank(metrics, selection)
    item = dict(
        metrics=metrics,
        evaluation_sha256=evaluation_sha256,
        checkpoint_receipt_sha256=sha256(directory / f"iteration-{iteration:04d}/complete.json"),
    )
    if str(iteration) in state["completed"]:
        validate_binding(item, state["completed"][str(iteration)])
    state["completed"][str(iteration)] = item
    incumbent = state["best_iteration"]
    if incumbent is None or rank > validation_rank(
        state["completed"][str(incumbent)]["metrics"], selection
    ):
        state["best_iteration"] = iteration
    atomic_json(path, state)
    return state
