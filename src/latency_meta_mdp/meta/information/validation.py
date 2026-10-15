"""Paired validation metrics and hash-bound best/latest weight retention."""

import json
import math
from pathlib import Path

from latency_meta_mdp.meta.information.config import sha256


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validation_metrics(rows, *, scenes, starvation_mode="guarded"):
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
    for key in ("zero/reward", "nominal/reward", "family/reward", "selection_score"):
        if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError("invalid validation reward")
    if not math.isclose(
        metrics["selection_score"], (metrics["nominal/reward"] + metrics["family/reward"]) / 2
    ):
        raise ValueError("selection score must use delayed modes")
    item = dict(
        metrics=metrics,
        evaluation_sha256=evaluation_sha256,
        checkpoint_receipt_sha256=sha256(directory / f"iteration-{iteration:04d}/complete.json"),
    )
    if str(iteration) in state["completed"]:
        validate_binding(item, state["completed"][str(iteration)])
    state["completed"][str(iteration)] = item
    incumbent = state["best_iteration"]
    if incumbent is None or (
        metrics["selection_score"]
        > state["completed"][str(incumbent)]["metrics"]["selection_score"]
        and not math.isclose(
            metrics["selection_score"],
            state["completed"][str(incumbent)]["metrics"]["selection_score"],
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        state["best_iteration"] = iteration
    atomic_json(path, state)
    return state
