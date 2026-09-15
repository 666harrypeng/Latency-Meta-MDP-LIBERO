"""Restartable collection, budgeted Q fitting and policy-driven replay expansion."""

import argparse
import json
import subprocess
from pathlib import Path

from latency_meta_mdp.meta.cost import budget_multiplier_step
from latency_meta_mdp.meta.cycle_config import load_cycle_config
from latency_meta_mdp.meta.rollouts import run_rollouts, summarize_results, worker_environment
from latency_meta_mdp.meta.success_data import append_success_replay, create_success_replay


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fit_phase(cfg, replay, multiplier, init, output):
    summary = output / "training-summary.json"
    if summary.exists() and json.loads(summary.read_text()).get("status") == "completed":
        if not all((output / name).is_file() for name in ("model.safetensors", "config.json")):
            raise ValueError("completed phase lacks its inference checkpoint")
        return output
    command = [
        cfg["training_python"],
        "-u",
        "-m",
        "latency_meta_mdp.meta.train",
        "--replay-manifest",
        str(replay),
        "--config",
        cfg["training_config"],
        "--cost-profile",
        cfg["cost_profile_path"],
        "--budget",
        str(cfg["budget"]),
        "--cost-multiplier",
        str(multiplier),
        "--output-dir",
        str(output),
        "--visual-cache",
        str(output.parent / "visual-cache"),
        "--device",
        "cuda:0",
    ]
    if not cfg["use_future"]:
        command.append("--no-future")
    if (output / "recovery.pt").exists():
        command += ["--resume-from", str(output / "recovery.pt")]
    elif init is not None:
        command += ["--init-from", str(init)]
    # Training prints progress and syncs W&B itself; inherit its stdout in the user's tmux.
    subprocess.run(command, env=worker_environment(cfg["gpu_ids"][0]), check=True)
    if not summary.is_file() or json.loads(summary.read_text()).get("status") != "completed":
        raise RuntimeError("Meta training returned without a completed checkpoint")
    return output


def select_candidate(candidates, *, budget):
    feasible = [c for c in candidates if c["mean_cost"] <= budget]

    def score(candidate):
        groups = candidate["validation"].values()
        success = sum(g["successes"] / g["episodes"] for g in groups) / len(groups)
        return success, -candidate["mean_cost"]

    selected = max(feasible or candidates, key=score)
    return {
        **selected,
        "feasible_candidate_found": bool(feasible),
        "selection_scope": "internal_validation_and_train_budget_feedback",
    }


def run_cycle(config_path, output, *, resume=False):
    cfg = load_cycle_config(config_path)
    output = Path(output).resolve()
    journal = output / "cycle.json"
    if journal.exists():
        if not resume:
            raise FileExistsError(f"{journal}; use --resume")
        state = json.loads(journal.read_text())
        if state["identity"] != cfg["identity"]:
            raise ValueError("cycle configuration changed; use a new output directory")
        if state["status"] == "completed":
            return state
    else:
        state = {"schema": 1, "identity": cfg["identity"], "completed": {}, "phases": []}

    def stage(name, action):
        if name in state["completed"]:
            return state["completed"][name]
        state.update(status="running", stage=name)
        state.pop("error", None)
        write_json(journal, state)
        print(f"Meta cycle: {name}", flush=True)
        result = action()
        state["completed"][name] = result
        write_json(journal, state)
        return result

    def rollouts(name, partition, checkpoint, *, collect, epsilon, replica):
        return stage(
            name,
            lambda: run_rollouts(
                cfg,
                partition,
                output / "rollouts" / name,
                checkpoint=checkpoint,
                collect=collect,
                epsilon=epsilon,
                replica=replica,
            ),
        )

    try:
        snapshot = cfg["initial_replay"]
        if snapshot is None:
            entries = []
            for group in ("train", "validation"):
                entries += rollouts(
                    "initial-" + group,
                    group,
                    None,
                    collect=True,
                    epsilon=1.0,
                    replica=cfg["replica_start"],
                )
            target = output / "replay/000000.json"
            if not target.exists():
                create_success_replay(entries, target)
            snapshot = str(target)
        multiplier, init = cfg["initial_multiplier"], cfg["initial_checkpoint"]
        for index in range(cfg["phases"]):
            if index < len(state["phases"]):
                previous = state["phases"][index]
                multiplier, init = previous["next_multiplier"], previous["model"]
                snapshot = previous["next_snapshot"]
                continue
            name = f"phase-{index:03d}"
            model = stage(
                name + "-fit",
                lambda: str(fit_phase(cfg, snapshot, multiplier, init, output / "models" / name)),
            )
            replica = cfg["replica_start"] + 1 + 2 * index
            feedback = rollouts(
                name + "-feedback", "feedback", model, collect=True, epsilon=0.0, replica=replica
            )
            feedback_stats = summarize_results(feedback, cfg["cost_profile"])
            mean_cost = sum(g["mean_cost"] for g in feedback_stats.values()) / len(feedback_stats)
            validation = rollouts(
                name + "-validation",
                "validation",
                model,
                collect=False,
                epsilon=0.0,
                replica=replica,
            )
            validation_stats = summarize_results(validation, cfg["cost_profile"])
            entries = list(feedback)
            if index + 1 < cfg["phases"]:
                entries += rollouts(
                    name + "-exploration",
                    "train",
                    model,
                    collect=True,
                    epsilon=cfg["exploration_epsilon"],
                    replica=replica + 1,
                )
            next_snapshot = output / "replay" / f"{index + 1:06d}.json"
            if not next_snapshot.exists():
                append_success_replay(snapshot, entries, next_snapshot)
            next_multiplier = budget_multiplier_step(
                multiplier, mean_cost, cfg["budget"], cfg["multiplier_step"]
            )
            record = {
                "phase": index,
                "model": model,
                "training_snapshot": str(snapshot),
                "multiplier": multiplier,
                "budget": cfg["budget"],
                "feedback": feedback_stats,
                "validation": validation_stats,
                "mean_cost": mean_cost,
                "next_multiplier": next_multiplier,
                "next_snapshot": str(next_snapshot),
            }
            state["phases"].append(record)
            write_json(journal, state)
            multiplier, init, snapshot = next_multiplier, model, str(next_snapshot)
        state["selected"] = select_candidate(state["phases"], budget=cfg["budget"])
        state.update(status="completed", stage="complete")
        write_json(output / "selected.json", state["selected"])
        write_json(journal, state)
        return state
    except BaseException as error:
        state.update(status="failed", error=str(error))
        write_json(journal, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        config = load_cycle_config(args.config)
        print(json.dumps({"status": "inputs_checked", "identity": config["identity"]}))
        return
    state = run_cycle(args.config, args.output_dir, resume=args.resume)
    print(json.dumps({"status": state["status"], "selected": state["selected"]}), flush=True)


if __name__ == "__main__":
    main()
