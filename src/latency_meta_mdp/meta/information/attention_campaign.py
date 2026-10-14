"""Version-two recipes using the shared collection, checkpoint and campaign journal."""

import json
import sys
from contextlib import ExitStack
from pathlib import Path

from latency_meta_mdp.meta.information.campaign import (
    CampaignJournal,
    _bootstrap,
    _cohort_artifacts,
    _link,
)
from latency_meta_mdp.meta.information.config import identity_hash, load_config, sha256
from latency_meta_mdp.meta.information.validation import atomic_json, verified_validation_metrics


def checkpoint_artifacts(checkpoint):
    """Pin both the receipt and its weights before a resumable campaign starts."""
    checkpoint = Path(checkpoint).resolve()
    receipt_path = checkpoint / "complete.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["files"] != {"state.pt": sha256(checkpoint / "state.pt")}:
        raise ValueError("checkpoint artifact differs from its receipt")
    return {
        str(receipt_path): sha256(receipt_path),
        str(checkpoint / "state.pt"): receipt["files"]["state.pt"],
    }


def attention_cycle_loop(
    *, journal, recipe, initial, initial_metrics, collect, fit, startup_result=None
):
    """Advance only after every fitted iteration has complete validation."""
    if recipe["reward_regression_policy"] != "report_only":
        raise ValueError("v2 requires explicit report_only reward regression policy")
    parent, best, best_metrics = initial, initial, initial_metrics
    online, results = [], []

    def totals():
        work = ([startup_result] if startup_result is not None else []) + results
        return dict(
            startup=startup_result,
            optimizer_updates=sum(r["optimizer_updates"] for r in work),
            observed_optimizer_updates=sum(
                r.get("observed_optimizer_updates", r["optimizer_updates"]) for r in work
            ),
            validation_episodes=sum(r["validation_episodes"] for r in work),
        )

    for index, definition in enumerate(recipe["cycles"], 1):
        if definition["generation"] is not None:
            collected = journal.run(f"{index}/collect", lambda: collect(index, definition, parent))
            online.append(collected["source"])
        sources = [
            *recipe["startup_sources"],
            *[
                {**s, "group": "latest" if i == len(online) - 1 else "old"}
                for i, s in enumerate(online)
            ],
        ]
        fitted = journal.run(
            f"{index}/fit_validate", lambda: fit(index, definition, parent, sources)
        )
        for candidate in fitted["candidates"]:
            metrics = candidate["metrics"]
            if best_metrics is None or round(60 * metrics["selection_score"]) > round(
                60 * best_metrics["selection_score"]
            ):
                best, best_metrics = Path(candidate["checkpoint"]), metrics
        parent = Path(fitted["latest"])
        _link(journal.directory, "latest", parent)
        _link(journal.directory, "best", best)
        results.append(fitted)
        report = dict(
            status="running",
            best=str(best),
            best_metrics=best_metrics,
            latest=str(parent),
            cycles=results,
            reward_regression_policy="report_only",
            **totals(),
        )
        atomic_json(journal.directory / "report.json", report)
    if not results:
        report = dict(
            status="running",
            best=str(best),
            best_metrics=best_metrics,
            latest=str(parent),
            cycles=[],
            **totals(),
            reward_regression_policy="report_only",
        )
    report["status"] = "complete"
    atomic_json(journal.directory / "report.json", report)
    return report


def validate_recipe(config, recipe, audit, *, root):
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest

    required = {
        "schema_version",
        "startup_sources",
        "initial_checkpoint",
        "validation_bootstrap",
        "startup",
        "cycles",
        "global_batch",
        "seed",
        "reward_regression_policy",
    }
    if set(recipe) != required or recipe["schema_version"] != 2 or not recipe["startup_sources"]:
        raise ValueError("invalid attention campaign recipe fields")
    if recipe["reward_regression_policy"] != "report_only":
        raise ValueError("attention reward regression policy must be report_only")
    if any(
        type(recipe[k]) is not int or recipe[k] < (1 if k == "global_batch" else 0)
        for k in ("global_batch", "seed")
    ):
        raise ValueError("invalid campaign batch/seed")
    if any(s["group"] != "startup" for s in recipe["startup_sources"]):
        raise ValueError("startup sources must use startup sampling group")
    if set(recipe["startup"]) != {"fitted_iterations", "optimizer_updates"}:
        raise ValueError("explicit startup target and optimizer budgets required")
    planned = []
    for i, definition in enumerate([recipe["startup"], *recipe["cycles"]]):
        expected = {"fitted_iterations", "optimizer_updates"} | ({"generation"} if i else set())
        if set(definition) != expected or any(
            type(definition[k]) is not int
            or definition[k] < (0 if i == 0 and k == "fitted_iterations" else 1)
            for k in ("fitted_iterations", "optimizer_updates")
        ):
            raise ValueError("invalid fitted-iteration/update budget")
        if i and definition["generation"] is not None:
            manifest = build_cycle_manifest(
                config,
                audit,
                root=root,
                code_hash="planned",
                policy={},
                phase="online",
                baselines=False,
                generation=definition["generation"],
            )
            planned.extend(manifest["episodes"])
    if len({e["episode_id"] for e in planned}) != len(planned):
        raise ValueError("duplicate planned generation episode IDs")
    return dict(
        episodes=planned,
        collection_episodes=len(planned),
        optimizer_updates=sum(
            d["fitted_iterations"] * d["optimizer_updates"]
            for d in [recipe["startup"], *recipe["cycles"]]
        ),
        target_refreshes=sum(
            d["fitted_iterations"] for d in [recipe["startup"], *recipe["cycles"]]
        ),
        validation_episodes=9
        * (
            int(recipe["startup"]["fitted_iterations"] > 0)
            + sum(d["fitted_iterations"] for d in [recipe["startup"], *recipe["cycles"]])
        ),
    )


def run_attention_campaign(
    config_path,
    output_dir,
    *,
    root,
    recipe_path,
    audit_path,
    gpus,
    training_gpus,
    worker_python=None,
    cpu_threads=4,
    replay_cache_dir=None,
    replay_prepare_workers=2,
    microbatch=2,
    bf16=False,
    dry_run=False,
):
    from latency_meta_mdp.meta.information.attention_training import (
        initialize_attention,
        run_attention_training,
    )
    from latency_meta_mdp.meta.information.parallel import run_parallel
    from latency_meta_mdp.meta.information.resources import interrupt_on_termination, worker_lock

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    config, recipe = load_config(config_path), json.loads(Path(recipe_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    planned = validate_recipe(config, recipe, audit, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output_dir / "planned.json",
        dict(**planned, config_hash=identity_hash(config), recipe=recipe),
    )
    if dry_run:
        return planned
    for selected in (gpus, training_gpus):
        if (
            not selected
            or any(not d.isdigit() for d in selected.split(","))
            or len(set(selected.split(","))) != len(selected.split(","))
        ):
            raise ValueError("explicit distinct rollout and learner GPUs required")
    code = {
        str(p.relative_to(root)): sha256(p)
        for folder in ("src/latency_meta_mdp", "scripts")
        for p in sorted((root / folder).rglob("*.py"))
    }
    bootstrap = (root / recipe["validation_bootstrap"]).resolve()
    binding = dict(
        recipe=recipe,
        initial_checkpoint=(
            checkpoint_artifacts(root / recipe["initial_checkpoint"])
            if recipe["initial_checkpoint"]
            else None
        ),
        config_sha256=sha256(config_path),
        audit_sha256=sha256(audit_path),
        code_hash=identity_hash(code),
        bootstrap={
            str(s): sha256(bootstrap / f"seed-{s}.npz") for s in config["splits"]["validation"][:3]
        },
    )

    def fit_directory(directory, definition, parent, sources, *, initial=False):
        from latency_meta_mdp.meta.online_tracking import recover_events

        result = run_attention_training(
            config_path,
            sources,
            directory,
            root=root,
            checkpoint=parent,
            seed=recipe["seed"],
            total_iterations=definition["fitted_iterations"],
            optimizer_updates=definition["optimizer_updates"],
            global_batch=recipe["global_batch"],
            training_gpus=training_gpus,
            microbatch=microbatch,
            bf16=bf16,
            cpu_threads=cpu_threads,
            replay_cache_dir=replay_cache_dir or output_dir / "replay-cache",
            replay_prepare_workers=replay_prepare_workers,
            validation_bootstrap=bootstrap,
            validation_gpus=gpus,
            audit_path=audit_path,
            worker_python=worker_python,
            validate_initial=initial,
        )
        state = json.loads((directory / "validation-state.json").read_text())
        candidates, artifacts = [], {}
        for iteration, item in sorted(state["completed"].items(), key=lambda x: int(x[0])):
            checkpoint = directory / f"iteration-{int(iteration):04d}"
            evaluation = directory / f"validation/iteration-{int(iteration):04d}"
            metrics = verified_validation_metrics(checkpoint, evaluation, item, state["protocol"])
            candidates.append(dict(checkpoint=str(checkpoint), metrics=metrics))
            for p in (
                checkpoint / "state.pt",
                checkpoint / "complete.json",
                directory / f"validation/iteration-{int(iteration):04d}/rollout-summary.json",
            ):
                artifacts[str(p)] = sha256(p)
            artifacts.update(
                _cohort_artifacts(directory / f"validation/iteration-{int(iteration):04d}")
            )
        return dict(
            artifacts=artifacts,
            latest=str(directory / f"iteration-{result['completed_iterations']:04d}"),
            candidates=candidates,
            optimizer_updates=definition["fitted_iterations"] * definition["optimizer_updates"],
            observed_optimizer_updates=sum(
                item["event"]["phase"] == "fit"
                for item in recover_events(directory / "events.jsonl")
            ),
            validation_episodes=9 * len(candidates),
            training=result,
        )

    def collect(index, definition, parent):
        directory = output_dir / f"cycle-{index:04d}"
        directory.mkdir(exist_ok=True)
        collection = directory / "collection"
        generation_path = directory / "generation.json"
        generation = definition["generation"]
        atomic_json(generation_path, generation)
        if not all(
            (collection / "bootstrap" / f"seed-{s}.npz").exists() for s in generation["scenes"]
        ):
            command = [
                str(worker_python or sys.executable),
                "-u",
                str(root / "scripts/run_conveyor_meta.py"),
                "cycle-bootstrap",
                "--config",
                str(Path(config_path).resolve()),
                "--output-dir",
                str(collection),
                "--scene-audit",
                str(Path(audit_path).resolve()),
                "--checkpoint",
                str(parent),
                "--phase",
                "online",
                "--meta-only",
                "--generation",
                str(generation_path),
            ]
            _bootstrap(
                command,
                root=root,
                directory=directory,
                device=gpus.split(",")[0],
                cpu_threads=cpu_threads,
            )
        result = run_parallel(
            config_path,
            collection,
            root=root,
            audit_path=audit_path,
            checkpoint=parent,
            phase="online",
            bootstrap_dir=collection / "bootstrap",
            gpus=gpus,
            worker_python=worker_python,
            cpu_threads=cpu_threads,
            baselines=False,
            generation=generation,
        )
        return dict(
            artifacts=_cohort_artifacts(collection),
            collection=result,
            source=dict(
                path=str(collection),
                manifest_sha256=sha256(collection / "cohort.json"),
                group="latest",
            ),
        )

    def fit(index, definition, parent, sources):
        return fit_directory(output_dir / f"cycle-{index:04d}/fit", definition, parent, sources)

    with ExitStack() as stack:
        stack.enter_context(interrupt_on_termination())
        stack.enter_context(worker_lock(output_dir))
        journal = CampaignJournal(output_dir, binding=binding)
        initial = (
            (root / recipe["initial_checkpoint"]).resolve()
            if recipe["initial_checkpoint"]
            else initialize_attention(
                config, output_dir / "initial", root=root, seed=recipe["seed"]
            )
        )
        initial_metrics, startup = None, None
        if recipe["startup"]["fitted_iterations"]:
            startup = journal.run(
                "startup/fit_validate",
                lambda: fit_directory(
                    output_dir / "startup",
                    recipe["startup"],
                    initial,
                    recipe["startup_sources"],
                    initial=True,
                ),
            )
            selected = max(
                startup["candidates"], key=lambda c: round(60 * c["metrics"]["selection_score"])
            )
            initial, initial_metrics = Path(selected["checkpoint"]), selected["metrics"]
        result = attention_cycle_loop(
            journal=journal,
            recipe=recipe,
            initial=initial,
            initial_metrics=initial_metrics,
            collect=collect,
            fit=fit,
            startup_result=startup,
        )
        result["planned_budget"] = planned
        atomic_json(output_dir / "report.json", result)
        return result
