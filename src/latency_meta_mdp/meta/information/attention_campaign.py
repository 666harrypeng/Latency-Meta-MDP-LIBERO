"""Version-two recipes using the shared collection, checkpoint and campaign journal."""

import json
import math
import os
import shutil
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
from latency_meta_mdp.meta.information.validation import (
    BUDGET_SELECTION,
    DELAYED_SELECTION,
    atomic_json,
    validation_rank,
    verified_validation_metrics,
)


def curriculum_sampling(spec, cycle):
    """Resolve the declared schedule at an absolute completed-cohort boundary."""
    required = {
        "total_cycles",
        "start",
        "end",
        "curvature",
        "latest_fraction_cap",
        "latest_online_fraction",
        "oversampling_cap",
    }
    if set(spec) != required or not all(math.isfinite(v) for v in spec.values()):
        raise ValueError("invalid replay curriculum")
    total = spec["total_cycles"]
    if (
        type(total) is not int
        or total < 2
        or type(cycle) is not int
        or not 1 <= cycle <= total
        or not 0 <= spec["start"] <= spec["end"] < 1
        or not 0 <= spec["curvature"] <= 10
    ):
        raise ValueError("invalid replay curriculum range")
    u = (cycle - 1) / (total - 1)
    k = spec["curvature"]
    fraction = math.expm1(k * u) / math.expm1(k) if k else u
    return dict(
        online_fraction=spec["start"] + (spec["end"] - spec["start"]) * fraction,
        **{
            k: spec[k]
            for k in ("latest_fraction_cap", "latest_online_fraction", "oversampling_cap")
        },
    )


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
    *,
    journal,
    recipe,
    initial,
    initial_metrics,
    collect,
    fit,
    startup_result=None,
    retention=None,
    stop_after_cycles=None,
    selection=DELAYED_SELECTION,
):
    """Advance after the configured validation boundaries are complete."""
    if recipe["reward_regression_policy"] != "report_only":
        raise ValueError("v2 requires explicit report_only reward regression policy")
    parent, best, best_metrics = initial, initial, initial_metrics
    online, results = list(recipe.get("initial_online_sources", [])), []
    if stop_after_cycles is not None and (
        type(stop_after_cycles) is not int or stop_after_cycles < 1
    ):
        raise ValueError("positive cycle stop required")

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

    definitions = recipe["cycles"][:stop_after_cycles]
    for index, definition in enumerate(definitions, 1 + recipe.get("cycle_offset", 0)):
        if definition["generation"] is not None:
            if retention is not None and "result" not in journal.state["phases"].get(
                f"{index}/collect", {}
            ):
                retention.check_storage()
            collected = journal.run(f"{index}/collect", lambda: collect(index, definition, parent))
            online.append(collected["source"])
        active = [
            {**s, "group": "latest" if i == len(online) - 1 else "old"}
            for i, s in enumerate(online)
        ]
        if retention is not None:
            active = retention.select(index, collected["source"])
        sources = [
            *recipe["startup_sources"],
            *active,
        ]
        fitted = journal.run(
            f"{index}/fit_validate", lambda: fit(index, definition, parent, sources)
        )
        for candidate in fitted["candidates"]:
            metrics = candidate["metrics"]
            if best_metrics is None or validation_rank(metrics, selection) > validation_rank(
                best_metrics, selection
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
        if retention is not None:
            retention.cleanup(index)
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
    report["status"] = (
        "complete" if len(definitions) == len(recipe["cycles"]) else "bounded_complete"
    )
    atomic_json(journal.directory / "report.json", report)
    return report


def validate_recipe(config, recipe, audit, *, root):
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest

    if config["schema_version"] == 3 and (
        recipe.get("initial_checkpoint")
        or recipe.get("cycle_offset", 0)
        or recipe.get("initial_online_sources")
    ):
        raise ValueError("protocol 3 campaigns start from scratch; resume the original journal")
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
    if (
        set(recipe) - {"retention", "cycle_offset", "initial_online_sources", "replay_curriculum"}
        != required
        or recipe["schema_version"] != 2
        or not recipe["startup_sources"]
    ):
        raise ValueError("invalid attention campaign recipe fields")
    offset = recipe.get("cycle_offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("invalid cycle offset")
    if any(s["group"] not in ("old", "latest") for s in recipe.get("initial_online_sources", [])):
        raise ValueError("initial online sources must be online")
    if "replay_curriculum" in recipe:
        curriculum_sampling(recipe["replay_curriculum"], offset + 1)
        curriculum_sampling(recipe["replay_curriculum"], offset + len(recipe["cycles"]))
    if "retention" in recipe:
        from latency_meta_mdp.meta.information.replay_retention import validate_retention

        validate_retention(recipe["retention"])
        if offset != len(recipe.get("initial_online_sources", [])):
            raise ValueError("retention offset requires every preceding online source")
        if any(d["generation"] is None for d in recipe["cycles"]):
            raise ValueError("bounded replay requires collection in every online cycle")
    if recipe["reward_regression_policy"] != "report_only":
        raise ValueError("attention reward regression policy must be report_only")
    if any(
        type(recipe[k]) is not int or recipe[k] < (1 if k == "global_batch" else 0)
        for k in ("global_batch", "seed")
    ):
        raise ValueError("invalid campaign batch/seed")
    if any(s["group"] != "startup" for s in recipe["startup_sources"]):
        raise ValueError("startup sources must use startup sampling group")
    if set(recipe["startup"]) - {"validation_every"} != {"fitted_iterations", "optimizer_updates"}:
        raise ValueError("explicit startup target and optimizer budgets required")
    planned = []
    for i, definition in enumerate([recipe["startup"], *recipe["cycles"]]):
        expected = {"fitted_iterations", "optimizer_updates"} | ({"generation"} if i else set())
        if set(definition) - {"validation_every"} != expected or any(
            type(definition[k]) is not int
            or definition[k] < (0 if i == 0 and k == "fitted_iterations" else 1)
            for k in ("fitted_iterations", "optimizer_updates")
        ):
            raise ValueError("invalid fitted-iteration/update budget")
        cadence = definition.get("validation_every", 1)
        if type(cadence) is not int or cadence < 1:
            raise ValueError("invalid validation cadence")
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
            + sum(
                (d["fitted_iterations"] + d.get("validation_every", 1) - 1)
                // d.get("validation_every", 1)
                for d in [recipe["startup"], *recipe["cycles"]]
            )
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
    compile_model=False,
    checkpoint_every_updates=256,
    dry_run=False,
    rollout_workers_per_gpu=1,
    validation_workers_per_gpu=None,
    validation_gpus=None,
    learner_cpu_threads=None,
    validation_cpu_threads=None,
    compact_replay=False,
    stop_after_cycles=None,
    episode_cache_size=None,
    min_free_gib=32,
):
    from latency_meta_mdp.meta.information.attention_training import (
        initialize_attention,
        run_attention_training,
    )
    from latency_meta_mdp.meta.information.parallel import run_parallel
    from latency_meta_mdp.meta.information.resources import interrupt_on_termination, worker_lock

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    config, recipe = load_config(config_path), json.loads(Path(recipe_path).read_text())
    validation_gpus = validation_gpus or gpus
    validation_workers_per_gpu = (
        rollout_workers_per_gpu
        if validation_workers_per_gpu is None
        else validation_workers_per_gpu
    )
    learner_cpu_threads = cpu_threads if learner_cpu_threads is None else learner_cpu_threads
    validation_cpu_threads = (
        cpu_threads if validation_cpu_threads is None else validation_cpu_threads
    )
    if any(
        type(v) is not int or v < 1
        for v in (
            rollout_workers_per_gpu,
            validation_workers_per_gpu,
            cpu_threads,
            learner_cpu_threads,
            validation_cpu_threads,
        )
    ):
        raise ValueError("positive worker counts and CPU threads required")
    if (
        not math.isfinite(min_free_gib)
        or min_free_gib < 0
        or (
            episode_cache_size is not None
            and (type(episode_cache_size) is not int or episode_cache_size < 1)
        )
    ):
        raise ValueError("invalid free-space reserve or episode cache capacity")
    audit = json.loads(Path(audit_path).read_text())
    planned = validate_recipe(config, recipe, audit, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output_dir / "planned.json",
        dict(**planned, config_hash=identity_hash(config), recipe=recipe),
    )
    if dry_run:
        return planned
    for selected in (gpus, training_gpus, validation_gpus):
        if (
            not selected
            or any(not d.isdigit() for d in selected.split(","))
            or len(set(selected.split(","))) != len(selected.split(","))
        ):
            raise ValueError("explicit distinct rollout and learner GPUs required")
    from latency_meta_mdp.meta.information.runtime_identity import source_identity

    source = source_identity(root)
    atomic_json(output_dir / "runtime-source.json", source)
    resources = dict(
        host=os.uname().nodename,
        gpus=gpus,
        training_gpus=training_gpus,
        validation_gpus=validation_gpus,
        rollout_workers_per_gpu=rollout_workers_per_gpu,
        validation_workers_per_gpu=validation_workers_per_gpu,
        cpu_threads=cpu_threads,
        learner_cpu_threads=learner_cpu_threads,
        validation_cpu_threads=validation_cpu_threads,
        microbatch=microbatch,
        compile_model=compile_model,
        replay_prepare_workers=replay_prepare_workers,
        compact_replay=compact_replay,
        episode_cache_size=episode_cache_size,
        min_free_gib=min_free_gib,
    )
    atomic_json(output_dir / "resource-config.json", resources)
    with (output_dir / "resource-history.jsonl").open("a") as stream:
        stream.write(json.dumps(resources) + "\n")
    print("CAMPAIGN_RESOURCES " + json.dumps(resources), flush=True)
    bootstrap = (
        (root / recipe["validation_bootstrap"]).resolve() if config["schema_version"] != 3 else None
    )
    binding = dict(
        recipe=recipe,
        initial_checkpoint=(
            checkpoint_artifacts(root / recipe["initial_checkpoint"])
            if recipe["initial_checkpoint"]
            else None
        ),
        config_sha256=sha256(config_path),
        audit_sha256=sha256(audit_path),
        code_hash=source["code_hash"],
        bootstrap={}
        if config["schema_version"] == 3
        else {
            str(s): sha256(bootstrap / f"seed-{s}.npz") for s in config["splits"]["validation"][:3]
        },
    )
    initial_metrics = None
    if recipe["initial_checkpoint"]:
        parent = (root / recipe["initial_checkpoint"]).resolve()
        state_path = parent.parent / "validation-state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            number = str(json.loads((parent / "complete.json").read_text())["iteration"])
            if number in state["completed"]:
                evaluation = parent.parent / "validation" / parent.name
                initial_metrics = verified_validation_metrics(
                    parent, evaluation, state["completed"][number], state["protocol"]
                )
                binding["initial_validation"] = dict(
                    state_sha256=sha256(state_path),
                    evaluation_sha256=sha256(evaluation / "rollout-summary.json"),
                )

    def fit_directory(directory, definition, parent, sources, *, initial=False, sampling=None):
        from latency_meta_mdp.meta.online_tracking import recover_events

        if shutil.disk_usage(output_dir).free < min_free_gib * 2**30:
            raise RuntimeError(
                "insufficient free space before fitting; data and checkpoint retained"
            )
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
            compile_model=compile_model,
            checkpoint_every_updates=checkpoint_every_updates,
            cpu_threads=learner_cpu_threads,
            replay_cache_dir=replay_cache_dir or output_dir / "replay-cache",
            replay_prepare_workers=replay_prepare_workers,
            validation_bootstrap=bootstrap,
            validation_gpus=validation_gpus,
            validation_workers_per_gpu=validation_workers_per_gpu,
            validation_cpu_threads=validation_cpu_threads,
            audit_path=audit_path,
            worker_python=worker_python,
            validate_initial=initial,
            validation_every=definition.get("validation_every", 1),
            episode_cache_size=(
                episode_cache_size
                if episode_cache_size is not None
                else recipe.get("retention", {}).get("episode_cache_size", 128)
            ),
            replay_cache_max_bytes=recipe.get("retention", {}).get("max_cache_bytes"),
            compact_replay=compact_replay,
            sampling=sampling,
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
        if shutil.disk_usage(output_dir).free < min_free_gib * 2**30:
            raise RuntimeError("insufficient free space before collection; checkpoint retained")
        directory = output_dir / f"cycle-{index:04d}"
        directory.mkdir(exist_ok=True)
        collection = directory / "collection"
        generation_path = directory / "generation.json"
        generation = definition["generation"]
        atomic_json(generation_path, generation)
        if config["schema_version"] != 3 and not all(
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
            workers_per_gpu=rollout_workers_per_gpu,
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
        sampling = (
            curriculum_sampling(recipe["replay_curriculum"], index)
            if "replay_curriculum" in recipe
            else None
        )
        directory = output_dir / f"cycle-{index:04d}"
        atomic_json(
            directory / "cycle-recipe.json",
            dict(
                cycle=index,
                definition=definition,
                global_batch=recipe["global_batch"],
                sampling=sampling,
            ),
        )
        return fit_directory(directory / "fit", definition, parent, sources, sampling=sampling)

    with ExitStack() as stack:
        stack.enter_context(interrupt_on_termination())
        stack.enter_context(worker_lock(output_dir))
        journal = CampaignJournal(output_dir, binding=binding, allow_retired="retention" in recipe)
        retention = None
        if "retention" in recipe:
            from latency_meta_mdp.meta.information.replay_retention import BoundedReplay

            retention = BoundedReplay(
                output_dir, recipe["retention"], seed=recipe["seed"], cache_dir=replay_cache_dir
            )
            retention.check_storage()
            for index, source in enumerate(recipe.get("initial_online_sources", []), 1):
                retention.select(index, source)
        initial = (
            (root / recipe["initial_checkpoint"]).resolve()
            if recipe["initial_checkpoint"]
            else initialize_attention(
                config, output_dir / "initial", root=root, seed=recipe["seed"]
            )
        )
        selection = BUDGET_SELECTION if config["schema_version"] == 3 else DELAYED_SELECTION
        startup = None
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
                startup["candidates"], key=lambda c: validation_rank(c["metrics"], selection)
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
            selection=selection,
            retention=retention,
            stop_after_cycles=stop_after_cycles,
        )
        result["planned_budget"] = planned
        atomic_json(output_dir / "report.json", result)
        return result
