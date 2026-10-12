"""Bounded synchronous collection/refit campaigns over explicit immutable recipes."""

import json
import subprocess
import sys
import time
from pathlib import Path

from latency_meta_mdp.meta.information.config import (
    identity_hash,
    load_config,
    sha256,
    validate_binding,
)
from latency_meta_mdp.meta.information.validation import atomic_json, validation_metrics


def _verify_artifacts(result):
    for name, digest in result["artifacts"].items():
        if sha256(name) != digest:
            raise ValueError(f"campaign artifact hash differs: {name}")


class CampaignJournal:
    """Only complete hash-verified phases can be skipped after interruption."""

    def __init__(self, directory, *, binding):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "campaign-state.json"
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else dict(binding=binding, phases={})
        )
        validate_binding(binding, self.state["binding"])
        atomic_json(self.path, self.state)

    def run(self, name, operation):
        phase = self.state["phases"].setdefault(name, dict(attempts=[]))
        if "result" in phase:
            _verify_artifacts(phase["result"])
            return phase["result"]
        if phase["attempts"] and phase["attempts"][-1]["status"] == "running":
            phase["attempts"][-1]["status"] = "interrupted"
        attempt = dict(status="running", started_at=time.time())
        phase["attempts"].append(attempt)
        atomic_json(self.path, self.state)
        started = time.perf_counter()
        print(f"CAMPAIGN_PHASE {name}", flush=True)
        try:
            result = operation()
            _verify_artifacts(result)
            phase["result"] = result
            attempt["status"] = "complete"
            return result
        except BaseException as error:
            attempt.update(status="failed", error=repr(error))
            raise
        finally:
            attempt.update(finished_at=time.time(), wall_seconds=time.perf_counter() - started)
            atomic_json(self.path, self.state)


def selection_decision(initial, incumbent, candidate, *, drop_successes, repeat=None):
    loss = round(60 * initial["selection_score"]) - round(60 * candidate["selection_score"])
    if loss >= drop_successes:
        if repeat is None:
            return "repeat"
        if (
            round(60 * initial["selection_score"]) - round(60 * repeat["selection_score"])
            >= drop_successes
        ):
            return "stop"
    return (
        "best"
        if round(60 * candidate["selection_score"]) > round(60 * incumbent["selection_score"])
        else "keep"
    )


def _link(directory, role, checkpoint):
    temporary = directory / f".{role}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(Path(checkpoint).resolve(), target_is_directory=True)
    temporary.replace(directory / role)


def _cohort_artifacts(directory):
    from latency_meta_mdp.meta.information.cohort import verify_complete

    manifest_path = directory / "cohort.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = {str(manifest_path): sha256(manifest_path)}
    for ep in manifest["episodes"]:
        attempts = [
            p
            for p in (directory / "episodes" / ep["episode_id"]).glob("attempt-*")
            if verify_complete(p, binding=dict(cohort=manifest["binding"], episode=ep))
        ]
        if len(attempts) != 1:
            raise ValueError("campaign collection must have exactly one complete attempt")
        path = attempts[0] / "complete.json"
        receipt = json.loads(path.read_text())
        artifacts[str(path)] = sha256(path)
        artifacts.update(
            {str(path.parent / name): digest for name, digest in receipt["files"].items()}
        )
    return artifacts


def _metrics(directory, scenes):
    path = directory / "rollout-summary.json"
    return validation_metrics(json.loads(path.read_text())["episodes"], scenes=scenes)


def _verified_reference(path, manifest, scenes):
    artifacts = _cohort_artifacts(path.parent)
    expected_rows = []
    for ep in manifest["episodes"]:
        directory = path.parent / "episodes" / ep["episode_id"]
        attempt = next(
            p for p in directory.glob("attempt-*") if str(p / "complete.json") in artifacts
        )
        expected_rows.append(
            dict(episode=ep, report=json.loads((attempt / "summary.json").read_text()))
        )
    rows = [
        dict(episode=r["episode"], report=r["report"])
        for r in json.loads(path.read_text())["episodes"]
    ]
    validate_binding(
        sorted(expected_rows, key=lambda r: r["episode"]["episode_id"]),
        sorted(rows, key=lambda r: r["episode"]["episode_id"]),
    )
    return validation_metrics(expected_rows, scenes=scenes), artifacts


def _bootstrap(command, *, root, directory, device, cpu_threads):
    from latency_meta_mdp.meta.information.parallel import worker_environment
    from latency_meta_mdp.meta.information.resources import worker_lock

    with worker_lock(Path.home() / ".cache/conveyor-meta/gpu-locks" / device):
        with (directory / "bootstrap.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=worker_environment(device, cpu_threads=cpu_threads),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                if process.wait() != 0:
                    raise RuntimeError(f"bootstrap failed: {directory / 'bootstrap.log'}")
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run_campaign(
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
    replay_prepare_workers=4,
):
    """Scientific cohorts/budgets come from recipe JSON; execution resources are arguments."""
    from contextlib import ExitStack

    from latency_meta_mdp.meta.information.cohort import build_manifest
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest
    from latency_meta_mdp.meta.information.deployment import (
        deployment_binding,
        load_scheduler,
        scheduler_identity,
    )
    from latency_meta_mdp.meta.information.online_training import (
        run_online_training,
        validate_mixture,
    )
    from latency_meta_mdp.meta.information.parallel import run_parallel
    from latency_meta_mdp.meta.information.resources import interrupt_on_termination, worker_lock

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    recipe = json.loads(Path(recipe_path).read_text())
    required = {
        "initial_checkpoint",
        "behavior_checkpoint",
        "offline_dir",
        "online_dirs",
        "validation_bootstrap",
        "initial_validation",
        "cycles",
        "global_batch",
        "seed",
        "drop_successes",
        "compatibility",
    }
    if set(recipe) != required or not recipe["cycles"] or not recipe["online_dirs"]:
        raise ValueError("invalid campaign recipe fields")
    if (
        type(recipe["global_batch"]) is not int
        or recipe["global_batch"] < 1
        or type(recipe["seed"]) is not int
        or type(recipe["drop_successes"]) is not int
        or recipe["drop_successes"] < 1
    ):
        raise ValueError("invalid campaign update/stopping budget")
    for cycle in recipe["cycles"]:
        if set(cycle) != {"generation", "optimizer_updates"}:
            raise ValueError("cycle requires explicit generation and update budget")
        budget = cycle["optimizer_updates"]
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("invalid cycle optimizer budget")
        if cycle["generation"] is None and budget is None:
            raise ValueError("fixed-data control requires an explicit update budget")
    for selection in (gpus, training_gpus):
        devices = selection.split(",") if isinstance(selection, str) else []
        if (
            not devices
            or any(not d.isdigit() for d in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ValueError("campaign GPUs must be explicit distinct physical indices")

    def resolve(path):
        return (root / path).resolve()

    initial = resolve(recipe["initial_checkpoint"])
    offline = resolve(recipe["offline_dir"])
    online = [resolve(p) for p in recipe["online_dirs"]]
    bootstrap = resolve(recipe["validation_bootstrap"])
    config = load_config(config_path)
    audit = json.loads(Path(audit_path).read_text())
    compatibility_path = resolve(recipe["compatibility"])
    compatibility = json.loads(compatibility_path.read_text())
    manifests = [json.loads((p / "cohort.json").read_text()) for p in [offline, *online]]
    validate_mixture(config, manifests, compatibility)
    validate_binding(
        build_manifest(config, audit, root=root, code_hash=manifests[0]["binding"]["code_hash"]),
        manifests[0],
    )
    for manifest in manifests[1:]:
        bound = manifest["binding"]
        validate_binding(
            build_cycle_manifest(
                config,
                audit,
                root=root,
                code_hash=bound["code_hash"],
                policy=bound["policy"],
                phase="online",
                baselines=bound["baselines"],
                generation=bound.get("generation"),
            ),
            manifest,
        )
    known_ids = {e["episode_id"] for m in manifests for e in m["episodes"]}
    known_scenes = {e["scene"] for m in manifests for e in m["episodes"]}
    for definition in recipe["cycles"]:
        if definition["generation"] is None:
            continue
        planned = build_cycle_manifest(
            config,
            audit,
            root=root,
            code_hash="planned",
            policy={},
            phase="online",
            baselines=False,
            generation=definition["generation"],
        )
        ids = {e["episode_id"] for e in planned["episodes"]}
        if known_ids & ids:
            raise ValueError("campaign generations overlap existing or planned training episodes")
        new_scenes = {e["scene"] for e in planned["episodes"]}
        if known_scenes & new_scenes:
            raise ValueError("campaign generations overlap existing or planned training scenes")
        known_ids.update(ids)
        known_scenes.update(new_scenes)
    expected = deployment_binding(config)
    initial_policy = scheduler_identity(load_scheduler(initial, expected))
    behavior = resolve(recipe["behavior_checkpoint"])
    validate_binding(
        scheduler_identity(
            load_scheduler(
                behavior,
                expected,
                source_manifest=manifests[1]["binding"]["policy"].get("source_manifest"),
            )
        ),
        manifests[1]["binding"]["policy"],
    )
    scenes = config["splits"]["validation"][:3]
    reference_path = resolve(recipe["initial_validation"])
    reference_manifest_path = reference_path.parent / "cohort.json"
    reference_manifest = json.loads(reference_manifest_path.read_text())
    for key in ("receipt_sha256", "model_hash"):
        validate_binding(initial_policy[key], reference_manifest["binding"]["policy"][key])
    validate_binding(identity_hash(config), reference_manifest["binding"]["config_hash"])
    validate_binding(
        build_cycle_manifest(
            config,
            audit,
            root=root,
            code_hash=reference_manifest["binding"]["code_hash"],
            policy=reference_manifest["binding"]["policy"],
            phase="monitor",
            baselines=False,
        ),
        reference_manifest,
    )
    reference, reference_artifacts = _verified_reference(reference_path, reference_manifest, scenes)
    code = {
        str(p.relative_to(root)): sha256(p) for p in (root / "src/latency_meta_mdp").rglob("*.py")
    }
    code.update({str(p.relative_to(root)): sha256(p) for p in (root / "scripts").rglob("*.py")})
    inputs = [
        Path(config_path),
        Path(audit_path),
        compatibility_path,
        initial / "complete.json",
        initial / "state.pt",
        behavior / "complete.json",
        behavior / "state.pt",
        reference_path,
        reference_manifest_path,
        *(bootstrap / f"seed-{s}.npz" for s in scenes),
        offline / "cohort.json",
        *(p / "cohort.json" for p in online),
    ]
    binding = dict(
        recipe=recipe,
        inputs={str(p.resolve()): sha256(p) for p in inputs},
        reference_artifacts=reference_artifacts,
        code_hash=identity_hash(code),
    )
    with ExitStack() as stack:
        stack.enter_context(interrupt_on_termination())
        stack.enter_context(worker_lock(output_dir))
        journal = CampaignJournal(output_dir, binding=binding)
        parent, best, best_metrics = initial, initial, reference
        _link(output_dir, "best", best)
        results = []
        for index, definition in enumerate(recipe["cycles"], 1):
            directory = output_dir / f"cycle-{index:04d}"
            directory.mkdir(exist_ok=True)
            generation = definition["generation"]
            if generation is not None:
                collection = directory / "collection"
                generation_path = directory / "generation.json"
                atomic_json(generation_path, generation)

                def collect():
                    if not all(
                        (collection / "bootstrap" / f"seed-{s}.npz").exists()
                        for s in generation["scenes"]
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
                    return dict(artifacts=_cohort_artifacts(collection), collection=result)

                journal.run(f"{index}/collect", collect)
                online.append(collection)
            manifests = [json.loads((p / "cohort.json").read_text()) for p in [offline, *online]]
            admitted_hashes = set(compatibility["collection_code_hashes"]) | {binding["code_hash"]}
            if any(m["binding"]["code_hash"] not in admitted_hashes for m in manifests):
                raise ValueError("unreviewed collector code in campaign replay")
            compatibility_path = directory / "collector-compatibility.json"
            atomic_json(
                compatibility_path,
                dict(
                    config_hash=identity_hash(config),
                    collection_code_hashes=sorted({m["binding"]["code_hash"] for m in manifests}),
                    rationale=(
                        "Previously reviewed collectors plus this pinned campaign source; "
                        "unchanged public packet, reward, masks and frozen assets."
                    ),
                ),
            )
            fit = directory / "fit"

            def update():
                result = run_online_training(
                    config_path,
                    offline,
                    online[0],
                    fit,
                    root=root,
                    checkpoint=resolve(recipe["behavior_checkpoint"]),
                    warm_start_checkpoint=parent,
                    compatibility_path=compatibility_path,
                    audit_path=audit_path,
                    seed=recipe["seed"] + index - 1,
                    validation_bootstrap=bootstrap,
                    validation_gpus=gpus,
                    training_gpus=training_gpus,
                    worker_python=worker_python,
                    cpu_threads=cpu_threads,
                    global_batch=recipe["global_batch"],
                    replay_storage="mmap",
                    replay_cache_dir=replay_cache_dir or output_dir / "replay-cache",
                    replay_prepare_workers=replay_prepare_workers,
                    total_iterations=1,
                    additional_online_dirs=online[1:],
                    optimizer_updates=definition["optimizer_updates"],
                    validate_initial=False,
                )
                checkpoint = fit / "iteration-0001"
                evaluation = fit / "validation/iteration-0001"
                files = [
                    checkpoint / "complete.json",
                    checkpoint / "state.pt",
                    evaluation / "rollout-summary.json",
                ]
                return dict(
                    artifacts={str(p): sha256(p) for p in files},
                    training=result,
                    metrics=_metrics(evaluation, scenes),
                    checkpoint=str(checkpoint),
                )

            fitted = journal.run(f"{index}/fit_validate", update)
            parent = Path(fitted["checkpoint"])
            decision = selection_decision(
                reference, best_metrics, fitted["metrics"], drop_successes=recipe["drop_successes"]
            )
            repeated = None
            if decision == "repeat":

                def repeat():
                    target = directory / "repeat"
                    run_parallel(
                        config_path,
                        target,
                        root=root,
                        audit_path=audit_path,
                        checkpoint=parent,
                        phase="monitor",
                        bootstrap_dir=bootstrap,
                        gpus=gpus,
                        worker_python=worker_python,
                        cpu_threads=cpu_threads,
                        baselines=False,
                    )
                    return dict(
                        artifacts=_cohort_artifacts(target), metrics=_metrics(target, scenes)
                    )

                repeated = journal.run(f"{index}/repeat", repeat)
                decision = selection_decision(
                    reference,
                    best_metrics,
                    fitted["metrics"],
                    drop_successes=recipe["drop_successes"],
                    repeat=repeated["metrics"],
                )
            if decision == "best":
                best, best_metrics = parent, fitted["metrics"]
            _link(output_dir, "best", best)
            _link(output_dir, "latest", parent)
            results.append(
                dict(
                    cycle=index,
                    checkpoint=str(parent),
                    metrics=fitted["metrics"],
                    repeat_metrics=repeated["metrics"] if repeated else None,
                    decision=decision,
                    training=fitted["training"],
                )
            )
            report = dict(
                initial_metrics=reference,
                cycles=results,
                best=str(best),
                latest=str(parent),
                status="stopped_regression" if decision == "stop" else "running",
            )
            atomic_json(output_dir / "report.json", report)
            if decision == "stop":
                return report
        report["status"] = "complete"
        atomic_json(output_dir / "report.json", report)
        return report
