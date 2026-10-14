"""Iteration-boundary FQI recovery, independent of robot collection services."""

import json
import random
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from latency_meta_mdp.meta.information.config import sha256, validate_binding
from latency_meta_mdp.meta.information.fqi import fit_iteration, model_hash


def _save(path, model, *, binding, iteration, metrics, learning_state=None):
    path = Path(path)
    target = path / f"iteration-{iteration:04d}"
    if target.exists():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=path))
    numpy_rng = np.random.get_state()
    state = dict(
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all()
        if next(model.parameters()).device.type == "cuda"
        else [],
        python_rng=random.getstate(),
        numpy_rng=(numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
    )
    if hasattr(model, "config"):
        state["model_config"] = model.config
    if learning_state is not None:
        state.update(
            model_config=model.config,
            optimizer=learning_state["optimizer"].state_dict(),
            sampler=learning_state["sampler"].state_dict(),
        )
    torch.save(state, temporary / "state.pt")
    receipt = dict(
        binding=binding,
        iteration=iteration,
        metrics=metrics,
        model_hash=model_hash(model),
        files={"state.pt": sha256(temporary / "state.pt")},
        recovery=(
            "outer iteration boundary; persistent Adam and global decision sampler"
            if learning_state is not None
            else "outer iteration boundary; next target is current Q; Adam resets each iteration"
        ),
    )
    (temporary / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.rename(target)
    link = path / ".latest.tmp"
    link.unlink(missing_ok=True)
    link.symlink_to(target.name, target_is_directory=True)
    link.replace(path / "latest")
    return receipt


def _restore(path, model, *, binding, learning_state=None):
    entries = sorted(Path(path).glob("iteration-*"))
    if not entries:
        return None
    retention = Path(path) / "retention.json"
    pruned = json.loads(retention.read_text())["pruned"] if retention.exists() else {}
    for index, entry in enumerate(entries):
        receipt = json.loads((entry / "complete.json").read_text())
        validate_binding(binding, receipt["binding"])
        if receipt["iteration"] != index or entry.name != f"iteration-{index:04d}":
            raise ValueError("checkpoint iteration sequence differs")
        if not (entry / "state.pt").exists() and entry != entries[-1]:
            if pruned.get(entry.name) != sha256(entry / "complete.json"):
                raise ValueError("missing checkpoint without matching retention receipt")
        elif receipt["files"] != {"state.pt": sha256(entry / "state.pt")}:
            raise ValueError("checkpoint hash differs")
    state = torch.load(entries[-1] / "state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    if learning_state is not None:
        validate_binding(model.config, state["model_config"])
        learning_state["optimizer"].load_state_dict(state["optimizer"])
        learning_state["sampler"].load_state_dict(state["sampler"])
    if model_hash(model) != receipt["model_hash"]:
        raise ValueError("restored model hash differs")
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state((numpy_rng[0], np.array(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
    return receipt


def train_iterations(
    model,
    episodes,
    output_dir,
    *,
    binding,
    total_iterations,
    epochs,
    learning_rate,
    conservative_weight,
    stop_after_iterations=None,
    progress=None,
    validate_checkpoint=None,
    fitter=None,
    fitter_options=None,
    validate_initial=True,
    attention_options=None,
    initial_learning_state=None,
):
    """Resume only published complete iterations; restart an interrupted iteration."""
    if total_iterations < 1 or (stop_after_iterations is not None and stop_after_iterations < 1):
        raise ValueError("invalid fitted iteration budget")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = dict(
        total_iterations=total_iterations,
        epochs=epochs,
        learning_rate=learning_rate,
        conservative_weight=conservative_weight,
        gamma=1.0,
        budget_lambda=0.0,
        chunk_size=64,
        optimizer="Adam-reset-per-fitted-iteration",
        validate_initial=validate_initial,
    )
    learning_state = None
    if attention_options is not None:
        from latency_meta_mdp.meta.information.attention_replay import DecisionSampler

        learning_state = dict(
            optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
            sampler=DecisionSampler(episodes, seed=binding.get("seed", 0)),
        )
        if initial_learning_state is not None:
            learning_state["optimizer"].load_state_dict(initial_learning_state["optimizer"])
            learning_state["sampler"].load_state_dict(initial_learning_state["sampler"])
        recipe.pop("epochs")
        recipe.pop("chunk_size")
        recipe.update(
            optimizer="Adam-persistent",
            batch_unit="decision-context",
            attention_fqi={
                k: v
                for k, v in attention_options.items()
                if k not in ("devices", "microbatch", "cpu_threads")
            },
        )
        binding = {
            **binding,
            "architecture": {k: model.config[k] for k in ("model", "memory", "actions")},
        }
    if fitter is not None:
        recipe["parallel_fqi"] = {
            key: value
            for key, value in (fitter_options or {}).items()
            if key not in {"devices", "cache_bytes", "cpu_threads"}
        }
    binding = {**binding, "recipe": recipe}
    receipt = _restore(output_dir, model, binding=binding, learning_state=learning_state)
    if receipt is None:
        receipt = _save(
            output_dir,
            model,
            binding=binding,
            iteration=0,
            metrics=None,
            learning_state=learning_state,
        )

    def validate(receipt):
        if validate_checkpoint is None:
            return
        checkpoint = output_dir / f"iteration-{receipt['iteration']:04d}"
        # Evaluation cannot alter the learner's random stream or training mode.
        numpy_rng, python_rng = np.random.get_state(), random.getstate()
        training = model.training
        devices = (
            [next(model.parameters()).device.index or 0]
            if next(model.parameters()).device.type == "cuda"
            else []
        )
        try:
            with torch.random.fork_rng(devices=devices):
                validate_checkpoint(checkpoint, receipt)
        finally:
            np.random.set_state(numpy_rng)
            random.setstate(python_rng)
            model.train(training)

    if validate_initial or receipt["iteration"] > 0:
        validate(receipt)
    first = receipt["iteration"]
    last = total_iterations
    if stop_after_iterations is not None:
        last = min(last, first + stop_after_iterations)
    for iteration in range(first + 1, last + 1):
        started = perf_counter()

        def notify(value):
            if progress is not None:
                progress(dict(iteration=iteration, **value))

        if learning_state is None:
            metrics = (fitter or fit_iteration)(
                model,
                episodes,
                mode=binding["mode"],
                epochs=epochs,
                learning_rate=learning_rate,
                conservative_weight=conservative_weight,
                progress=notify,
                sources=getattr(episodes, "sources", None),
                **(fitter_options or {}),
            )
        else:
            from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration

            metrics = (fitter or fit_attention_iteration)(
                model,
                episodes,
                conservative_weight=conservative_weight,
                progress=notify,
                **learning_state,
                **attention_options,
            )
            previous_updates = (receipt.get("metrics") or {}).get("ancestor_optimizer_updates", 0)
            if not previous_updates and initial_learning_state is not None:
                previous_updates = initial_learning_state.get("ancestor_optimizer_updates", 0)
            metrics["ancestor_optimizer_updates"] = previous_updates + metrics["optimizer_updates"]
        if metrics["target_hash_before"] != metrics["target_hash_after"]:
            raise ValueError("frozen target changed during regression")
        metrics["seconds"] = perf_counter() - started
        metrics["transition_visits_per_second"] = (
            metrics.get("transition_visits", (1 + epochs) * metrics["transitions"])
            / metrics["seconds"]
        )
        if getattr(episodes, "cache", None) is not None:
            metrics["replay_cache"] = episodes.cache.metrics()
        if next(model.parameters()).device.type == "cuda":
            metrics["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt = _save(
            output_dir,
            model,
            binding=binding,
            iteration=iteration,
            metrics=metrics,
            learning_state=learning_state,
        )
        status = dict(
            completed_iterations=iteration,
            total_iterations=total_iterations,
            metrics=metrics,
            checkpoint=f"iteration-{iteration:04d}",
        )
        temporary = output_dir / "progress.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output_dir / "progress.json")
        notify(dict(phase="iteration_complete", **status))
        validate(receipt)
    return dict(completed_iterations=receipt["iteration"], metrics=receipt["metrics"])


class VerifiedReplay:
    """Hash-admitted episode stream with optional bounded host caching."""

    def __init__(self, root, manifest, *, progress=None, cache=None):
        from latency_meta_mdp.io.paths import repository_root
        from latency_meta_mdp.meta.information.cohort import verify_complete
        from latency_meta_mdp.runtime.evaluate_conveyor import conveyor_delay_sampler

        self.cache = cache
        root = Path(root)
        self.entries, self.receipts = [], {}
        cohort = manifest["binding"]
        if cohort["scope"] not in ("starter_training", "online_training"):
            raise ValueError("only training cohorts are admitted")
        identities = [e["episode_id"] for e in manifest["episodes"]]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate training episode identity")
        for definition in manifest["episodes"]:
            if definition["split"] != "train":
                raise ValueError("non-training split in replay")
            matches = []
            for attempt in sorted((root / "episodes" / definition["episode_id"]).glob("attempt-*")):
                if verify_complete(attempt, binding=dict(cohort=cohort, episode=definition)):
                    matches.append(attempt)
            if len(matches) != 1:
                raise ValueError("exactly one complete attempt required per training episode")
            attempt = matches[0]
            summary = json.loads((attempt / "summary.json").read_text())
            parcels = summary["parcels"]
            if (
                not summary["terminated"]
                or summary["truncated"]
                or (
                    summary["starvation_ticks"]
                    and cohort.get("actions", {}).get("starvation_mode") != "allow_hold"
                )
                or summary["scope"] != cohort["scope"]
                or parcels["spawned"] != 10
                or parcels["active"]
                or sum(parcels[k] for k in ("successes", "misses", "timeouts")) != 10
                or not np.isclose(summary["task_reward"], parcels["successes"] / 10)
            ):
                raise ValueError("invalid training episode outcome")
            inner_path = attempt / "episode/complete.json"
            inner = json.loads(inner_path.read_text())
            actual = inner["binding"]
            _, delay = conveyor_delay_sampler(
                repository_root(),
                regime=definition["regime"],
                scene_seed=definition["scene"],
                policy_seed=definition["policy_seed"],
            )
            expected = dict(
                assets=cohort["assets"]["identity"],
                mode=cohort["mode"],
                scene=definition["scene"],
                scope=cohort["scope"],
                code_hash=cohort["code_hash"],
                scheduler_seed=definition["scheduler_seed"],
                behavior=definition["behavior"],
                policy_seed=definition["policy_seed"],
                delay=delay,
            )
            if cohort["scope"] == "online_training":
                if cohort["phase"] != "online" or cohort["schema_version"] not in (1, 2):
                    raise ValueError("invalid online training cohort")
                if definition["behavior"] == "fqi":
                    from latency_meta_mdp.meta.information.cycle import policy_binding

                    expected["learned_policy"] = policy_binding(cohort, definition)
            if cohort.get("schema_version", 1) == 2:
                expected.update(actions=cohort["actions"], pattern=definition.get("pattern"))
            validate_binding(expected, actual)
            if actual["delay"]["regime"] != definition["regime"] or not inner["terminated"]:
                raise ValueError("training delay/terminal binding differs")
            self.entries.append((attempt / "episode", actual, sha256(inner_path)))
            self.receipts[str(attempt.relative_to(root))] = sha256(attempt / "complete.json")
            if progress is not None:
                progress(dict(phase="admission", episodes=len(self.entries)))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        from latency_meta_mdp.meta.information.dataset import load_episode

        path, binding, receipt_hash = self.entries[index]
        if sha256(path / "complete.json") != receipt_hash:
            raise ValueError("replay receipt changed after admission")
        if self.cache is not None:
            return self.cache.load(path, binding=binding)
        return load_episode(path, binding=binding)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def run_training(
    config_path,
    cohort_dir,
    output_dir,
    *,
    root,
    device="cuda:0",
    seed=0,
    stop_after_iterations=None,
):
    import importlib.metadata
    import os

    from latency_meta_mdp.meta.information.config import identity_hash, load_config
    from latency_meta_mdp.meta.information.model import StageQ
    from latency_meta_mdp.meta.information.preflight import (
        verify_assets,
        verify_runtime_dependencies,
    )

    root, cohort_dir, output_dir = Path(root), Path(cohort_dir), Path(output_dir)
    config = load_config(config_path)
    manifest = json.loads((cohort_dir / "cohort.json").read_text())
    validate_binding(identity_hash(config), manifest["binding"]["config_hash"])
    validate_binding(config["assets"], manifest["binding"]["assets"])
    if manifest["binding"]["mode"] != "bound" or manifest["binding"]["schema_version"] != 1:
        raise ValueError("first FQI training requires bound schema1")
    from latency_meta_mdp.meta.information.cohort import BEHAVIORS

    expected = {
        (s, r, b)
        for s in config["splits"]["train"][:16]
        for r in ("nominal", "family")
        for b in BEHAVIORS
    }
    actual = {(e["scene"], e["regime"], e["behavior"]) for e in manifest["episodes"]}
    if actual != expected or len(manifest["episodes"]) != 128:
        raise ValueError("complete paired starter128 required")
    if torch.device(device).type == "cuda":
        from latency_meta_mdp.meta.information.resources import gpu_processes

        active, _ = gpu_processes()
        if active:
            raise RuntimeError("GPU compute process already active")
    verify_runtime_dependencies(root, config)
    verify_assets(root, config["assets"])

    def progress(value):
        print(json.dumps(value), flush=True)

    from latency_meta_mdp.meta.information.replay_cache import EpisodeCache

    replay = VerifiedReplay(cohort_dir, manifest, progress=progress, cache=EpisodeCache())
    code = {
        str(p.relative_to(root)): sha256(p)
        for directory in ("src/latency_meta_mdp/meta/information",)
        for p in sorted((root / directory).glob("*.py"))
    }
    code["scripts/run_conveyor_meta.py"] = sha256(root / "scripts/run_conveyor_meta.py")
    binding = dict(
        mode=config["mode"],
        algorithm="fqi",
        seed=seed,
        cohort_sha256=sha256(cohort_dir / "cohort.json"),
        replay_receipts_hash=identity_hash(replay.receipts),
        assets=config["assets"]["identity"],
        trainer_code_hash=identity_hash(code),
        schema_version=1,
        normalization="fixed-physical-scales-and-learned-LayerNorm",
        device=device,
        versions={k: importlib.metadata.version(k) for k in ("torch", "numpy")},
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = StageQ().to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training = config["training"]
    return train_iterations(
        model,
        replay,
        output_dir,
        binding=binding,
        total_iterations=training["fitted_iterations"],
        epochs=training["epochs_per_iteration"],
        learning_rate=training["learning_rate"],
        conservative_weight=training["conservative_weight"],
        stop_after_iterations=stop_after_iterations,
        progress=progress,
    )
