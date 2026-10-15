"""IQL initialization and a bounded, current-batch-only PPO campaign."""

import argparse
import copy
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import yaml

from latency_meta_mdp.meta.information.actor_batches import (
    offline_weights,
    on_policy_auxiliary,
    score_models,
)
from latency_meta_mdp.meta.information.actor_critic import initialize_from_encoder
from latency_meta_mdp.meta.information.actor_learning import (
    balanced_probabilities,
    fit,
    initial_state,
    load_state,
    restore_models,
    save_state,
)
from latency_meta_mdp.meta.information.attention_replay import (
    DecisionReplay,
    prepare_attention_replay,
)
from latency_meta_mdp.meta.information.config import (
    identity_hash,
    load_config,
    sha256,
    validate_binding,
)
from latency_meta_mdp.meta.information.deployment import deployment_binding, load_scheduler
from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.monitor import IterationValidator
from latency_meta_mdp.meta.information.parallel import run_parallel
from latency_meta_mdp.meta.information.training import _save
from latency_meta_mdp.meta.information.validation import atomic_json


class EpisodeSubset:
    def __init__(self, episodes, indices):
        self.episodes, self.indices = episodes, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.episodes[self.indices[index]]


def generation(config, name, count, trial):
    if count % 3 or not 0 < count // 3 <= len(config["splits"]["train"]):
        raise ValueError("episode count must cover triplets of distinct training scenes")
    seeds = config["splits"]["train"]
    start = (trial * (count // 3)) % len(seeds)
    scenes = [seeds[(start + i) % len(seeds)] for i in range(count // 3)]
    return dict(
        name=name,
        scenes=scenes,
        behaviors=["actor"] * len(scenes),
        regimes=["zero", "nominal", "family"],
        trial=trial,
        epsilon=0.0,
    )


def reset_batch(state, binding):
    """Carry online Adam forward, but never carry old batch sampling rights or counters."""
    state = copy.deepcopy(state)
    state.update(phase="ppo", step=0, epoch=0, binding=binding, actor_stopped=False)
    state.pop("fit_binding", None)
    if set(state["optimizers"]) != {"actor", "v"}:
        state["optimizers"] = {}
    return state


def tree_bytes(path):
    return sum(
        p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink()
    )


def capacity(path, *, maximum, reserve, required=0):
    used, free = tree_bytes(path), shutil.disk_usage(path).free
    if used + required > maximum or free - required < reserve:
        raise RuntimeError(
            f"PPO storage admission failed: used={used}, free={free}, required={required}"
        )
    return dict(used_bytes=used, free_bytes=free, maximum_bytes=maximum, reserve_bytes=reserve)


def prune_batch(directory, *, successor_checkpoint, receipts):
    """Keep lightweight audit and validation; only delete this campaign's committed raw batch."""
    directory = Path(directory).resolve()
    successor_checkpoint = Path(successor_checkpoint).resolve()
    load_state(successor_checkpoint)
    if not directory.name == "collection" or not (directory.parent / "complete.json").is_file():
        raise ValueError("only completed PPO cycle collection can be retired")
    receipt = dict(
        collection=str(directory),
        successor=str(successor_checkpoint),
        successor_sha256=sha256(successor_checkpoint / "complete.json"),
        receipts=receipts,
    )
    atomic_json(directory.parent / "retired-batch.json", receipt)
    # Canonical episode links point inside workers. Delete only these owned payload subtrees.
    for path in (directory / "episodes", directory / "workers"):
        if path.is_symlink():
            raise ValueError("unexpected collection root symlink")
        if path.exists():
            shutil.rmtree(path)


class Tracker:
    def __init__(self, directory, recipe, enabled):
        self.run = None
        if enabled:
            import wandb
            from latency_meta_mdp.meta.tracking_transport import start_run

            path = directory / "wandb-run.json"
            saved = json.loads(path.read_text()) if path.exists() else dict(id=uuid.uuid4().hex[:8])
            self.run = start_run(
                directory,
                id=saved["id"],
                resume="allow",
                name=directory.name,
                project=os.environ.get("WANDB_PROJECT", "latency-meta-mdp-robosuite-delta-ppo"),
                entity=os.environ.get("WANDB_ENTITY"),
                job_type="iql-to-ppo",
                tags=["conveyor", "attention", "iql", "ppo", "delta"],
                config=recipe,
                dir=str(directory),
                settings=wandb.Settings(disable_git=True, save_code=False),
            )
            atomic_json(path, dict(id=saved["id"], url=self.run.url))
            print("WANDB_RUN " + str(self.run.url), flush=True)

    def event(self, phase, values):
        print("PPO_CAMPAIGN " + json.dumps(dict(values, phase=phase)), flush=True)
        if self.run:
            self.run.log(
                {f"{phase}/{k}": v for k, v in values.items() if isinstance(v, (int, float, bool))}
            )


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root, self.output = Path(args.root).resolve(), Path(args.output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = load_config(args.config)
        self.recipe = yaml.safe_load(Path(args.recipe).read_text())
        if self.recipe["algorithm"] != "iql_then_ppo" or self.recipe["reward"] != dict(
            gamma=1.0, budget_lambda=0.0, gae_lambda=1.0
        ):
            raise ValueError("only approved success-only finite-horizon IQL/PPO supported")
        physical = args.gpus.split(",")
        if (
            not physical
            or len(set(physical)) != len(physical)
            or any(not x.isdigit() for x in physical)
        ):
            raise ValueError("unique runtime GPU indices required")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        self.devices = [f"cuda:{i}" for i in range(len(physical))]
        self.binding = dict(
            config=identity_hash(self.config),
            recipe=identity_hash(self.recipe),
            sources=sha256(args.sources),
            audit=sha256(args.scene_audit),
            code=identity_hash(
                {
                    str(p.relative_to(self.root)): sha256(p)
                    for folder in ("src/latency_meta_mdp", "scripts")
                    for p in sorted((self.root / folder).rglob("*.py"))
                }
            ),
        )
        path = self.output / "binding.json"
        if path.exists():
            validate_binding(self.binding, json.loads(path.read_text()))
        else:
            atomic_json(path, self.binding)
        self.tracker = Tracker(self.output, self.recipe, not args.no_wandb)
        self.validator = IterationValidator(
            args.config,
            self.output / "validation",
            root=self.root,
            audit_path=args.scene_audit,
            bootstrap_dir=args.bootstrap_dir,
            worker_python=sys.executable,
            gpus=args.gpus,
            cpu_threads=args.cpu_threads,
            prune=False,
            progress=lambda e: self.tracker.event(
                "validation", dict(iteration=e["iteration"], **e.get("metrics", {}))
            ),
        )

    def options(self, phase, **overrides):
        recipe = self.recipe[phase]
        keys = {
            "updates",
            "epochs",
            "global_batch",
            "learning_rate",
            "actor_lr",
            "tau",
            "ema",
            "checkpoint_every",
            "clip",
            "target_kl",
            "entropy_coefficient",
            "grad_clip",
        }
        return dict(
            {k: v for k, v in recipe.items() if k in keys},
            microbatch=self.args.microbatch,
            cpu_threads=self.args.cpu_threads,
            seed=self.recipe["seed"],
            **overrides,
        )

    def train(self, phase, state, replay, directory, options, **kwargs):
        directory = Path(directory)
        if (directory / "latest").exists():
            saved = load_state(directory / "latest")
            validate_binding(saved.get("binding", {}), state.get("binding", {}))
            state = saved
        return fit(
            phase,
            state,
            replay,
            directory,
            options=options,
            devices=self.devices,
            progress=lambda e: self.tracker.event(phase, e),
            **kwargs,
        )

    def replay(self, sources, cache):
        # A shared node cache ceiling covers startup, calibration, pilot and online together.
        cache_root = Path(self.args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        elsewhere = tree_bytes(cache_root) - (tree_bytes(cache) if Path(cache).exists() else 0)
        allowance = self.args.cache_gib * 1024**3 - elsewhere
        if allowance <= 0:
            raise RuntimeError("global actor replay cache capacity exhausted")
        return prepare_attention_replay(
            self.config,
            sources,
            root=self.root,
            cache_dir=cache,
            workers=self.args.prepare_workers,
            episode_cache_size=self.args.episode_cache,
            cache_max_bytes=allowance,
            progress=lambda e: print("REPLAY " + json.dumps(e), flush=True),
        )

    def export(self, state, number):
        directory = self.output / "policies"
        directory.mkdir(exist_ok=True)
        path = directory / f"iteration-{number:04d}"
        model = restore_models(dict(state, models={"actor": state["models"]["actor"]}))["actor"]
        binding = dict(
            deployment_binding(self.config, algorithm="actor"),
            normalization="fixed-physical-scales-and-learned-LayerNorm",
            campaign=self.binding,
        )
        if not path.exists():
            _save(directory, model, binding=binding, iteration=number, metrics=None)
        scheduler = load_scheduler(path, deployment_binding(self.config, algorithm="actor"))
        validate_binding(model_hash(model), scheduler.version)
        return path

    def validate(self, checkpoint):
        return self.validator(checkpoint, json.loads((checkpoint / "complete.json").read_text()))

    def collect(self, checkpoint, directory, count, trial, name):
        capacity(
            self.output,
            maximum=self.args.max_run_gib * 1024**3,
            reserve=self.args.min_free_gib * 1024**3,
        )
        manifest_path = directory / "cohort.json"
        if not manifest_path.exists():
            directory.mkdir(parents=True, exist_ok=True)
        run_parallel(
            self.args.config,
            directory,
            root=self.root,
            audit_path=self.args.scene_audit,
            checkpoint=checkpoint,
            phase="online",
            bootstrap_dir=self.args.bootstrap_dir,
            gpus=self.args.gpus,
            worker_python=sys.executable,
            cpu_threads=self.args.cpu_threads,
            enabled=False,
            generation=generation(self.config, name, count, trial),
            on_episode=lambda _: capacity(
                self.output,
                maximum=self.args.max_run_gib * 1024**3,
                reserve=self.args.min_free_gib * 1024**3,
            ),
        )
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads((directory / "rollout-summary.json").read_text())
        if summary["completed"] != count or summary["expected"] != count:
            raise ValueError("incomplete actor collection batch")
        source = dict(path=str(directory), manifest_sha256=sha256(manifest_path), group="latest")
        summaries = {r["episode"]["episode_id"]: r["report"] for r in summary["episodes"]}
        return (
            source,
            manifest["episodes"],
            [summaries[e["episode_id"]] for e in manifest["episodes"]],
        )

    def score(self, state, replay, directory, kind, conversion):
        return score_models(
            state,
            replay,
            directory,
            devices=self.devices,
            microbatch=self.args.microbatch,
            kind=kind,
            binding=conversion,
            cpu_threads=self.args.cpu_threads,
        )

    def offline(self):
        directory = self.output / "offline"
        completed = directory / "complete.json"
        if completed.exists():
            return load_state(Path(json.loads(completed.read_text())["selected"]) / "latest")
        directory.mkdir(exist_ok=True)
        sources = json.loads(Path(self.args.sources).read_text())
        replay, conversion = self.replay(sources, Path(self.args.cache_dir) / "startup")
        metadata = [
            e
            for s in sources
            for e in json.loads((self.root / s["path"] / "cohort.json").read_text())["episodes"]
        ]
        if len(metadata) != 448:
            raise ValueError("approved startup requires exactly448 archived training episodes")
        probabilities = balanced_probabilities(replay.lengths.tolist(), metadata)
        norm = json.loads((self.root / self.config["assets"]["paths"]["normalization"]).read_text())
        state = initial_state(self.config, seed=self.recipe["seed"], normalization=norm)
        state["binding"] = dict(campaign=self.binding, conversion=identity_hash(conversion))
        state = self.train(
            "iql",
            state,
            replay,
            directory / "value",
            self.options("iql"),
            probabilities=probabilities,
        )
        scores = self.score(state, replay, directory / "teacher", "iql", conversion)
        auxiliary, diagnostics = offline_weights(
            scores,
            probabilities,
            temperature=self.recipe["actor"]["temperature"],
            maximum=self.recipe["actor"]["maximum_weight"],
        )
        atomic_json(directory / "weight-diagnostics.json", diagnostics)
        self.tracker.event("offline_weights", diagnostics)
        teacher = restore_models(state)["v"]
        actor = initialize_from_encoder(teacher)
        actor_state = dict(
            config=self.config,
            phase="new_actor",
            step=0,
            epoch=0,
            optimizers={},
            models={"actor": actor.state_dict()},
            binding=dict(
                campaign=self.binding, teacher=sha256(directory / "teacher/complete.json")
            ),
        )
        selected, best = None, -1.0
        for passes in self.recipe["actor"]["passes"]:
            frozen = directory / f"actor-pass-{passes}"
            options = self.options(
                "actor",
                updates=math.ceil(len(replay) * passes / self.recipe["actor"]["global_batch"]),
            )
            if (frozen / "latest").exists():
                actor_state = load_state(frozen / "latest")
            else:
                actor_state = self.train(
                    "actor",
                    actor_state,
                    replay,
                    directory / "actor",
                    options,
                    auxiliary=auxiliary,
                    probabilities=probabilities,
                )
            checkpoint = self.export(actor_state, passes)
            metrics = self.validate(checkpoint)
            frozen = directory / f"actor-pass-{passes}"
            if not (frozen / "latest").exists():
                save_state(frozen, actor_state)
            if metrics["selection_score"] > best:
                selected, best = frozen, metrics["selection_score"]
        if best <= 0:
            raise RuntimeError(
                "offline actor validation has no delayed success; inspect before PPO"
            )
        atomic_json(
            completed,
            dict(
                selected=str(selected), selection_score=best, conversion=identity_hash(conversion)
            ),
        )
        return load_state(selected / "latest")

    def calibrate(self, actor_state):
        directory = self.output / "calibration"
        directory.mkdir(exist_ok=True)
        if (directory / "complete.json").exists():
            return load_state(directory / "fit/latest")
        actor = restore_models(actor_state)["actor"]
        state = dict(
            config=self.config,
            phase="new_critic",
            step=0,
            epoch=0,
            optimizers={},
            models=dict(
                actor=actor.state_dict(), v=initialize_from_encoder(actor, value=True).state_dict()
            ),
        )
        checkpoint = self.export(state, 30)
        source, metadata, summaries = self.collect(
            checkpoint,
            directory / "collection",
            self.recipe["calibration"]["episodes"],
            0,
            "calibration",
        )
        replay, conversion = self.replay([source], Path(self.args.cache_dir) / "calibration")
        scores = self.score(state, replay, directory / "scores", "online", conversion)
        auxiliary = on_policy_auxiliary(
            replay, scores, metadata, summaries, actor_hash=model_hash(actor)
        )
        if not np.any(auxiliary["returns"] > 0):
            raise RuntimeError(
                "stochastic warm-start actor has no success; calibration cannot qualify"
            )
        scenes = sorted({e["scene"] for e in metadata})
        fit_scenes = set(scenes[: self.recipe["calibration"]["fit_scenes"]])
        episode_indices = [i for i, e in enumerate(metadata) if e["scene"] in fit_scenes]
        starts = np.r_[0, replay.ends[:-1]]
        fit_indices = np.concatenate(
            [np.arange(starts[i], replay.ends[i]) for i in episode_indices]
        )
        training = DecisionReplay(EpisodeSubset(replay.episodes, episode_indices))
        state["binding"] = dict(
            campaign=self.binding,
            conversion=identity_hash(conversion),
            actor_hash=model_hash(actor),
            fit_scenes=sorted(fit_scenes),
        )
        state = self.train(
            "critic",
            state,
            training,
            directory / "fit",
            self.options("calibration"),
            auxiliary={"returns": auxiliary["returns"][fit_indices]},
        )
        validate_binding(model_hash(actor), model_hash(restore_models(state)["actor"]))
        calibrated = self.score(
            state, replay, directory / "calibrated-scores", "online", conversion
        )
        heldout = np.ones(len(replay), bool)
        heldout[fit_indices] = False
        errors = {}
        for name, mask in [("fit", ~heldout), ("heldout", heldout)]:
            target, prediction = auxiliary["returns"][mask], calibrated["old_values"][mask]
            variance = float(target.var())
            errors[name] = dict(
                mse=float(np.mean((target - prediction) ** 2)),
                bias=float(np.mean(prediction - target)),
                explained_variance=1 - float((target - prediction).var()) / variance
                if variance > 0
                else None,
            )
        atomic_json(
            directory / "complete.json",
            dict(
                actor_hash=model_hash(actor),
                diagnostics=errors,
                fit_checkpoint=str(directory / "fit/latest"),
            ),
        )
        self.tracker.event("calibration_complete", {"episodes": len(metadata)})
        return state

    def online(self, initial, *, pilot):
        name = "pilot" if pilot else "online"
        budget = self.recipe["pilot"] if pilot else self.recipe["ppo"]
        base = self.output / name
        base.mkdir(exist_ok=True)
        state = initial
        for cycle in range(1, budget["cycles"] + 1):
            directory = base / f"cycle-{cycle:04d}"
            directory.mkdir(exist_ok=True)
            complete = directory / "complete.json"
            if complete.exists():
                state = load_state(directory / "fit/latest")
                self.retire_previous(base, directory, cycle)
                continue
            self.tracker.event(name, dict(cycle=cycle, stage="collect"))
            checkpoint = self.export(state, (1000 if pilot else 2000) + cycle - 1)
            source, metadata, summaries = self.collect(
                checkpoint,
                directory / "collection",
                budget["episodes"],
                cycle + (10000 if pilot else 0),
                f"{name}-{cycle}",
            )
            cache = Path(self.args.cache_dir) / name / f"cycle-{cycle:04d}"
            replay, conversion = self.replay([source], cache)
            scores = self.score(state, replay, directory / "scores", "online", conversion)
            actor_hash = model_hash(restore_models(state)["actor"])
            auxiliary = on_policy_auxiliary(
                replay, scores, metadata, summaries, actor_hash=actor_hash
            )
            batch_binding = dict(
                campaign=self.binding,
                policy=actor_hash,
                conversion=identity_hash(conversion),
                cycle=cycle,
                scope=name,
            )
            state = reset_batch(state, batch_binding)
            state = self.train(
                "ppo", state, replay, directory / "fit", self.options("ppo"), auxiliary=auxiliary
            )
            checkpoint = self.export(state, (1000 if pilot else 2000) + cycle)
            metrics = self.validate(checkpoint)
            storage = capacity(
                self.output,
                maximum=self.args.max_run_gib * 1024**3,
                reserve=self.args.min_free_gib * 1024**3,
            )
            record = dict(
                cycle=cycle,
                episodes=len(metadata),
                decisions=len(replay),
                metrics=metrics,
                storage=storage,
                raw_bytes=tree_bytes(directory / "collection"),
                cache_bytes=tree_bytes(cache),
                checkpoint=str(checkpoint),
                actor_before=actor_hash,
                batch=batch_binding,
            )
            atomic_json(complete, record)
            self.tracker.event(name, dict(cycle=cycle, decisions=len(replay), **metrics))
            self.retire_previous(base, directory, cycle)
        atomic_json(
            base / "complete.json", dict(cycles=budget["cycles"], episodes=budget["episodes"])
        )
        return state

    def retire_previous(self, base, directory, cycle):
        if cycle <= 1:
            return
        previous = base / f"cycle-{cycle - 1:04d}"
        # Retrying this action also completes deletion after an interrupted cleanup.
        prune_batch(
            previous / "collection",
            successor_checkpoint=directory / "fit/latest",
            receipts=json.loads((previous / "complete.json").read_text()),
        )
        old_cache = Path(self.args.cache_dir) / base.name / previous.name
        if old_cache.exists():
            shutil.rmtree(old_cache)

    def run(self):
        from latency_meta_mdp.meta.information.preflight import (
            verify_assets,
            verify_runtime_dependencies,
        )

        verify_runtime_dependencies(self.root, self.config)
        verify_assets(self.root, self.config["assets"])
        actor = self.offline()
        if self.args.stage == "offline":
            return
        calibrated = self.calibrate(actor)
        self.online(calibrated, pilot=True)
        if self.args.stage == "pilot":
            return
        # A real two-cycle pilot must qualify storage for96, including current+recovery batch.
        records = [
            json.loads(p.read_text()) for p in (self.output / "pilot").glob("cycle-*/complete.json")
        ]
        multiplier = self.recipe["ppo"]["episodes"] / self.recipe["pilot"]["episodes"]
        estimate = max(r["raw_bytes"] for r in records) * multiplier * 2 * 1.25
        cache_estimate = max(r["cache_bytes"] for r in records) * multiplier * 2 * 1.25
        capacity(
            self.args.cache_dir,
            maximum=self.args.cache_gib * 1024**3,
            reserve=self.args.min_free_gib * 1024**3,
            required=int(cache_estimate),
        )
        capacity(
            self.output,
            maximum=self.args.max_run_gib * 1024**3,
            reserve=self.args.min_free_gib * 1024**3,
            required=int(estimate),
        )
        self.tracker.event("formal_admitted", dict(estimated_two_batch_bytes=estimate))
        self.online(calibrated, pilot=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-audit", required=True)
    parser.add_argument("--bootstrap-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--episode-cache", type=int, default=512)
    parser.add_argument("--cache-gib", type=int, default=768)
    parser.add_argument("--max-run-gib", type=int, default=360)
    parser.add_argument("--min-free-gib", type=int, default=32)
    parser.add_argument("--stage", choices=["offline", "pilot", "formal"], default="pilot")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    from latency_meta_mdp.meta.information.resources import interrupt_on_termination, worker_lock

    with interrupt_on_termination(), worker_lock(Path(args.output_dir)):
        campaign = Campaign(args)
        try:
            campaign.run()
        finally:
            if campaign.tracker.run:
                campaign.tracker.run.finish()
