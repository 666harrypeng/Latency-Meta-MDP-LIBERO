"""Attention FQI entrypoint over explicit source receipts and existing recovery machinery."""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from latency_meta_mdp.meta.information.config import (
    identity_hash,
    load_config,
    sha256,
    validate_binding,
)
from latency_meta_mdp.meta.information.deployment import deployment_binding, load_scheduler
from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.training import _save, train_iterations


def warm_start_budget(config, source_config, checkpoint, directory):
    """Convert success-only Q units without changing the representation or old run."""
    from latency_meta_mdp.meta.information.budget_reward import validate_reward_spec
    from latency_meta_mdp.meta.information.config import validate_binding

    if config["schema_version"] == 3:
        raise ValueError("optional prediction requires scratch initialization; no Q rescaling")
    for key in set(config) | set(source_config):
        if key not in ("training", "output_dir"):
            validate_binding(config[key], source_config[key])
    if (
        source_config["schema_version"] != 2
        or source_config["training"]["budget_lambda"] != 0
        or "reward" in source_config["training"]
    ):
        raise ValueError("budget warm-start requires a success-only attention checkpoint")
    validate_reward_spec(config["training"]["reward"])
    checkpoint, directory = Path(checkpoint).resolve(), Path(directory)
    scale = config["training"]["reward"]["delivery_reward"] / 0.1
    binding = dict(
        deployment_binding(config),
        normalization="fixed-physical-scales-and-learned-LayerNorm",
        warm_start=dict(
            parent_receipt_sha256=sha256(checkpoint / "complete.json"),
            source_config_hash=identity_hash(source_config),
            q_scale=scale,
            optimizer="reset",
            sampler="reset",
        ),
    )
    target = directory / "iteration-0000"
    if target.exists():
        receipt = json.loads((target / "complete.json").read_text())
        validate_binding(binding, receipt["binding"])
        load_scheduler(target, deployment_binding(config))
        return target
    model = load_scheduler(checkpoint, deployment_binding(source_config)).model
    model.config = copy.deepcopy(config)
    with torch.no_grad():
        for head in model.heads:
            head.weight.mul_(scale)
            head.bias.mul_(scale)
    directory.mkdir(parents=True, exist_ok=True)
    _save(directory, model, binding=binding, iteration=0, metrics=None)
    return target


def initialize_attention(config, directory, *, root, seed):
    """Publish random new-architecture weights; no old GRU parameters or updates."""
    from latency_meta_mdp.meta.information.model import build_model

    directory = Path(directory)
    checkpoint = directory / "iteration-0000"
    binding = dict(
        deployment_binding(config),
        seed=seed,
        normalization="fixed-physical-scales-and-learned-LayerNorm",
    )
    if checkpoint.exists():
        if config["schema_version"] == 3:
            previous = json.loads((checkpoint / "complete.json").read_text())["binding"]
            validate_binding(seed, previous["seed"])
            _scratch_parent_identity(config, checkpoint, seed=seed)
        load_scheduler(checkpoint, deployment_binding(config))
        return checkpoint
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = build_model(config, root=root)
    if config["schema_version"] == 3:
        binding["initialization"] = _scratch_identity(config, model, seed=seed)
    _save(directory, model, binding=binding, iteration=0, metrics=None)
    return checkpoint


def _scratch_identity(config, model, *, seed):
    return dict(
        kind="scratch",
        seed=seed,
        initial_model_hash=model_hash(model),
        experiment_id=config["experiment_id"],
    )


def _scratch_parent_identity(config, checkpoint, *, seed):
    """Only same-experiment continuation may supply weights to protocol 3."""
    receipt = json.loads((Path(checkpoint) / "complete.json").read_text())
    binding = receipt["binding"]
    for key, value in deployment_binding(config).items():
        validate_binding(value, binding.get(key))
    initial = binding.get("initialization", {})
    if (
        set(initial) != {"kind", "seed", "initial_model_hash", "experiment_id"}
        or initial["kind"] != "scratch"
        or initial["seed"] != seed
        or initial["experiment_id"] != config["experiment_id"]
        or not isinstance(initial["initial_model_hash"], str)
        or len(initial["initial_model_hash"]) != 64
    ):
        raise ValueError("same-experiment scratch provenance required")
    return initial


def run_attention_training(
    config_path,
    sources,
    output_dir,
    *,
    root,
    checkpoint=None,
    seed=0,
    total_iterations,
    optimizer_updates,
    global_batch,
    training_gpus=None,
    device="cpu",
    microbatch=2,
    bf16=False,
    compile_model=False,
    checkpoint_every_updates=256,
    cpu_threads=4,
    replay_cache_dir=None,
    replay_prepare_workers=2,
    validation_bootstrap=None,
    validation_gpus=None,
    validation_workers_per_gpu=1,
    validation_cpu_threads=None,
    audit_path=None,
    worker_python=None,
    validate_initial=True,
    stop_after_iterations=None,
    validation_every=1,
    episode_cache_size=128,
    replay_cache_max_bytes=None,
    compact_replay=False,
    sampling=None,
):
    from latency_meta_mdp.meta.information.attention_replay import prepare_attention_replay
    from latency_meta_mdp.meta.information.distributed_fqi import fit_attention_distributed
    from latency_meta_mdp.meta.information.model import build_model
    from latency_meta_mdp.meta.information.monitor import IterationValidator
    from latency_meta_mdp.meta.information.preflight import (
        verify_assets,
        verify_runtime_dependencies,
    )
    from latency_meta_mdp.meta.information.validation import atomic_json
    from latency_meta_mdp.meta.online_tracking import OnlineFitTracker

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if config["schema_version"] not in (2, 3):
        raise ValueError("attention entrypoint requires schema 2 or 3")
    initialization = None
    if config["schema_version"] == 3 and checkpoint is not None:
        initialization = _scratch_parent_identity(config, checkpoint, seed=seed)
    devices = ["cpu"]
    if training_gpus is not None:
        physical = training_gpus.split(",")
        if any(not d.isdigit() for d in physical) or len(set(physical)) != len(physical):
            raise ValueError("distinct physical learner GPUs required")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = training_gpus
        devices = [f"cuda:{i}" for i in range(len(physical))]
    elif device != "cpu":
        raise ValueError("use train-gpus for GPU fitting, including single GPU")
    from latency_meta_mdp.meta.information.fqi_batch import batch_candidates

    if type(optimizer_updates) is not int or optimizer_updates < 1:
        raise ValueError("optimizer_updates must be a positive integer")
    batch_candidates(microbatch, global_batch, len(devices))
    verify_runtime_dependencies(root, config)
    verify_assets(root, config["assets"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    preparation_started = time.monotonic()

    def preparation_progress(event):
        event = dict(event, elapsed_seconds=time.monotonic() - preparation_started)
        atomic_json(output_dir / "preparation-progress.json", event)
        print(json.dumps(event), flush=True)

    replay, conversion = prepare_attention_replay(
        config,
        sources,
        root=root,
        cache_dir=replay_cache_dir or output_dir / "replay-cache",
        workers=replay_prepare_workers,
        episode_cache_size=episode_cache_size,
        cache_max_bytes=replay_cache_max_bytes,
        compact_replay=compact_replay,
        progress=preparation_progress,
    )
    atomic_json(output_dir / "replay-conversion.json", conversion)
    state = None
    if checkpoint is None:
        model = build_model(config, root=root)
        parent = None
        if config["schema_version"] == 3:
            initialization = _scratch_identity(config, model, seed=seed)
    else:
        checkpoint = Path(checkpoint).resolve()
        model = load_scheduler(checkpoint, deployment_binding(config)).model.requires_grad_(True)
        parent = dict(path=str(checkpoint), receipt_sha256=sha256(checkpoint / "complete.json"))
        saved = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        if "optimizer" in saved:
            receipt = json.loads((checkpoint / "complete.json").read_text())
            state = {k: saved[k] for k in ("optimizer", "sampler")}
            state["ancestor_optimizer_updates"] = (receipt.get("metrics") or {}).get(
                "ancestor_optimizer_updates", 0
            )
    from latency_meta_mdp.meta.information.runtime_identity import source_identity

    source = source_identity(root)
    atomic_json(output_dir / "runtime-source.json", source)
    binding = dict(
        deployment_binding(config),
        seed=seed,
        parent=parent,
        trainer_code_hash=source["code_hash"],
        replay_conversion_hash=identity_hash(conversion),
        normalization="fixed-physical-scales-and-learned-LayerNorm",
        offline_training_episodes=sum(g == "startup" for g in replay.groups),
        online_training_episodes=sum(g != "startup" for g in replay.groups),
        sampling=sampling
        if sampling is not None
        else config["training"].get(
            "sampling", "startup=.5; old=.25; latest=.25; uniform decisions within source group"
        ),
    )
    if sampling is not None and state is not None:
        binding["sampling_transition"] = dict(
            previous=state["sampler"].get("sampling"),
            current=sampling,
            boundary="new_fit_directory",
            optimizer="preserved",
            rng="preserved",
        )
    if initialization is not None:
        binding["initialization"] = initialization
    validator = None
    if "reward" in config["training"]:
        binding["reward"] = config["training"]["reward"]
    if validation_bootstrap is not None or (
        config["schema_version"] == 3 and audit_path is not None
    ):
        validator = IterationValidator(
            config_path,
            output_dir / "validation",
            root=root,
            audit_path=audit_path,
            bootstrap_dir=validation_bootstrap,
            model=model,
            worker_python=worker_python,
            gpus=validation_gpus,
            workers_per_gpu=validation_workers_per_gpu,
            cpu_threads=cpu_threads if validation_cpu_threads is None else validation_cpu_threads,
            prune=False,
        )
        binding["validation_protocol"] = validator.protocol
    tracker = OnlineFitTracker(output_dir, binding=binding, episodes=optimizer_updates, epochs=1)
    if validator:
        validator.progress = tracker
    options = dict(
        devices=devices,
        microbatch=microbatch,
        cpu_threads=cpu_threads,
        bf16=bf16,
        compile_model=compile_model,
        global_batch=global_batch,
        optimizer_updates=optimizer_updates,
        grad_clip=config["training"]["grad_clip"],
    )
    for key in ("huber_beta", "conservative_temperature"):
        if key in config["training"]:
            options[key] = config["training"][key]
    with (output_dir / "execution-history.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                dict(
                    host=os.uname().nodename, physical_gpus=training_gpus, source=source, **options
                )
            )
            + "\n"
        )
    success = False
    try:
        result = train_iterations(
            model,
            replay,
            output_dir,
            binding=binding,
            total_iterations=total_iterations,
            epochs=1,
            learning_rate=config["training"]["learning_rate"],
            conservative_weight=config["training"]["conservative_weight"],
            validate_checkpoint=validator,
            progress=tracker,
            fitter=fit_attention_distributed,
            attention_options=options,
            checkpoint_every_updates=checkpoint_every_updates,
            initial_learning_state=state,
            validate_initial=validate_initial,
            stop_after_iterations=stop_after_iterations,
            validation_every=validation_every,
            sampling=sampling,
        )
        success = True
        return result
    finally:
        tracker.close(success=success)
