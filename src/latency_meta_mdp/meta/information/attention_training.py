"""Attention FQI entrypoint over explicit source receipts and existing recovery machinery."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from latency_meta_mdp.meta.information.config import identity_hash, load_config, sha256
from latency_meta_mdp.meta.information.deployment import deployment_binding, load_scheduler
from latency_meta_mdp.meta.information.training import _save, train_iterations


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
        load_scheduler(checkpoint, deployment_binding(config))
        return checkpoint
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = build_model(config, root=root)
    _save(directory, model, binding=binding, iteration=0, metrics=None)
    return checkpoint


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
    cpu_threads=4,
    replay_cache_dir=None,
    replay_prepare_workers=2,
    validation_bootstrap=None,
    validation_gpus=None,
    audit_path=None,
    worker_python=None,
    validate_initial=True,
    stop_after_iterations=None,
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
    if config["schema_version"] != 2:
        raise ValueError("attention entrypoint requires schema 2")
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
    from latency_meta_mdp.meta.information.memory_schema import validate_batch

    if type(optimizer_updates) is not int or optimizer_updates < 1:
        raise ValueError("optimizer_updates must be a positive integer")
    if type(global_batch) is not int or type(microbatch) is not int or microbatch < 1:
        raise ValueError("global_batch and microbatch must be positive integers")
    validate_batch(
        global_batch=global_batch,
        world_size=len(devices),
        microbatch=microbatch,
        accumulation=global_batch // (len(devices) * microbatch),
    )
    verify_runtime_dependencies(root, config)
    verify_assets(root, config["assets"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    replay, conversion = prepare_attention_replay(
        config,
        sources,
        root=root,
        cache_dir=replay_cache_dir or output_dir / "replay-cache",
        workers=replay_prepare_workers,
        progress=lambda event: print(json.dumps(event), flush=True),
    )
    atomic_json(output_dir / "replay-conversion.json", conversion)
    state = None
    if checkpoint is None:
        model = build_model(config, root=root)
        parent = None
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
    code = {
        str(p.relative_to(root)): sha256(p)
        for path in ("src/latency_meta_mdp", "scripts")
        for p in sorted((root / path).rglob("*.py"))
    }
    binding = dict(
        deployment_binding(config),
        seed=seed,
        parent=parent,
        trainer_code_hash=identity_hash(code),
        replay_conversion_hash=identity_hash(conversion),
        normalization="fixed-physical-scales-and-learned-LayerNorm",
        offline_training_episodes=sum(g == "startup" for g in replay.groups),
        online_training_episodes=sum(g != "startup" for g in replay.groups),
        sampling="startup=.5; old=.25; latest=.25; uniform decisions within source group",
    )
    validator = None
    if validation_bootstrap is not None:
        validator = IterationValidator(
            config_path,
            output_dir / "validation",
            root=root,
            audit_path=audit_path,
            bootstrap_dir=validation_bootstrap,
            model=model,
            worker_python=worker_python,
            gpus=validation_gpus,
            cpu_threads=cpu_threads,
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
        global_batch=global_batch,
        optimizer_updates=optimizer_updates,
        grad_clip=config["training"]["grad_clip"],
    )
    with (output_dir / "execution-history.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(host=os.uname().nodename, physical_gpus=training_gpus, **options))
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
            initial_learning_state=state,
            validate_initial=validate_initial,
            stop_after_iterations=stop_after_iterations,
        )
        success = True
        return result
    finally:
        tracker.close(success=success)
