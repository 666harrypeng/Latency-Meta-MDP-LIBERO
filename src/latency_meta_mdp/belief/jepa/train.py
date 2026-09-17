"""Train a configured Direct predictor; immutable milestone and rolling resume checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from latency_meta_mdp.belief.jepa.data import (
    BalancedQuerySampler,
    collate_direct_samples,
)
from latency_meta_mdp.belief.jepa.job import atomic_json
from latency_meta_mdp.belief.jepa.model import (
    DirectJepaPredictor,
    save_direct_prediction_weights,
)
from latency_meta_mdp.belief.jepa.optimization import (
    train_direct_epoch,
)
from latency_meta_mdp.io.artifacts import sha256_file as digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run(args)


def run(args):
    root = Path.cwd()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("Commit the local implementation before formal training")
    from latency_meta_mdp.belief.jepa.job import (
        load_direct_data,
        load_direct_job,
        reconcile_epoch_journal,
    )

    job = load_direct_job(args.config, project_root=root)
    config_path, cfg = job.training_config, job.training
    preflight = args.preflight
    evidence = json.loads(preflight.read_text())
    if (
        evidence["status"] != "preflight_passed"
        or evidence["microbatch_size"] != job.microbatch_size
        or evidence["level"] != job.level
        or evidence.get("task_id") != job.task_id
        or evidence["training_config"] != asdict(cfg)
        or evidence["normalization_sha256"] != digest(job.normalization)
    ):
        raise ValueError("Preflight does not match this Direct job")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY=UNSET")
    if args.resume:
        manifest = json.loads((args.output_dir / "run.json").read_text())
        if (args.output_dir / "completion.json").exists():
            raise ValueError("Completed run cannot resume")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    norm_path = job.normalization
    print(f"Preparing {job.label} train records with metadata/size validation", flush=True)
    backbone_config, normalization, corpus, dataset = load_direct_data(job, split="train")
    identity = dict(
        training_config=asdict(cfg),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=corpus.source_manifest_sha256,
        cache_sha256=corpus.cache_manifest_sha256,
        split_sha256=corpus.split_manifest_sha256,
        normalization_sha256=digest(norm_path),
        launcher_sha256=digest(Path(__file__)),
        config_sha256=digest(config_path),
        job_sha256=digest(args.config),
        preflight_sha256=digest(preflight),
    )
    if args.resume and manifest["identity"] != identity:
        raise ValueError("Resume identity differs from committed code/data/config/launcher")
    for key in ("source", "cache", "split"):
        if identity[key + "_sha256"] != evidence[key + "_manifest_sha256"]:
            raise ValueError("Preflight and training input provenance differ")
    device = torch.device(job.device)
    model = DirectJepaPredictor(
        backbone_config=backbone_config, proprio_normalization=normalization, project_root=root
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay_start,
    )
    sampler = BalancedQuerySampler(dataset, sample_count=cfg.examples_per_epoch, seed=cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=job.microbatch_size,
        sampler=sampler,
        num_workers=job.num_workers,
        persistent_workers=job.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
        collate_fn=collate_direct_samples,
    )
    completed = 0
    if args.resume:
        latest = args.output_dir / "latest.pt"
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume checkpoint identity mismatch")
        completed = checkpoint["completed_epochs"]
        if not 0 < completed <= cfg.max_epochs:
            raise ValueError("Invalid checkpoint epoch")
        if checkpoint["optimizer_steps"] != completed * cfg.optimizer_steps_per_epoch:
            raise ValueError("Invalid checkpoint step count")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        reconcile_epoch_journal(
            args.output_dir / "epochs.jsonl",
            completed_epochs=completed,
            last_report=checkpoint["last_epoch_report"],
        )
        run_id = manifest["wandb_id"]
        del checkpoint
    else:
        run_id = uuid.uuid4().hex
        manifest = dict(
            format_id="direct_query_training_v1",
            identity=identity,
            initialization="scratch",
            level=job.level,
            task_id=job.task_id,
            action_contract_id=backbone_config.action_contract.contract_id,
            seed=cfg.seed,
            microbatch_size=job.microbatch_size,
            accumulation_steps=cfg.global_batch_size // job.microbatch_size,
            parameter_count=model.parameter_count,
            train_episodes=len(corpus.records),
            train_masters=len({r.logical_master_task_index for r in corpus.records}),
            real_pair_count=len(dataset),
            real_pairs_by_query=dataset.horizon_counts,
            source_start_tick=10,
            target_absorbing=False,
            sampling="balanced_q_uniform_valid_pairs_with_replacement",
            milestones=[25, 50, 75],
            expected_optimizer_steps=cfg.max_epochs * cfg.optimizer_steps_per_epoch,
            expected_examples=cfg.max_epochs * cfg.examples_per_epoch,
            expected_examples_per_query=cfg.max_epochs * cfg.examples_per_epoch // 20,
            wandb_id=run_id,
            validation_loaded=False,
            intended_device=torch.cuda.get_device_name(device),
        )
        atomic_json(args.output_dir / "run.json", manifest)
        (args.output_dir / "proprio_normalization.json").write_bytes(norm_path.read_bytes())
        (args.output_dir / "launcher.py").write_bytes(Path(__file__).read_bytes())
    run = wandb.init(
        project="latency-meta-mdp-action-conditioned-jepa",
        id=run_id,
        resume="must" if args.resume else "never",
        name=f"rtc-forecast-{job.label.lower()}-direct-q20-s{cfg.seed}-v1",
        group="rtc-forecast-direct-query-v1",
        config=manifest,
        dir=str(args.output_dir),
    )
    started = time.perf_counter()
    try:
        for epoch in range(completed, cfg.max_epochs):
            sampler.set_epoch(epoch)
            progress = epoch / max(1, cfg.max_epochs - 1)
            decay = (
                cfg.weight_decay_start
                + (cfg.weight_decay_final - cfg.weight_decay_start)
                * (1 - math.cos(math.pi * progress))
                / 2
            )
            for group in optimizer.param_groups:
                group["weight_decay"] = decay

            def observe(row):
                step = epoch * cfg.optimizer_steps_per_epoch + row["optimizer_step_in_epoch"]
                if step % 5 == 0 or row["optimizer_step_in_epoch"] == 1:
                    values = dict(
                        epoch=epoch + 1,
                        optimizer_step=step,
                        examples_seen=step * cfg.global_batch_size,
                        **row,
                    )
                    print(json.dumps(values), flush=True)
                    run.log(values, step=step)

            result = train_direct_epoch(
                model, loader, optimizer, config=cfg, device=device, on_update=observe
            )
            if result["seen_by_query"] != [0] + [cfg.examples_per_epoch // 20] * 20:
                raise ValueError("Epoch exposure is not exactly query balanced")
            if len(result["updates"]) != cfg.optimizer_steps_per_epoch:
                raise ValueError("Epoch optimizer step count mismatch")
            completed = epoch + 1
            if completed in (25, 50, 75):
                milestone = args.output_dir / "checkpoints" / f"epoch-{completed:03d}"
                milestone.mkdir(parents=True, exist_ok=True)
                model_path = milestone / "model.safetensors"
                # A crash after the milestone write but before latest.pt may leave this
                # file. Never overwrite it or treat it as a fresh completed epoch.
                if model_path.exists():
                    from safetensors.torch import load_file

                    old = load_file(str(model_path))
                    if set(old) != set(model.state_dict()) or any(
                        not torch.equal(old[k], v.detach().cpu())
                        for k, v in model.state_dict().items()
                    ):
                        raise ValueError("Existing milestone differs from resumed training")
                else:
                    save_direct_prediction_weights(model, model_path)
                atomic_json(
                    milestone / "manifest.json",
                    dict(
                        identity=identity,
                        epoch=completed,
                        optimizer_steps=completed * cfg.optimizer_steps_per_epoch,
                        examples_seen=completed * cfg.examples_per_epoch,
                        model_sha256=digest(model_path),
                    ),
                )
            epoch_report = dict(
                completed_epochs=completed,
                optimizer_steps=completed * cfg.optimizer_steps_per_epoch,
                examples_seen=completed * cfg.examples_per_epoch,
                result=result,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            )
            checkpoint = dict(
                identity=identity,
                completed_epochs=completed,
                optimizer_steps=completed * cfg.optimizer_steps_per_epoch,
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device),
                last_epoch_report=epoch_report,
            )
            temp = args.output_dir / "latest.pt.tmp"
            with temp.open("wb") as f:
                torch.save(checkpoint, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, args.output_dir / "latest.pt")
            del checkpoint
            atomic_json(args.output_dir / "progress.json", epoch_report)
            with (args.output_dir / "epochs.jsonl").open("a") as f:
                f.write(json.dumps(epoch_report) + "\n")
        atomic_json(
            args.output_dir / "completion.json",
            dict(
                status="training_complete_not_admitted",
                completed_epochs=completed,
                optimizer_steps=completed * cfg.optimizer_steps_per_epoch,
                examples_seen=completed * cfg.examples_per_epoch,
                wall_seconds=time.perf_counter() - started,
                identity=identity,
                formal_validation_pending=True,
            ),
        )
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
