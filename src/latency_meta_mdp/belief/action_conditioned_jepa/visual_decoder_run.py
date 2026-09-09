"""Replicated-DDP training of only the RGB decoder, with exact epoch/batch resume."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder import (
    DualViewVisualDecoder,
    VisualDecoderConfig,
    visual_reconstruction_loss,
)
from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder_data import (
    VisualDecoderDataset,
    verify_visual_decoder_data,
)
from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder_evaluation import (
    decode,
    evaluate_reconstruction,
    save_comparison,
)


def _write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_decoder_training_checkpoint(model, optimizer, output, progress):
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    temp = output / "model.safetensors.tmp"
    save_file(tensors, str(temp))
    temp.replace(output / "model.safetensors")
    state = {
        "progress": progress,
        "optimizer": optimizer.state_dict(),
        "model_sha256": sha256_file(output / "model.safetensors"),
    }
    temp = output / "trainer.pt.tmp"
    torch.save(state, temp)
    temp.replace(output / "trainer.pt")
    _write(output / "checkpoint.json", {**progress, "model_sha256": state["model_sha256"]})


def train_visual_decoder(
    *,
    project_root: Path,
    data_manifest: Path,
    output: Path,
    epochs: int = 40,
    batch_per_device: int = 64,
    workers: int = 2,
    seed: int = 27,
    limit_steps: int | None = None,
    resume: bool = False,
    wandb_enabled: bool = False,
):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (
        min(epochs, batch_per_device) <= 0
        or workers < 0
        or (limit_steps is not None and limit_steps <= 0)
    ):
        raise ValueError("invalid decoder training budget")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = output.resolve()
    data_manifest = data_manifest.resolve()
    model_config = VisualDecoderConfig()
    config = {
        "model": dataclasses.asdict(model_config),
        "epochs": epochs,
        "batch_per_device": batch_per_device,
        "world_size": world,
        "global_batch": world * batch_per_device,
        "seed": seed,
        "data_manifest_sha256": sha256_file(data_manifest),
        "learning_rate": 3e-4,
        "minimum_learning_rate": 3e-5,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "edge_weight": 0.1,
        "trainable": "visual_decoder_only",
    }
    if rank == 0:
        if resume:
            previous = json.loads((output / "run.json").read_text())
            assert all(previous[k] == v for k, v in config.items()), "resume configuration mismatch"
        else:
            output.mkdir(parents=True, exist_ok=False)
            _write(
                output / "run.json",
                {
                    **config,
                    "source_code_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                },
            )
        _write(output / "status.json", {"phase": "verifying_data"})
        verify_visual_decoder_data(data_manifest, project_root=project_root)
    if world > 1:
        dist.barrier()
    train = VisualDecoderDataset(data_manifest, project_root=project_root, partition="fit")
    heldout = VisualDecoderDataset(data_manifest, project_root=project_root, partition="holdout")
    sampler = torch.utils.data.DistributedSampler(
        train, num_replicas=world, rank=rank, seed=seed, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_per_device,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
    )
    model = DualViewVisualDecoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    epoch_start = batch_start = steps = examples = 0
    if resume:
        state = torch.load(output / "trainer.pt", map_location=device, weights_only=True)
        assert sha256_file(output / "model.safetensors") == state["model_sha256"]
        model.load_state_dict(
            load_file(str(output / "model.safetensors"), device=str(device)), strict=True
        )
        optimizer.load_state_dict(state["optimizer"])
        progress = state["progress"]
        epoch_start, batch_start = progress["next_epoch"], progress["next_batch"]
        steps, examples = progress["optimizer_steps"], progress["examples_seen"]
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    tracker = None
    if rank == 0 and wandb_enabled:
        import wandb

        tracker = wandb.init(
            project="meta-mdp-visual-decoder",
            name=output.name,
            config=config,
            id=output.name,
            resume="allow",
        )
    start = time.monotonic()
    total_steps = epochs * len(loader)
    initial = model.input_projection.weight.detach().clone()
    try:
        for epoch in range(epoch_start, epochs):
            wrapped.train()
            sampler.set_epoch(epoch)
            running = torch.zeros(4, device=device, dtype=torch.float64)
            for batch_index, (z, rgb) in enumerate(loader):
                if epoch == epoch_start and batch_index < batch_start:
                    continue
                if steps < len(loader):
                    lr = 3e-4 * (steps + 1) / len(loader)
                else:
                    phase = (steps - len(loader)) / max(1, total_steps - len(loader))
                    lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * phase))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                z = z.to(device, non_blocking=True)
                target = rgb.to(device, non_blocking=True).float() / 255
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = wrapped(z)
                    terms = visual_reconstruction_loss(predicted, target)
                if not torch.isfinite(terms["total"]):
                    raise FloatingPointError("nonfinite decoder training loss")
                terms["total"].backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                n = z.shape[0]
                running += torch.stack(
                    [
                        terms["total"].detach() * n,
                        terms["pixel"].detach() * n,
                        terms["edge"].detach() * n,
                        torch.tensor(n, device=device),
                    ]
                )
                steps += 1
                examples += n * world
                finished_epoch = batch_index + 1 == len(loader)
                cursor = {
                    "next_epoch": epoch + int(finished_epoch),
                    "next_batch": 0 if finished_epoch else batch_index + 1,
                    "optimizer_steps": steps,
                    "examples_seen": examples,
                }
                if rank == 0 and (steps % 50 == 0 or steps == 1):
                    log = {
                        "phase": "training",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "steps": steps,
                        "total_steps": total_steps,
                        "loss": float(terms["total"].detach()),
                        "pixel": float(terms["pixel"].detach()),
                        "edge": float(terms["edge"].detach()),
                        "gradient_norm": float(grad),
                        "lr": lr,
                        "examples_seen": examples,
                        "elapsed_seconds": time.monotonic() - start,
                    }
                    _write(output / "status.json", log)
                    print(json.dumps(log), flush=True)
                    if tracker is not None:
                        tracker.log(log, step=steps)
                if limit_steps is not None and steps >= limit_steps:
                    if rank == 0:
                        save_decoder_training_checkpoint(model, optimizer, output, cursor)
                        _write(
                            output / "status.json",
                            {
                                "phase": "smoke_complete",
                                **cursor,
                                "parameters_updated": not torch.equal(
                                    initial, model.input_projection.weight
                                ),
                                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                            },
                        )
                    if world > 1:
                        dist.barrier()
                    return
            if world > 1:
                dist.all_reduce(running)
            if rank == 0:
                save_decoder_training_checkpoint(model, optimizer, output, cursor)
                record = {
                    "epoch": epoch + 1,
                    **cursor,
                    "mean_loss": float(running[0] / running[3]),
                    "mean_pixel": float(running[1] / running[3]),
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                if epoch + 1 in {1, 5, 10, 20, epochs}:
                    model.eval()
                    record["heldout"] = evaluate_reconstruction(
                        model, heldout, batch_size=batch_per_device, workers=workers
                    )
                    z, rgb = heldout[len(heldout) // 2]
                    save_comparison(
                        rgb.float() / 255,
                        decode(model, z[None])[0],
                        output / f"epoch-{epoch + 1:03d}-preview.png",
                        caption="GT main | decoded main | GT wrist | decoded wrist; heldout master",
                    )
                if epoch + 1 in {10, 20, epochs}:
                    checkpoint = output / f"epoch-{epoch + 1:03d}"
                    checkpoint.mkdir(exist_ok=False)
                    save_file(
                        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                        str(checkpoint / "model.safetensors"),
                    )
                    _write(checkpoint / "run.json", config)
                    _write(
                        checkpoint / "checkpoint.json",
                        {**cursor, "model_sha256": sha256_file(checkpoint / "model.safetensors")},
                    )
                with (output / "epochs.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
                if tracker is not None:
                    tracker.log(record, step=steps)
            if world > 1:
                dist.barrier()
        if rank == 0:
            _write(
                output / "status.json",
                {
                    "phase": "complete",
                    **cursor,
                    "epochs": epochs,
                    "elapsed_seconds": time.monotonic() - start,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                },
            )
    finally:
        if tracker is not None:
            tracker.finish()
        if world > 1:
            dist.destroy_process_group()
