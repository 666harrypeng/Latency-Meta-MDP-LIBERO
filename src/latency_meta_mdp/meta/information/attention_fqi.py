"""Fixed-target FQI on explicit decision contexts and a reproducible global batch."""

import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from latency_meta_mdp.meta.information.config import validate_binding
from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.fqi_batch import batch_candidates, select_microbatch
from latency_meta_mdp.meta.information.fqi_inputs import FQIInputs


def bellman_targets(rewards, terminated, next_values, next_legal):
    if torch.any(~terminated & ~next_legal.any(-1)):
        raise ValueError("nonterminal next state has no legal action")
    bootstrap = next_values.float().masked_fill(~next_legal, -torch.inf).max(-1).values
    return rewards.float() + torch.where(terminated, 0.0, bootstrap)


def regression_terms(
    values, actions, targets, legal, *, huber_beta=1.0, conservative_temperature=1.0
):
    if not np.isfinite(huber_beta) or huber_beta <= 0:
        raise ValueError("positive finite Huber beta required")
    if not np.isfinite(conservative_temperature) or conservative_temperature <= 0:
        raise ValueError("positive finite conservative temperature required")
    if not torch.all(legal.gather(1, actions[:, None])):
        raise ValueError("replay action is not legal")
    q = values.float()
    taken = q.gather(1, actions[:, None]).squeeze(1)
    td = F.smooth_l1_loss(taken, targets.float(), reduction="none", beta=huber_beta)
    conservative = (
        conservative_temperature
        * torch.logsumexp((q / conservative_temperature).masked_fill(~legal, -torch.inf), dim=-1)
        - taken
    )
    return td, conservative


def regression_loss(values, actions, targets, legal, conservative_weight, **scales):
    td, conservative = regression_terms(values, actions, targets, legal, **scales)
    return td + conservative_weight * conservative


def fit_attention_iteration(
    model,
    replay,
    *,
    optimizer,
    optimizer_updates,
    global_batch,
    microbatch,
    sampler,
    conservative_weight=0.01,
    huber_beta=1.0,
    conservative_temperature=1.0,
    grad_clip=0.5,
    progress=None,
    bf16=False,
    compile_model=False,
    rank=0,
    world_size=1,
    checkpoint=None,
):
    if type(optimizer_updates) is not int or optimizer_updates < 1:
        raise ValueError("positive optimizer update budget required")
    batch_candidates(microbatch, global_batch, world_size)
    device = next(model.parameters()).device
    if compile_model and device.type != "cuda":
        raise ValueError("compiled attention FQI requires CUDA")
    target = copy.deepcopy(model).eval().requires_grad_(False)
    # Sampling precedes fitting. Every rank sees the identical global stream.
    checkpoint_path = Path(checkpoint["path"]) if checkpoint else None
    losses, gradients = [], []
    # Additive sufficient statistics are small, globally reduced, and resumable.
    diagnostic_keys = [f"{g}/{s}" for g in ("startup", "old", "latest") for s in ("G", "O", "P")]
    statistics = torch.zeros((9, 8), device=device, dtype=torch.float64)
    diagnostic_updates = 0
    next_update = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_binding(checkpoint["binding"], saved["binding"])
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        target.load_state_dict(saved["target"])
        sampler.load_state_dict(saved["sampler"])
        global_plan = saved["global_plan"].numpy()
        next_update = saved["next_update"]
        losses, gradients = saved["losses"], saved["gradients"]
        if "diagnostic_statistics" in saved:
            statistics.copy_(saved["diagnostic_statistics"])
            diagnostic_updates = saved["diagnostic_updates"]
    else:
        global_plan = sampler.sample(global_batch * optimizer_updates).reshape(
            optimizer_updates, global_batch
        )
    requested_microbatch = microbatch
    local_plan = global_plan[:, rank::world_size]
    microbatch, batch_probe = select_microbatch(
        requested_microbatch,
        model=model,
        replay=replay,
        global_batch=global_batch,
        world_size=world_size,
        indices=local_plan[0],
        bf16=bf16,
    )
    execution = dict(
        global_batch=global_batch,
        microbatch=microbatch,
        accumulation=global_batch // (world_size * microbatch),
        compile_model=compile_model,
    )
    # Keep original modules for state/optimizer ownership. Compilation changes
    # execution only; public input checks remain outside captured tensor graphs.
    forward, target_forward = model, target
    if compile_model:
        # Preserve BF16 rounding boundaries while fusing intermediate operators.
        options = {"triton.cudagraphs": True, "emulate_precision_casts": True}
        forward = torch.compile(model, options=options, dynamic=False)
        target_forward = torch.compile(target, options=options, dynamic=False)
    before = model_hash(target)
    unique = np.unique(local_plan)
    targets = {}
    started = perf_counter()
    last_progress = started
    if progress:
        progress(
            dict(
                phase="target_progress",
                completed_decisions=0,
                unique_decisions=len(unique),
                seconds=0.0,
                **execution,
            )
        )
    config = model.config["memory"]
    with torch.no_grad(), FQIInputs(replay, config, microbatch=microbatch, device=device) as inputs:
        for ids, rows, batch in inputs.batches(unique, following=True):
            if compile_model:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                # Avoid compiling a one-off final short batch.
                values = (target_forward if len(ids) == microbatch else target)(batch)
            y = bellman_targets(
                torch.tensor(
                    [
                        r["training_reward"]
                        if "reward" in model.config["training"]
                        else r["task_reward"]
                        for r in rows
                    ],
                    device=device,
                ),
                torch.tensor([r["terminated"] for r in rows], dtype=torch.bool, device=device),
                values,
                torch.as_tensor(np.stack([r["next_legal"] for r in rows]), device=device),
            )
            targets.update(zip(map(int, ids), y.cpu().tolist(), strict=True))
            now = perf_counter()
            if progress and now - last_progress >= 10:
                progress(
                    dict(
                        phase="target_progress",
                        completed_decisions=len(targets),
                        unique_decisions=len(unique),
                        seconds=now - started,
                    )
                )
                last_progress = now
    target_seconds = perf_counter() - started
    if progress:
        progress(
            dict(
                phase="targets",
                completed_decisions=len(targets),
                unique_decisions=len(unique),
                seconds=target_seconds,
            )
        )
    fit_started = perf_counter()
    model.train()
    with FQIInputs(replay, config, microbatch=microbatch, device=device) as inputs:
        batches = iter(inputs.batches(local_plan[next_update:].ravel(), following=False))
        for update in range(next_update, optimizer_updates):
            if compile_model:
                torch.compiler.cudagraph_mark_step_begin()
            ids = local_plan[update]
            update_started = perf_counter()
            input_seconds = 0.0
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            update_statistics = torch.zeros_like(statistics)
            for start in range(0, len(ids), microbatch):
                input_started = perf_counter()
                batch_ids, rows, batch = next(batches)
                input_seconds += perf_counter() - input_started
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                    values = forward(batch)
                actions = torch.tensor([r["action"] for r in rows], device=device)
                y = torch.tensor([targets[int(i)] for i in batch_ids], device=device)
                legal = torch.as_tensor(np.stack([r["legal"] for r in rows]), device=device)
                td, conservative = regression_terms(
                    values,
                    actions,
                    y,
                    legal,
                    huber_beta=huber_beta,
                    conservative_temperature=conservative_temperature,
                )
                loss = (td + conservative_weight * conservative).sum() / global_batch
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite attention FQI loss")
                loss.backward()
                total += loss.detach()
                with torch.no_grad():
                    taken = values.float().gather(1, actions[:, None]).squeeze(1)
                    ceiling = 10 * model.config["training"].get("reward", {}).get(
                        "delivery_reward", 0.1
                    )
                    measurements = torch.stack(
                        (
                            torch.ones_like(td),
                            td,
                            conservative,
                            taken - y,
                            taken,
                            y,
                            legal.sum(-1),
                            (taken > ceiling).float(),
                        ),
                        dim=1,
                    ).double()
                    episodes = np.searchsorted(replay.ends, batch_ids, side="right")
                    keys = [
                        diagnostic_keys.index(f"{replay.groups[int(e)]}/{r['context'].state.stage}")
                        for e, r in zip(episodes, rows, strict=True)
                    ]
                    update_statistics.index_add_(0, torch.tensor(keys, device=device), measurements)
            if world_size > 1:
                parameters = list(model.parameters())
                gradients_flat = torch.cat(
                    [
                        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                        for p in parameters
                    ]
                )
                used = torch.tensor(
                    [p.grad is not None for p in parameters], device=device, dtype=torch.int32
                )
                torch.distributed.all_reduce(gradients_flat)
                torch.distributed.all_reduce(used, op=torch.distributed.ReduceOp.MAX)
                offset = 0
                for p, active in zip(parameters, used.cpu().tolist(), strict=True):
                    p.grad = (
                        gradients_flat[offset : offset + p.numel()].view_as(p) if active else None
                    )
                    offset += p.numel()
                torch.distributed.all_reduce(total)
                torch.distributed.all_reduce(update_statistics)
            statistics += update_statistics
            diagnostic_updates += 1
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(total.cpu()))
            gradients.append(float(norm))
            if checkpoint and rank == 0 and (update + 1) % checkpoint["every_updates"] == 0:
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(
                    dict(
                        binding=checkpoint["binding"],
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        target=target.state_dict(),
                        sampler=sampler.state_dict(),
                        global_plan=torch.from_numpy(global_plan),
                        next_update=update + 1,
                        losses=losses,
                        gradients=gradients,
                        diagnostic_statistics=statistics.cpu(),
                        diagnostic_updates=diagnostic_updates,
                    ),
                    temporary,
                )
                temporary.replace(checkpoint_path)
            if progress:
                progress(
                    dict(
                        phase="fit",
                        optimizer_update=update + 1,
                        optimizer_updates=optimizer_updates,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        **execution,
                        input_seconds=input_seconds,
                        update_seconds=perf_counter() - update_started,
                        updates_per_second=(update + 1 - next_update)
                        / (perf_counter() - fit_started),
                        loss=losses[-1],
                        grad_norm=gradients[-1],
                        td_loss=float(update_statistics[:, 1].sum() / global_batch),
                        conservative_loss=float(update_statistics[:, 2].sum() / global_batch),
                        weighted_conservative_loss=float(
                            conservative_weight * update_statistics[:, 2].sum() / global_batch
                        ),
                    )
                )
    after = model_hash(target)
    diagnostics = {}
    metric_names = (
        "td_loss",
        "conservative_loss",
        "signed_td_error",
        "taken_q_mean",
        "target_mean",
        "legal_actions_mean",
        "q_above_return_ceiling_fraction",
    )
    for key, values in zip(diagnostic_keys, statistics.cpu().tolist(), strict=True):
        if values[0]:
            diagnostics[key] = dict(
                count=int(values[0]),
                **dict(zip(metric_names, [v / values[0] for v in values[1:]], strict=True)),
            )
    sums = statistics.sum(0).cpu().tolist()
    return dict(
        target_hash_before=before,
        target_hash_after=after,
        optimizer_updates=optimizer_updates,
        resumed_updates=next_update,
        performed_updates=optimizer_updates - next_update,
        compile_model=compile_model,
        microbatch=microbatch,
        requested_microbatch=requested_microbatch,
        batch_probe=batch_probe,
        transitions=len(replay),
        transition_visits=optimizer_updates * global_batch,
        unique_decisions=len(np.unique(global_plan)),
        sampling=sampler.describe(global_plan),
        target_decisions=len(unique),
        target_seconds=target_seconds,
        regression_seconds=perf_counter() - fit_started,
        target_min=min(targets.values()),
        target_max=max(targets.values()),
        initial_loss=losses[0],
        final_loss=losses[-1],
        loss=float(np.mean(losses)),
        grad_norm_max=max(gradients),
        grad_clip_fraction=float(np.mean(np.asarray(gradients) > grad_clip)),
        diagnostics=diagnostics,
        diagnostic_updates=diagnostic_updates,
        td_loss=sums[1] / sums[0] if sums[0] else None,
        conservative_loss=sums[2] / sums[0] if sums[0] else None,
        weighted_conservative_loss=conservative_weight * sums[2] / sums[0] if sums[0] else None,
    )
