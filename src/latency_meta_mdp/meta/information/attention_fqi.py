"""Fixed-target FQI on explicit decision contexts and a reproducible global batch."""

import copy
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.memory import collate_contexts
from latency_meta_mdp.meta.information.memory_schema import validate_batch


def bellman_targets(rewards, terminated, next_values, next_legal):
    if torch.any(~terminated & ~next_legal.any(-1)):
        raise ValueError("nonterminal next state has no legal action")
    bootstrap = next_values.float().masked_fill(~next_legal, -torch.inf).max(-1).values
    return rewards.float() + torch.where(terminated, 0.0, bootstrap)


def regression_loss(values, actions, targets, legal, conservative_weight):
    if not torch.all(legal.gather(1, actions[:, None])):
        raise ValueError("replay action is not legal")
    q = values.float()
    taken = q.gather(1, actions[:, None]).squeeze(1)
    return F.smooth_l1_loss(taken, targets.float(), reduction="none") + conservative_weight * (
        torch.logsumexp(q.masked_fill(~legal, -torch.inf), dim=-1) - taken
    )


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
    grad_clip=0.5,
    progress=None,
    bf16=False,
    rank=0,
    world_size=1,
):
    if type(optimizer_updates) is not int or optimizer_updates < 1:
        raise ValueError("positive optimizer update budget required")
    validate_batch(
        global_batch=global_batch,
        world_size=world_size,
        microbatch=microbatch,
        accumulation=global_batch // (world_size * microbatch),
    )
    device = next(model.parameters()).device
    target = copy.deepcopy(model).eval().requires_grad_(False)
    before = model_hash(target)
    # Sampling precedes fitting. Every rank sees the identical global stream.
    global_plan = sampler.sample(global_batch * optimizer_updates).reshape(
        optimizer_updates, global_batch
    )
    local_plan = global_plan[:, rank::world_size]
    unique = np.unique(local_plan)
    targets = {}
    started = perf_counter()
    config = model.config["memory"]
    with torch.no_grad():
        for start in range(0, len(unique), microbatch):
            ids = unique[start : start + microbatch]
            rows = [replay[int(i)] for i in ids]
            contexts = [
                r["next_context"] if r["next_context"] is not None else r["context"] for r in rows
            ]
            batch = collate_contexts(contexts, device=device, config=config)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                values = target(batch)
            y = bellman_targets(
                torch.tensor([r["task_reward"] for r in rows], device=device),
                torch.tensor([r["terminated"] for r in rows], dtype=torch.bool, device=device),
                values,
                torch.as_tensor(np.stack([r["next_legal"] for r in rows]), device=device),
            )
            targets.update(zip(map(int, ids), y.cpu().tolist(), strict=True))
    target_seconds = perf_counter() - started
    if progress:
        progress(dict(phase="targets", unique_decisions=len(unique), seconds=target_seconds))
    losses, gradients = [], []
    fit_started = perf_counter()
    model.train()
    for update, ids in enumerate(local_plan):
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for start in range(0, len(ids), microbatch):
            batch_ids = ids[start : start + microbatch]
            rows = [replay[int(i)] for i in batch_ids]
            batch = collate_contexts([r["context"] for r in rows], device=device, config=config)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                values = model(batch)
            loss = (
                regression_loss(
                    values,
                    torch.tensor([r["action"] for r in rows], device=device),
                    torch.tensor([targets[int(i)] for i in batch_ids], device=device),
                    torch.as_tensor(np.stack([r["legal"] for r in rows]), device=device),
                    conservative_weight,
                ).sum()
                / global_batch
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite attention FQI loss")
            loss.backward()
            total += loss.detach()
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
                p.grad = gradients_flat[offset : offset + p.numel()].view_as(p) if active else None
                offset += p.numel()
            torch.distributed.all_reduce(total)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(total.cpu()))
        gradients.append(float(norm))
        if progress:
            progress(
                dict(
                    phase="fit",
                    optimizer_update=update + 1,
                    loss=losses[-1],
                    grad_norm=gradients[-1],
                )
            )
    after = model_hash(target)
    return dict(
        target_hash_before=before,
        target_hash_after=after,
        optimizer_updates=optimizer_updates,
        transitions=len(replay),
        transition_visits=optimizer_updates * global_batch,
        unique_decisions=len(np.unique(global_plan)),
        target_decisions=len(unique),
        target_seconds=target_seconds,
        regression_seconds=perf_counter() - fit_started,
        target_min=min(targets.values()),
        target_max=max(targets.values()),
        initial_loss=losses[0],
        final_loss=losses[-1],
        loss=float(np.mean(losses)),
        grad_norm_max=max(gradients),
    )
