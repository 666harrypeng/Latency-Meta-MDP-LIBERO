"""Synchronous episode minibatch FQI; one CPU coordinator, disposable GPU ranks."""

import copy
import json
import os
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from latency_meta_mdp.meta.information.fqi import _mask, model_hash, replay_weights, targets
from latency_meta_mdp.meta.information.model import StageQ


def _target(model, episode, mode, device):
    if not episode or not episode[-1]["terminated"]:
        raise ValueError("complete terminal episodes required")
    masks = np.stack([_mask(t["packet"], mode) for t in episode])
    for i, row in enumerate(episode):
        if not masks[i, row["action"]] or (i < len(episode) - 1 and row["terminated"]):
            raise ValueError("illegal replay action or premature terminal")
    with torch.no_grad():
        q, _ = model.sequence([t["packet"] for t in episode])
        y = targets(
            torch.tensor([t["task_reward"] for t in episode], device=device),
            torch.tensor([t["cost"] for t in episode], device=device),
            torch.tensor([t["terminated"] for t in episode], device=device),
            torch.cat((q[1:], torch.zeros_like(q[:1]))),
            torch.as_tensor(np.concatenate((masks[1:], np.zeros_like(masks[:1]))), device=device),
        )
    if not torch.isfinite(y).all():
        raise ValueError("nonfinite fitted target")
    return y.cpu().tolist()


def _backward(model, episode, target, weight, mode, conservative_weight, chunk_size, device):
    hidden = None
    totals = np.zeros(3)
    q_min, q_max, above, count = float("inf"), -float("inf"), 0, 0
    for start in range(0, len(episode), chunk_size):
        rows = episode[start : start + chunk_size]
        q, hidden = model.block([t["packet"] for t in rows], hidden)
        actions = torch.tensor([t["action"] for t in rows], device=device)
        chosen = q.gather(1, actions[:, None]).squeeze(1)
        mask = torch.as_tensor(np.stack([_mask(t["packet"], mode) for t in rows]), device=device)
        huber = (
            F.smooth_l1_loss(chosen, target[start : start + len(rows)], reduction="sum") * weight
        )
        conservative = (
            (torch.logsumexp(q.masked_fill(~mask, -torch.inf), dim=1) - chosen).sum() * weight
            if conservative_weight
            else q.new_zeros(())
        )
        loss = huber + conservative_weight * conservative
        if not torch.isfinite(loss):
            raise ValueError("nonfinite fitted loss")
        loss.backward()
        totals += torch.stack((loss.detach(), huber.detach(), conservative.detach())).cpu().numpy()
        values = q[mask].detach()
        q_min, q_max = min(q_min, values.min().item()), max(q_max, values.max().item())
        above += int((values > 1).sum())
        count += values.numel()
        hidden = hidden.detach()
    return totals, (q_min, q_max, above, count)


def _gather(value, world):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def _worker(rank, devices, state, episodes, options, orders, directory):
    world = len(devices)
    device = torch.device(devices[rank])
    torch.set_num_threads(options["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=5),
        )
    try:
        _fit_worker(rank, world, device, state, episodes, options, orders, Path(directory))
    except BaseException:
        # Publish the cause before teardown can make peers fail in collectives.
        (Path(directory) / f"failure-{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        if world > 1:
            dist.destroy_process_group()


def _fit_worker(rank, world, device, state, episodes, options, orders, directory):
    if options.get("architecture") == "attention":
        return _attention_worker(rank, world, device, state, episodes, options, directory)
    # Replay wrappers are metadata-only at spawn. A rank retains decoded episodes
    # across target/regression passes; the total host allowance is divided by ranks.
    from latency_meta_mdp.meta.information.packed_replay import PackedReplay
    from latency_meta_mdp.meta.information.replay_cache import EpisodeCache

    cache = (
        episodes
        if isinstance(episodes, PackedReplay)
        else EpisodeCache(options["cache_bytes"] // world)
    )
    if hasattr(episodes, "offline"):
        episodes.offline.cache = episodes.online.cache = episodes.cache = cache
    elif hasattr(episodes, "cache"):
        episodes.cache = cache
    model = StageQ().to(device)
    model.load_state_dict(state)
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    before = model_hash(frozen)
    started = perf_counter()
    own = {}
    for index in range(rank, len(episodes), world):
        own[index] = _target(frozen, episodes[index], options["mode"], device)
        with (directory / f"target-{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(index=index)) + "\n")
    fixed = {k: v for shard in _gather(own, world) for k, v in shard.items()}
    if set(fixed) != set(range(len(episodes))):
        raise ValueError("distributed targets do not cover replay exactly")
    target_seconds = perf_counter() - started
    lengths = [len(fixed[i]) for i in range(len(episodes))]
    weights = replay_weights(lengths, options["sources"])
    log = directory / "progress.jsonl"

    def report(event):
        if rank == 0:
            with log.open("a") as stream:
                stream.write(json.dumps(event) + "\n")

    report(dict(phase="target", episode=len(episodes), episodes=len(episodes)))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    fit_started = perf_counter()
    losses, updates = 0.0, 0
    q_min, q_max, above, count = float("inf"), -float("inf"), 0, 0
    params = list(model.parameters())
    for epoch, order in enumerate(orders):
        groups = [
            order[i : i + options["global_batch"]]
            for i in range(0, len(order), options["global_batch"])
        ]
        for position, batch in enumerate(groups):
            optimizer.zero_grad(set_to_none=True)
            source_totals = np.zeros((2, 3))
            for index in batch[rank::world]:
                episode = episodes[index]
                if len(episode) != lengths[index]:
                    raise ValueError("replay changed during iteration")
                totals, stats = _backward(
                    model,
                    episode,
                    torch.tensor(fixed[index], device=device),
                    weights[index] / len(batch),
                    options["mode"],
                    options["conservative_weight"],
                    options["chunk_size"],
                    device,
                )
                group = (
                    0 if options["sources"] is None or options["sources"][index] == "offline" else 1
                )
                source_totals[group] += totals
                q_min, q_max = min(q_min, stats[0]), max(q_max, stats[1])
                above, count = above + stats[2], count + stats[3]
            # SUM, not mean: each episode already has its global minibatch weight.
            # Include absent heads as zero; every rank executes identical collectives.
            gradient = torch.cat(
                [(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params]
            )
            used = torch.tensor(
                [p.grad is not None for p in params], device=device, dtype=torch.int32
            )
            if world > 1:
                dist.all_reduce(gradient)
                dist.all_reduce(used, op=dist.ReduceOp.MAX)
            offset = 0
            for p, active in zip(params, used.cpu().tolist(), strict=True):
                p.grad = gradient[offset : offset + p.numel()].view_as(p) if active else None
                offset += p.numel()
            torch.nn.utils.clip_grad_norm_(params, 0.5, error_if_nonfinite=True)
            optimizer.step()
            totals = torch.tensor(source_totals, device=device)
            if world > 1:
                dist.all_reduce(totals)
            sources = totals.cpu().numpy()
            loss, huber, conservative = sources.sum(axis=0)
            losses += float(loss) * len(batch)
            updates += 1
            report(
                dict(
                    phase="fit",
                    epoch=epoch + 1,
                    passes_per_iteration=len(orders),
                    episode=position + 1,
                    episodes=len(groups),
                    source="mixed" if options["sources"] else "offline",
                    loss=float(loss),
                    huber_loss=float(huber),
                    conservative_loss=float(conservative),
                    source_losses={
                        s: dict(
                            zip(
                                ("combined_loss", "huber_loss", "conservative_loss"),
                                map(float, sources[i]),
                                strict=True,
                            )
                        )
                        for i, s in enumerate(("offline", "online"))
                    },
                    replay_episodes=batch,
                    global_batch=len(batch),
                )
            )
    stats = _gather(
        dict(
            target_seconds=target_seconds,
            regression_seconds=perf_counter() - fit_started,
            q_min=q_min,
            q_max=q_max,
            above=above,
            count=count,
            model_hash=model_hash(model),
            cache=cache.metrics(),
        ),
        world,
    )
    if len({s["model_hash"] for s in stats}) != 1:
        raise ValueError("distributed learner weights diverged")
    if rank == 0:
        metrics = dict(
            loss=losses / sum(map(len, orders)),
            transitions=sum(lengths),
            transition_visits=sum(lengths) + sum(lengths[i] for order in orders for i in order),
            regression_transition_visits=sum(lengths[i] for order in orders for i in order),
            optimizer_updates=updates,
            regression_passes=len(orders),
            regression_episode_visits=sum(map(len, orders)),
            episode_transition_counts=lengths,
            target_seconds=max(s["target_seconds"] for s in stats),
            regression_seconds=max(s["regression_seconds"] for s in stats),
            target_min=min(min(t) for t in fixed.values()),
            target_max=max(max(t) for t in fixed.values()),
            target_hash_before=before,
            target_hash_after=model_hash(frozen),
            q_min=min(s["q_min"] for s in stats),
            q_max=max(s["q_max"] for s in stats),
            q_above_one_fraction=sum(s["above"] for s in stats) / sum(s["count"] for s in stats),
            ranks=world,
            global_batch=options["global_batch"],
            rank_metrics=stats,
        )
        torch.save(
            dict(model={k: v.cpu() for k, v in model.state_dict().items()}, metrics=metrics),
            directory / "result.pt",
        )


def _guarded_worker(rank, *args):
    try:
        _worker(rank, *args)
    except BaseException:
        failure = Path(args[-1]) / f"failure-{rank}.txt"
        if not failure.exists():
            failure.write_text(traceback.format_exc())
        raise


@contextmanager
def worker_processes(devices, state, episodes, options, orders, directory):
    processes = []
    context = mp.get_context("spawn")
    try:
        for rank in range(len(devices)):
            process = context.Process(
                target=_guarded_worker,
                args=(rank, devices, state, episodes, options, orders, directory),
            )
            processes.append(process)
            process.start()
        yield processes
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


def fit_distributed(
    model,
    episodes,
    *,
    devices,
    mode,
    epochs=2,
    learning_rate=3e-4,
    conservative_weight=0.01,
    global_batch=4,
    chunk_size=64,
    sources=None,
    cache_bytes=128 * 2**30,
    cpu_threads=4,
    optimizer_updates=None,
    progress=None,
):
    """All ranks exit before returning, so validation owns clean GPU contexts."""
    if (
        not devices
        or global_batch < 1
        or epochs < 1
        or not episodes
        or cache_bytes < 0
        or type(cpu_threads) is not int
        or cpu_threads < 1
    ):
        raise ValueError("invalid distributed FQI recipe")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("distributed coordinator model must remain on CPU")
    if optimizer_updates is not None and (
        type(optimizer_updates) is not int or optimizer_updates < 1
    ):
        raise ValueError("optimizer update budget must be positive")
    options = dict(
        mode=mode,
        learning_rate=learning_rate,
        conservative_weight=conservative_weight,
        global_batch=global_batch,
        chunk_size=chunk_size,
        sources=sources,
        cache_bytes=cache_bytes,
        cpu_threads=cpu_threads,
    )
    if optimizer_updates is None:
        orders = [np.random.permutation(len(episodes)).tolist() for _ in range(epochs)]
    else:
        orders = []
        remaining = optimizer_updates
        while remaining:
            order = np.random.permutation(len(episodes)).tolist()[: remaining * global_batch]
            orders.append(order)
            remaining -= (len(order) + global_batch - 1) // global_batch
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    from latency_meta_mdp.meta.information.resources import (
        gpu_lock_directory,
        gpu_processes,
        interrupt_on_termination,
        worker_lock,
    )

    with ExitStack() as stack:
        stack.enter_context(interrupt_on_termination())
        if any(torch.device(d).type == "cuda" for d in devices):
            physical = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if len(physical) != len(devices) or any(not d.isdigit() for d in physical):
                raise ValueError("set explicit CUDA_VISIBLE_DEVICES for learner ranks")
            for d in sorted(physical, key=int):
                stack.enter_context(worker_lock(gpu_lock_directory(d)))
            if gpu_processes()[0]:
                raise RuntimeError("selected learner GPU already active")
        directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="conveyor-fqi-"))
        offset = 0
        target_offsets, seen = [0] * len(devices), set()

        def drain():
            nonlocal offset
            for rank in range(len(devices)):
                path = Path(directory) / f"target-{rank}.jsonl"
                if not path.exists():
                    continue
                with path.open() as stream:
                    stream.seek(target_offsets[rank])
                    for line in stream:
                        if not line.endswith("\n"):
                            break
                        seen.add(json.loads(line)["index"])
                        target_offsets[rank] += len(line.encode())
                        if progress is not None:
                            progress(
                                dict(phase="target", episode=len(seen), episodes=len(episodes))
                            )
            path = Path(directory) / "progress.jsonl"
            if path.exists():
                with path.open() as stream:
                    stream.seek(offset)
                    for line in stream:
                        if not line.endswith("\n"):
                            break
                        if progress is not None:
                            progress(json.loads(line))
                        offset += len(line.encode())

        with worker_processes(devices, state, episodes, options, orders, directory) as processes:
            while True:
                drain()
                for rank, process in enumerate(processes):
                    if process.exitcode not in (None, 0):
                        failures = sorted(Path(directory).glob("failure-*.txt"))
                        detail = "\n".join(f"{p.name}: {p.read_text()}" for p in failures)
                        detail = detail or str(process.exitcode)
                        raise RuntimeError(f"FQI rank {rank} failed: {detail}")
                if all(p.exitcode == 0 for p in processes):
                    break
                sleep(0.1)
            drain()
            result = torch.load(
                Path(directory) / "result.pt", map_location="cpu", weights_only=True
            )
            model.load_state_dict(result["model"])
            return result["metrics"]


def _attention_worker(rank, world, device, state, replay, options, directory):
    from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.attention_replay import DecisionSampler

    model = AttentionStageQ(options["config"]).to(device)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.Adam(model.parameters())
    # Spawn shares CPU tensor storage. Adam keeps its step counters on CPU even
    # for CUDA parameters, so each rank must own a private restored state.
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    sampler = DecisionSampler(replay, seed=0)
    sampler.load_state_dict(state["sampler"])

    def report(event):
        if rank == 0:
            with (directory / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")

    metrics = fit_attention_iteration(
        model,
        replay,
        optimizer=optimizer,
        sampler=sampler,
        rank=rank,
        world_size=world,
        progress=report,
        **options["fit"],
    )
    ranks = _gather(
        dict(model_hash=model_hash(model), target_seconds=metrics["target_seconds"]), world
    )
    if len({r["model_hash"] for r in ranks}) != 1:
        raise ValueError("attention learner ranks diverged")
    if rank == 0:
        metrics.update(ranks=world, rank_metrics=ranks, global_batch=options["fit"]["global_batch"])
        torch.save(
            dict(
                model=model.cpu().state_dict(),
                optimizer=optimizer.state_dict(),
                sampler=sampler.state_dict(),
                metrics=metrics,
            ),
            directory / "result.pt",
        )


def fit_attention_distributed(
    model, replay, *, devices, optimizer, sampler, cpu_threads=4, progress=None, **fit
):
    """Disposable ranks; the coordinator retains weights, Adam and sampler state."""
    from latency_meta_mdp.meta.information.memory_schema import validate_batch
    from latency_meta_mdp.meta.information.resources import (
        gpu_lock_directory,
        gpu_processes,
        interrupt_on_termination,
        worker_lock,
    )

    if not devices or next(model.parameters()).device.type != "cpu" or cpu_threads < 1:
        raise ValueError("attention coordinator must be CPU with explicit learner devices")
    validate_batch(
        global_batch=fit["global_batch"],
        world_size=len(devices),
        microbatch=fit["microbatch"],
        accumulation=fit["global_batch"] // (len(devices) * fit["microbatch"]),
    )
    state = dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict()
    )
    options = dict(architecture="attention", config=model.config, fit=fit, cpu_threads=cpu_threads)
    with ExitStack() as stack:
        stack.enter_context(interrupt_on_termination())
        if any(torch.device(d).type == "cuda" for d in devices):
            physical = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if len(physical) != len(devices) or any(not d.isdigit() for d in physical):
                raise ValueError("set explicit CUDA_VISIBLE_DEVICES for learner ranks")
            for device in sorted(physical, key=int):
                stack.enter_context(worker_lock(gpu_lock_directory(device)))
            if gpu_processes()[0]:
                raise RuntimeError("selected learner GPU already active")
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="attention-fqi-")))
        offset = 0

        def drain():
            nonlocal offset
            path = directory / "progress.jsonl"
            if path.exists():
                with path.open() as stream:
                    stream.seek(offset)
                    for line in stream:
                        if not line.endswith("\n"):
                            break
                        if progress:
                            progress(json.loads(line))
                        offset += len(line.encode())

        with worker_processes(devices, state, replay, options, None, str(directory)) as processes:
            while True:
                drain()
                if any(p.exitcode not in (None, 0) for p in processes):
                    details = "\n".join(p.read_text() for p in directory.glob("failure-*.txt"))
                    raise RuntimeError(f"attention FQI worker failed: {details}")
                if all(p.exitcode == 0 for p in processes):
                    break
                sleep(0.1)
            drain()
        result = torch.load(directory / "result.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(result["model"])
        optimizer.load_state_dict(result["optimizer"])
        sampler.load_state_dict(result["sampler"])
        return result["metrics"]
