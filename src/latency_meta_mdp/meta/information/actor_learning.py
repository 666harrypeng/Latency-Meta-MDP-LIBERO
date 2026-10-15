"""Synchronous IQL/actor/PPO fitting with exact global microbatch weights and recovery."""

import copy
import json
import math
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latency_meta_mdp.meta.information.actor_critic import (
    AttentionValue,
    categorical,
    expectile_loss,
    ppo_loss,
    update_target,
)
from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.config import identity_hash, sha256
from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.memory import collate_contexts
from latency_meta_mdp.meta.information.validation import atomic_json


def balanced_probabilities(lengths, episodes):
    if len(lengths) != len(episodes) or not lengths or min(lengths) < 1:
        raise ValueError("nonempty episode lengths and matching metadata required")
    groups = {}
    offset = 0
    for n, ep in zip(lengths, episodes, strict=True):
        mode = (ep["behavior"], json.dumps(ep.get("pattern", {}), sort_keys=True))
        groups.setdefault(ep["regime"], {}).setdefault(mode, []).extend(range(offset, offset + n))
        offset += n
    probabilities = np.zeros(offset, np.float64)
    for modes in groups.values():
        for indices in modes.values():
            probabilities[indices] = 1 / (len(groups) * len(modes) * len(indices))
    return probabilities


def initial_state(config, *, seed=0, normalization=None):
    torch.manual_seed(seed)
    q1, q2 = [AttentionStageQ(config, normalization=normalization) for _ in range(2)]
    value = AttentionValue(config, normalization=normalization)
    return dict(
        config=config,
        phase="iql",
        step=0,
        epoch=0,
        optimizers={},
        models={
            k: m.state_dict()
            for k, m in dict(
                q1=q1, q2=q2, v=value, tq1=copy.deepcopy(q1), tq2=copy.deepcopy(q2)
            ).items()
        },
    )


def restore_models(state, device="cpu"):
    models = {}
    for name, weights in state["models"].items():
        cls = AttentionValue if name == "v" else AttentionStageQ
        model = cls(state["config"]).to(device)
        model.load_state_dict(weights, strict=True)
        models[name] = model
    return models


def save_state(directory, state):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step-{state['step']:08d}-epoch-{state['epoch']:04d}"
    if target.exists():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    torch.save(state, temporary / "state.pt")
    atomic_json(
        temporary / "complete.json",
        dict(
            phase=state["phase"],
            step=state["step"],
            epoch=state["epoch"],
            binding=state.get("binding", {}),
            files={"state.pt": sha256(temporary / "state.pt")},
            metrics=state.get("metrics", {}),
        ),
    )
    temporary.rename(target)
    link = directory / ".latest.tmp"
    link.unlink(missing_ok=True)
    link.symlink_to(target.name, target_is_directory=True)
    link.replace(directory / "latest")
    return target


def load_state(checkpoint):
    checkpoint = Path(checkpoint)
    receipt = json.loads((checkpoint / "complete.json").read_text())
    if receipt["files"] != {"state.pt": sha256(checkpoint / "state.pt")}:
        raise ValueError("actor/critic checkpoint content hash differs")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    if any(state[k] != receipt[k] for k in ("phase", "step", "epoch")):
        raise ValueError("actor/critic checkpoint counters differ")
    return state


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value


def fit(
    phase,
    state,
    replay,
    directory,
    *,
    options,
    devices,
    auxiliary=None,
    probabilities=None,
    progress=None,
):
    """One stage per process group; the CPU coordinator never reserves accelerator memory."""
    world, batch = len(devices), options["global_batch"]
    if world < 1 or batch < 1 or batch % world:
        raise ValueError("global batch must be positive and divisible by learner world size")
    if options["microbatch"] < 1:
        raise ValueError("positive microbatch required")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state = copy.deepcopy(state)
    if state["phase"] != phase:
        state.update(phase=phase, step=0, epoch=0, optimizers={})
        state.pop("actor_stopped", None)
        state.pop("fit_binding", None)
    binding = identity_hash(
        dict(
            phase=phase,
            config=state["config"],
            options={
                k: v
                for k, v in options.items()
                if k not in ("updates", "epochs", "checkpoint_every", "microbatch", "cpu_threads")
            },
            data=state.get("binding", {}),
        )
    )
    if state.get("fit_binding") not in (None, binding):
        raise ValueError("resume training recipe or dataset differs")
    state["fit_binding"] = binding
    with tempfile.TemporaryDirectory(prefix=".ranks-", dir=directory) as work:
        args = (
            devices,
            phase,
            state,
            replay,
            str(directory),
            str(work),
            options,
            auxiliary,
            probabilities,
        )
        if world == 1 and devices == ["cpu"]:
            _worker(0, *args)
        else:
            context = mp.spawn(_worker, args=args, nprocs=world, join=False)
            offset = (
                (directory / "metrics.jsonl").stat().st_size
                if (directory / "metrics.jsonl").exists()
                else 0
            )
            while True:
                finished = context.join(timeout=5)
                log = directory / "metrics.jsonl"
                if progress is not None and log.exists():
                    with log.open() as stream:
                        stream.seek(offset)
                        for line in stream:
                            progress(json.loads(line))
                        offset = stream.tell()
                if finished:
                    break
    return load_state(directory / "latest")


def _worker(rank, devices, phase, state, replay, directory, work, options, aux, probabilities):
    world, device = len(devices), torch.device(devices[rank])
    torch.set_num_threads(options.get("cpu_threads", 2))
    torch.manual_seed(options.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            init_method=f"file://{work}/rendezvous",
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=20),
        )
    try:
        _learn(
            rank,
            world,
            device,
            phase,
            state,
            replay,
            Path(directory),
            options,
            aux or {},
            probabilities,
        )
    finally:
        if world > 1:
            dist.destroy_process_group()


def _sync_step(model, optimizer, world, clip):
    parameters = list(model.parameters())
    gradient = torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters]
    )
    used = torch.tensor(
        [p.grad is not None for p in parameters], device=gradient.device, dtype=torch.int32
    )
    if world > 1:
        dist.all_reduce(gradient)
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
    offset = 0
    for p, active in zip(parameters, used.cpu().tolist(), strict=True):
        p.grad = gradient[offset : offset + p.numel()].view_as(p) if active else None
        offset += p.numel()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def _learn(rank, world, device, phase, state, replay, directory, options, aux, probabilities):
    models = restore_models(state, device)
    active = dict(iql=["q1", "q2", "v"], actor=["actor"], critic=["v"], ppo=["actor", "v"])[phase]
    for name, model in models.items():
        model.train(name in active).requires_grad_(name in active)
    optimizers = {
        name: torch.optim.Adam(
            models[name].parameters(),
            lr=options.get("actor_lr", options.get("learning_rate", 1e-4))
            if name == "actor"
            else options.get("learning_rate", 1e-4),
        )
        for name in active
    }
    for name, optimizer in optimizers.items():
        if name in state["optimizers"]:
            optimizer.load_state_dict(state["optimizers"][name])
    n, batch_size = len(replay), options["global_batch"]
    batches = math.ceil(n / batch_size)
    total = options["updates"] if phase in ("iql", "actor") else options["epochs"] * batches
    start = state["step"]
    if start > total:
        raise ValueError("checkpoint exceeds requested training budget")
    if phase in ("actor", "iql") and probabilities is None:
        probabilities = np.full(n, 1 / n)
    actor_stopped = state.get("actor_stopped", False)
    counts = dict(state.get("optimizer_updates", {}))
    visits = state.get("sample_visits", 0)
    epoch_order = None
    for step in range(start, total):
        epoch, within = divmod(step, batches)
        if phase in ("iql", "actor"):
            order = np.random.default_rng(options.get("seed", 0) + step).choice(
                n, batch_size, p=probabilities
            )
        else:
            if epoch_order is None or within == 0:
                epoch_order = np.random.default_rng(options.get("seed", 0) + epoch).permutation(n)
            order = epoch_order[within * batch_size : (within + 1) * batch_size]
        names = [name for name in active if not (name == "actor" and actor_stopped)]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(5, dtype=torch.float64, device=device)
        own = order[rank::world]
        for begin in range(0, len(own), options["microbatch"]):
            indices = own[begin : begin + options["microbatch"]]
            rows = [replay[int(i)] for i in indices]
            context = collate_contexts(
                [r["context"] for r in rows], device=device, config=state["config"]["memory"]
            )
            actions = torch.tensor([r["action"] for r in rows], device=device)
            legal = torch.as_tensor(np.stack([r["legal"] for r in rows]), device=device)
            if not legal.gather(1, actions[:, None]).all():
                raise ValueError("illegal training action")
            losses = {}
            if phase == "iql":
                with torch.no_grad():
                    qbar = torch.minimum(models["tq1"](context), models["tq2"](context))
                    chosen = qbar.gather(1, actions[:, None]).squeeze(1)
                    target = torch.tensor([r["task_reward"] for r in rows], device=device)
                    future = [i for i, r in enumerate(rows) if not r["terminated"]]
                    if future:
                        if any(rows[i]["next_context"] is None for i in future):
                            raise ValueError("nonterminal transition missing successor")
                        next_batch = collate_contexts(
                            [rows[i]["next_context"] for i in future],
                            device=device,
                            config=state["config"]["memory"],
                        )
                        target[future] += models["v"](next_batch)
                losses["v"] = expectile_loss(models["v"](context), chosen, options.get("tau", 0.7))
                for name in ("q1", "q2"):
                    q = models[name](context).gather(1, actions[:, None]).squeeze(1)
                    losses[name] = (q - target).square()
            elif phase == "actor":
                weights = torch.as_tensor(aux["weights"][indices], device=device)
                losses["actor"] = -categorical(models["actor"](context), legal).log_prob(actions)
                losses["actor"] = losses["actor"] * weights / aux["weight_mean"]
            else:
                target = torch.as_tensor(aux["returns"][indices], device=device)
                losses["v"] = (models["v"](context) - target).square()
                if phase == "ppo":
                    with torch.set_grad_enabled(not actor_stopped):
                        losses["actor"], metrics = ppo_loss(
                            models["actor"](context),
                            torch.as_tensor(aux["old_logits"][indices], device=device),
                            legal,
                            actions,
                            torch.as_tensor(aux["advantage"][indices], device=device),
                            clip=options.get("clip", 0.2),
                            entropy_coefficient=options.get("entropy_coefficient", 0.001),
                        )
                    sums[1] += metrics["kl"].double().sum()
                    sums[2] += metrics["entropy"].double().sum()
                    sums[3] += metrics["clip_fraction"].double().sum()
            for name in names:
                loss = losses[name].sum() / len(order)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite actor/critic objective")
                loss.backward()
                sums[0] += loss.detach().double()
            sums[4] += len(rows)
        if world > 1:
            dist.all_reduce(sums)
        # All ranks make the same stop decision, before this minibatch's actor step.
        if phase == "ppo" and float(sums[1] / sums[4]) > 1.5 * options.get("target_kl", 0.01):
            actor_stopped = True
            names = [name for name in names if name != "actor"]
        norms = {
            name: _sync_step(models[name], optimizers[name], world, options.get("grad_clip", 0.5))
            for name in names
        }
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        visits += len(order)
        if phase == "iql":
            for name in ("q1", "q2"):
                update_target(models["t" + name], models[name], options.get("ema", 0.005))
        metrics = dict(
            phase=phase,
            step=step + 1,
            epoch=epoch + 1,
            loss=float(sums[0]),
            kl=float(sums[1] / sums[4]),
            entropy=float(sums[2] / sums[4]),
            clip_fraction=float(sums[3] / sums[4]),
            actor_stopped=actor_stopped,
            samples=int(sums[4]),
            grad_norm=norms,
            actor_updates=counts.get("actor", 0),
            value_updates=counts.get("v", 0),
            sample_visits=visits,
        )
        if rank == 0 and (step % 10 == 0 or step + 1 == total):
            print("ACTOR_LEARN " + json.dumps(metrics), flush=True)
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
        boundary = (
            (step + 1) % options.get("checkpoint_every", 500) == 0
            if phase in ("iql", "actor")
            else within + 1 == batches
        )
        if boundary or step + 1 == total:
            hashes = {name: model_hash(model) for name, model in models.items()}
            all_hashes = [None] * world
            if world > 1:
                dist.all_gather_object(all_hashes, hashes)
                if any(h != hashes for h in all_hashes):
                    raise ValueError("actor/critic rank parameters diverged")
            if rank == 0:
                state.update(
                    step=step + 1,
                    epoch=(step + 1) // batches,
                    actor_stopped=actor_stopped,
                    models={k: _cpu(m.state_dict()) for k, m in models.items()},
                    optimizers={k: _cpu(o.state_dict()) for k, o in optimizers.items()},
                    metrics=metrics,
                    model_hashes=hashes,
                    torch_rng=torch.get_rng_state(),
                    optimizer_updates=counts,
                    sample_visits=visits,
                )
                save_state(directory, state)
            if world > 1:
                dist.barrier()
    if start == total and not (directory / "latest").exists() and rank == 0:
        save_state(directory, state)
