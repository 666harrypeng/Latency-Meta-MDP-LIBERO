"""Immutable scalar teachers and one-policy rollout batches; no learned feature cache."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from latency_meta_mdp.meta.information.actor_critic import (
    advantage_weights,
    categorical,
    returns_to_go,
)
from latency_meta_mdp.meta.information.actor_inputs import InputPipeline
from latency_meta_mdp.meta.information.actor_learning import restore_models
from latency_meta_mdp.meta.information.config import identity_hash, sha256, validate_binding
from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.validation import atomic_json


def score_models(state, replay, directory, *, devices, microbatch, kind, binding, cpu_threads=2):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(
        data=binding, models={k: model_hash(m) for k, m in restore_models(state).items()}, kind=kind
    )
    receipt_path = directory / "complete.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        validate_binding(identity, receipt["binding"])
        validate_binding(receipt["sha256"], sha256(directory / "scalars.npz"))
        with np.load(directory / "scalars.npz") as data:
            return dict(data)
    with tempfile.TemporaryDirectory(prefix=".scoring-", dir=directory) as work:
        args = (state, replay, devices, microbatch, kind, work, cpu_threads)
        # Always spawn, including one GPU: caller remains a CPU coordinator.
        mp.spawn(_score_worker, args=args, nprocs=len(devices), join=True)
        arrays = {}
        seen = np.zeros(len(replay), bool)
        for rank in range(len(devices)):
            with np.load(Path(work) / f"{rank}.npz") as shard:
                indices = shard["indices"]
                if seen[indices].any():
                    raise ValueError("duplicate scalar score index")
                seen[indices] = True
                for key in shard.files:
                    if key != "indices":
                        arrays.setdefault(
                            key,
                            np.empty((len(replay), *shard[key].shape[1:]), dtype=shard[key].dtype),
                        )[indices] = shard[key]
        if not seen.all():
            raise ValueError("incomplete scalar teacher scoring")
    np.savez(directory / "scalars.npz", **arrays)
    atomic_json(receipt_path, dict(binding=identity, sha256=sha256(directory / "scalars.npz")))
    return arrays


def _score_worker(rank, state, replay, devices, microbatch, kind, work, cpu_threads):
    torch.set_num_threads(cpu_threads)
    device = torch.device(devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    needed = ("tq1", "tq2", "v") if kind == "iql" else ("actor", "v")
    models = restore_models(dict(state, models={k: state["models"][k] for k in needed}), device)
    for model in models.values():
        model.eval().requires_grad_(False)
    # Contiguous episode/context visits avoid thrashing packed feature mmaps.
    indices = np.array_split(np.arange(len(replay)), len(devices))[rank]
    output = {}
    with (
        torch.inference_mode(),
        InputPipeline(
            replay,
            state["config"]["memory"],
            microbatch=microbatch,
            pin_memory=device.type == "cuda",
        ) as inputs,
    ):
        for step, prepared in enumerate(inputs.batches(indices, successors=False)):
            prepared = prepared.to(device)
            rows, batch = prepared.rows, prepared.context
            value = models["v"](batch)
            if kind == "iql":
                actions = torch.tensor([r["action"] for r in rows], device=device)
                q = torch.minimum(models["tq1"](batch), models["tq2"](batch))
                values = dict(advantage=q.gather(1, actions[:, None]).squeeze(1) - value)
            else:
                values = dict(old_logits=models["actor"](batch), old_values=value)
            for key, val in values.items():
                output.setdefault(key, []).append(val.cpu().numpy())
            if step % max(1, 1024 // microbatch) == 0:
                print(
                    f"ACTOR_SCORE rank={rank} kind={kind} "
                    f"decisions={step * microbatch}/{len(indices)}",
                    flush=True,
                )
    np.savez(
        Path(work) / f"{rank}.npz",
        indices=indices,
        **{k: np.concatenate(v) for k, v in output.items()},
    )


def offline_weights(scores, probabilities, *, temperature, maximum):
    advantage = torch.from_numpy(scores["advantage"])
    weights = advantage_weights(advantage, temperature, maximum).numpy()
    mean = float(np.dot(weights, probabilities))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid offline actor weight normalization")
    diagnostics = dict(
        advantage_quantiles=np.quantile(advantage.numpy(), [0, 0.1, 0.5, 0.9, 1]).tolist(),
        weight_mean=mean,
        clipped_fraction=float(np.mean(weights >= maximum)),
        normalized_ess=float(mean**2 / np.dot(weights**2, probabilities)),
    )
    return dict(weights=weights, weight_mean=mean), diagnostics


def on_policy_auxiliary(replay, scores, episodes, summaries, *, actor_hash, logprob_tolerance=2e-4):
    if len(episodes) != len(replay.episodes) or len(summaries) != len(episodes):
        raise ValueError("rollout metadata does not cover current batch")
    returns, observed, offset = [], [], 0
    for i, (ep, summary) in enumerate(zip(episodes, summaries, strict=True)):
        rows = replay.episodes[i]
        diagnostics = summary["scheduler_diagnostics"]
        if ep["behavior"] != "actor" or ep["deterministic"] or len(rows) != len(diagnostics):
            raise ValueError("only fresh stochastic actor decisions admitted")
        returns.extend(
            returns_to_go([r["task_reward"] for r in rows], [r["terminated"] for r in rows])
        )
        for j, (row, old) in enumerate(zip(rows, diagnostics, strict=True)):
            if (
                old["policy_version"] != actor_hash
                or old["action"] != row["action"]
                or old["tick"] != row["packet"].tick
                or old["stage"] != row["packet"].stage
                or old["deterministic"]
                or not np.array_equal(old["legal"], row["legal"])
                or not np.isclose(old["probability"], row["probability"], atol=1e-8, rtol=1e-6)
            ):
                raise ValueError("on-policy provenance or causal context mismatch")
            # Verify all categorical probabilities, not only the chosen action.
            mask = torch.tensor(row["legal"])[None]
            recomputed = categorical(torch.tensor(scores["old_logits"][offset + j])[None], mask)
            recorded = categorical(torch.tensor(old["logits"])[None], mask)
            delta = (recomputed.logits - recorded.logits)[mask].abs().max().item()
            action_logp = float(recomputed.log_prob(torch.tensor([row["action"]]))[0])
            if (
                delta > logprob_tolerance
                or abs(action_logp - old["log_probability"]) > logprob_tolerance
            ):
                raise ValueError("old actor log probability does not roundtrip from public history")
            observed.append(old["logits"])
        offset += len(rows)
    returns = np.asarray(returns, np.float32)
    raw = returns - scores["old_values"]
    return dict(
        returns=returns,
        old_logits=np.asarray(observed, np.float32),
        advantage=(raw - raw.mean()) / max(float(raw.std()), 1e-8),
        raw_advantage=raw,
        old_values=scores["old_values"],
    )


def scalar_binding(state, conversion):
    return dict(conversion=identity_hash(conversion), actor_version=state.get("actor_version"))
