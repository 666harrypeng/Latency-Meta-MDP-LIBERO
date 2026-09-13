"""Fit a small Launch/Wait Q model on frozen-policy decision-stage replay."""

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from latency_meta_mdp.meta_learning_state import (
    learning_rate_at,
    load_learning_state,
    save_learning_state,
)
from latency_meta_mdp.meta_q import MetaQNetwork, fitted_q_targets
from latency_meta_mdp.meta_replay import FEATURE_FORMAT

BINDING_KEYS = (
    "checkpoint_step",
    "checkpoint_verification_sha256",
    "forecast_identity",
    "client_config_sha256",
    "rtc_calibration",
    "rtc_max_guidance_weight",
    "decision_interval_ticks",
    "discount_per_tick",
    "runtime_patch_sha256",
)


def load_replay(root):
    states, vectors, legal, records = [], [], [], []
    binding = None
    masters = {"train": set(), "validation": set()}
    offset = 0
    inventory = []
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(d["metadata_utf8"].tobytes())
            if meta["feature_format"] != FEATURE_FORMAT:
                raise ValueError("Meta replay feature format differs")
            current = {key: meta["identity"][key] for key in BINDING_KEYS}
            if binding is not None and current != binding:
                raise ValueError("Meta replay mixes policy, forecast or timing identities")
            binding = current
            partition = meta["case"]["meta_partition"]
            masters[partition].add(meta["case"]["master_index"])
            x, v, mask = d["visual"], d["vector"], d["legal_actions"]
            if x.shape[1:] != (2, 2, 196, 384) or v.shape != (len(x), 501):
                raise ValueError("Meta replay feature shape mismatch")
            if not np.isfinite(x).all() or not np.isfinite(v).all():
                raise ValueError("nonfinite replay features")
            states.append(x)
            vectors.append(v)
            legal.append(mask)
            selected = ~d["truncated"]
            index = d["state_index"][selected]
            next_index = d["next_state_index"][selected]
            terminated = d["terminated"][selected]
            if (
                np.any(index < 0)
                or np.any(index >= len(x))
                or np.any(next_index >= len(x))
                or np.any(d["duration_ticks"] <= 0)
            ):
                raise ValueError("Meta replay has invalid state indices or duration")
            if (
                np.any((next_index < 0) & ~terminated)
                or not mask[index, d["action"][selected]].all()
            ):
                raise ValueError("replay lacks a valid decision successor or has an illegal action")
            discounts = d["bootstrap_discount"][selected]
            if not np.allclose(
                discounts,
                np.where(
                    terminated, 0, binding["discount_per_tick"] ** d["duration_ticks"][selected]
                ),
            ):
                raise ValueError("Meta replay discount does not match physical duration")
            records.append(
                {
                    "state": index + offset,
                    "next": np.where(next_index < 0, 0, next_index + offset),
                    "action": d["action"][selected],
                    "reward": d["reward"][selected],
                    "discount": discounts,
                    "train": np.full(len(index), partition == "train", bool),
                }
            )
            inventory.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "partition": partition,
                    "states": len(x),
                    "transitions": len(index),
                    "excluded_truncated": int((~selected).sum()),
                }
            )
            offset += len(x)
    if not records or not all(masters.values()) or masters["train"] & masters["validation"]:
        raise ValueError("Meta fitting requires nonempty disjoint grouped train/validation data")
    return (
        np.concatenate(states),
        np.concatenate(vectors),
        np.concatenate(legal),
        {key: np.concatenate([r[key] for r in records]) for key in records[0]},
        binding,
        inventory,
        {k: sorted(v) for k, v in masters.items()},
    )


def meta_batch_predictions(model, target, visual, vector, records, admissible, indices, cfg):
    """One minibatch; replay may live on CPU while Q runs on GPU."""
    device = next(model.parameters()).device
    current, nxt = records["state"][indices], records["next"][indices]
    actions = records["action"][indices].long().to(device)
    next_visual, next_vector = visual[nxt].to(device), vector[nxt].to(device)
    amp = device.type == "cuda" and cfg.get("q_precision", "bfloat16") != "float32"
    # A no-grad online forward can populate AMP's weight cache with detached
    # casts. Close that scope before the differentiable online forward.
    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp),
    ):
        labels = fitted_q_targets(
            reward=records["reward"][indices].float().to(device),
            action=actions,
            discount=records["discount"][indices].float().to(device),
            next_online=model(next_visual, next_vector).float(),
            next_target=target(next_visual, next_vector).float(),
            next_legal=admissible[nxt].to(device),
            call_cost=cfg["call_cost"],
            forecast_cost=cfg["forecast_cost"],
        )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        q = model(visual[current].to(device), vector[current].to(device)).float()
        predicted = q.gather(1, actions[:, None])[:, 0]
    return predicted, labels, q


def meta_td_loss(model, target, visual, vector, records, admissible, indices, cfg):
    prediction, labels, _ = meta_batch_predictions(
        model, target, visual, vector, records, admissible, indices, cfg
    )
    return torch.nn.functional.smooth_l1_loss(prediction, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay-root", type=Path)
    source.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--visual-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-future", action="store_true")
    args = parser.parse_args()
    if (args.replay_manifest is None) != (args.training_config is None):
        parser.error("success replay manifest requires its explicit training config")
    if args.replay_root is not None:
        collection = json.loads((args.replay_root / "status.json").read_text())
        if collection["status"] != "completed":
            raise ValueError("Meta training requires the completed collection manifest")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume_from is not None)
    torch.set_num_threads(4)
    torch.manual_seed(27)
    device = torch.device(args.device)
    if args.replay_manifest is not None:
        from latency_meta_mdp.meta_success_data import load_success_replay

        x, v, legal, data, binding, inventory, masters = load_success_replay(
            args.replay_manifest, visual_cache=args.visual_cache
        )
    else:
        x, v, legal, data, binding, inventory, masters = load_replay(args.replay_root)
    if args.replay_root is not None and (
        len(inventory) != 120 or len(masters["train"]) != 48 or len(masters["validation"]) != 12
    ):
        raise ValueError("first Meta pilot requires the complete 48/12-master120-episode cohort")
    cfg = {
        "feature_format": FEATURE_FORMAT,
        "use_future": not args.no_future,
        "policy_binding": binding,
        "seed": 27,
        "updates": 3000,
        "batch_size": 128,
        "target_update_interval": 100,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "call_cost": 0.01,
        "forecast_cost": 0.002,
        "gradient_clip": 1.0,
        "visual_projection": 16,
        "hidden_widths": [128, 64],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    if args.training_config is not None:
        import yaml

        requested = yaml.safe_load(args.training_config.read_text())
        if (requested["objective"] != "finite_horizon_success_v2" or requested["gamma"] != 1
                or requested["call_cost"] != 0 or requested["forecast_cost"] != 0
                or requested["task_horizon_ticks"] != 1000):
            raise ValueError("invalid success-only objective specification")
        cfg.update(requested)
        cfg["replay_manifest"] = str(args.replay_manifest.resolve())
    torch.manual_seed(cfg["seed"])
    train_states = np.unique(data["state"][data["train"]])
    mean = v[train_states].mean(axis=0)
    scale = v[train_states].std(axis=0).clip(0.01)
    model = MetaQNetwork(vector_mean=mean, vector_scale=scale, use_future=not args.no_future).to(
        device
    )
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    start_step = 0
    if args.resume_from is not None:
        start_step = load_learning_state(args.resume_from, model, target, optimizer, cfg)
        if start_step >= cfg["updates"]:
            raise ValueError("requested end step must exceed restored progress")
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    inventory_name = ("replay-inventory.json" if start_step == 0
                      else f"replay-inventory-after-{start_step:06d}.json")
    (args.output_dir / inventory_name).write_text(
        json.dumps({"files": inventory, "masters": masters}, indent=2)
    )
    # Keep the large expanded replay on host when it would crowd training activations.
    replay_device = device
    if device.type == "cuda" and x.nbytes + 3 * 1024**3 > torch.cuda.mem_get_info(device)[0]:
        replay_device = torch.device("cpu")
    if isinstance(x, np.ndarray):
        visual = torch.from_numpy(x).to(replay_device)
    else:
        visual = x.to_tensor(replay_device) if replay_device.type == "cuda" else x
    vector = torch.from_numpy(v).to(replay_device)
    admissible = torch.from_numpy(legal).to(replay_device)
    records = {k: torch.as_tensor(a, device=replay_device) for k, a in data.items()}
    train = torch.where(records["train"])[0]
    validation = torch.where(~records["train"])[0]
    visited = torch.zeros(len(data["state"]), dtype=torch.bool, device=replay_device)
    probe, fixed_labels, previous_q = None, None, None
    if args.replay_manifest is not None:
        # Common old-data fit probe for both fits; validation is never used for SGD.
        old_count = sum(e["transitions"] for e in inventory if e.get("source") == "old")
        old_train = train[train < old_count] if old_count else train
        rng = np.random.default_rng(27)
        fit_probe = rng.choice(old_train.cpu().numpy(), min(256, len(old_train)), replace=False)
        probe = torch.as_tensor(
            np.concatenate((validation.cpu().numpy(), fit_probe)), device=replay_device
        )
        probe_path = args.output_dir / "probe-indices.npy"
        if args.resume_from is not None and probe_path.exists():
            probe = torch.as_tensor(np.load(probe_path), device=replay_device)
        else:
            np.save(probe_path, probe.cpu().numpy())
        if args.resume_from is not None:
            reference = args.output_dir / "probe-005000.npz"
            if reference.exists():
                with np.load(reference) as previous:
                    fixed_labels = torch.from_numpy(previous["fixed_target"].copy())
            probes = sorted(p for p in args.output_dir.glob('probe-*.npz')
                            if int(p.stem.split('-')[1]) <= start_step)
            if probes:
                with np.load(probes[-1]) as previous:
                    previous_q = torch.from_numpy(previous["q"].copy())
    run = None
    import wandb

    run = wandb.init(
        project="latency-meta-mdp-robosuite",
        name=args.output_dir.name,
        dir=str(args.output_dir),
        config=cfg,
        mode="online" if os.environ.get("WANDB_API_KEY") else "offline",
    )
    started = time.monotonic()

    def loss_for(indices):
        return meta_td_loss(model, target, visual, vector, records, admissible, indices, cfg)

    print(
        json.dumps(
            {
                "stage": "training",
                "states": len(x),
                "transitions": len(data["state"]),
                "train_transitions": len(train),
                "validation_transitions": len(validation),
                "parameters": sum(p.numel() for p in model.parameters()),
                "wandb_url": run.url,
                "restored_step": start_step,
                "replay_storage": ("mapped_shards" if not isinstance(x, np.ndarray)
                                   else str(replay_device)),
            }
        ),
        flush=True,
    )
    last = {}
    completed = False
    try:
        for step in range(start_step + 1, cfg["updates"] + 1):
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, cfg)
            draw_device = 'cpu' if cfg.get("recoverable_training") else replay_device
            draws = torch.randint(len(train), (cfg["batch_size"],), device=draw_device)
            indices = train[draws.to(replay_device)]
            visited[indices] = True
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(indices)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite Meta Q loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if step % cfg["target_update_interval"] == 0:
                target.load_state_dict(model.state_dict())
            if step == start_step + 1 or step % cfg.get("log_interval", 50) == 0:
                model.eval()
                with torch.no_grad():
                    values = [
                        float(loss_for(validation[i : i + 128]))
                        for i in range(0, len(validation), 128)
                    ]
                last = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "validation_bellman_loss": float(np.mean(values)),
                    "grad_norm": float(grad),
                    "elapsed_seconds": time.monotonic() - started,
                    "sample_draws": step * cfg["batch_size"],
                    "unique_fit_transitions_visited": int(visited.sum()),
                    "mean_draws_per_fit_transition": (
                        (step - start_step) * cfg["batch_size"] / len(train)
                    ),
                    "segment_start_step": start_step,
                    "segment_sample_draws": (step - start_step) * cfg["batch_size"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                if probe is not None and step % 500 == 0:
                    predictions, labels, qs = [], [], []
                    with torch.no_grad():
                        for batch in probe.split(128):
                            p, y, q = meta_batch_predictions(
                                model, target, visual, vector, records, admissible, batch, cfg
                            )
                            predictions.append(p.cpu())
                            labels.append(y.cpu())
                            qs.append(q.cpu())
                    prediction, label, q = map(torch.cat, (predictions, labels, qs))
                    if step == 5000:
                        fixed_labels = label.clone()
                    both_legal = admissible[records["state"][probe]].cpu().all(dim=1)
                    last.update(
                        probe_td_loss=float(torch.nn.functional.smooth_l1_loss(prediction, label)),
                        probe_q_min=float(q.min()), probe_q_max=float(q.max()),
                    )
                    if previous_q is not None:
                        last["probe_q_mean_drift"] = float((q - previous_q).abs().mean())
                        last["probe_legal_action_flip_fraction"] = float(
                            (q.argmax(1) != previous_q.argmax(1))[both_legal].float().mean()
                        ) if both_legal.any() else 0.0
                    if fixed_labels is not None:
                        last["probe_fixed_target_loss"] = float(
                            torch.nn.functional.smooth_l1_loss(prediction, fixed_labels)
                        )
                    previous_q = q
                    np.savez_compressed(
                        args.output_dir / f"probe-{step:06d}.npz",
                        q=q.numpy(), prediction=prediction.numpy(), target=label.numpy(),
                        fixed_target=(fixed_labels.numpy() if fixed_labels is not None
                                      else np.array([], dtype=np.float32)),
                    )
                print(json.dumps(last), flush=True)
                with (args.output_dir / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(last) + "\n")
                run.log(last, step=step)
            if cfg.get("recoverable_training") and (
                step % cfg.get("save_interval", 6000) == 0 or step == cfg["updates"]
            ):
                save_learning_state(
                    args.output_dir / "recovery.pt", model, target, optimizer, step, cfg
                )
                checkpoint = args.output_dir / "checkpoints" / f"step-{step:06d}"
                checkpoint.mkdir(parents=True, exist_ok=False)
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                          checkpoint / "model.safetensors")
                (checkpoint / "config.json").write_text(
                    json.dumps({**cfg, "trained_step": step}, indent=2)
                )
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        if any(not torch.isfinite(v).all() for v in weights.values()):
            raise ValueError("nonfinite final Meta model")
        save_file(weights, args.output_dir / "model.safetensors")
        (args.output_dir / "training-summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    **last,
                    "train_masters": len(masters["train"]),
                    "validation_masters": len(masters["validation"]),
                    "selected_on_reportable_development": False,
                    "wandb_url": run.url,
                },
                indent=2,
            )
        )
        completed = True
    finally:
        run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
