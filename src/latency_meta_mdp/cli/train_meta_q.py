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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-future", action="store_true")
    args = parser.parse_args()
    collection = json.loads((args.replay_root / "status.json").read_text())
    if collection["status"] != "completed":
        raise ValueError("Meta training requires the completed collection manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(27)
    device = torch.device(args.device)
    x, v, legal, data, binding, inventory, masters = load_replay(args.replay_root)
    if len(inventory) != 120 or len(masters["train"]) != 48 or len(masters["validation"]) != 12:
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
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (args.output_dir / "replay-inventory.json").write_text(
        json.dumps({"files": inventory, "masters": masters}, indent=2)
    )
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
    visual = torch.from_numpy(x).to(device)
    vector = torch.from_numpy(v).to(device)
    admissible = torch.from_numpy(legal).to(device)
    records = {k: torch.as_tensor(a, device=device) for k, a in data.items()}
    train = torch.where(records["train"])[0]
    validation = torch.where(~records["train"])[0]
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
        current, nxt = records["state"][indices], records["next"][indices]
        actions = records["action"][indices].long()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.no_grad():
                labels = fitted_q_targets(
                    reward=records["reward"][indices].float(),
                    action=actions,
                    discount=records["discount"][indices].float(),
                    next_online=model(visual[nxt], vector[nxt]).float(),
                    next_target=target(visual[nxt], vector[nxt]).float(),
                    next_legal=admissible[nxt],
                    call_cost=cfg["call_cost"],
                    forecast_cost=cfg["forecast_cost"],
                )
            predicted = (
                model(visual[current], vector[current]).float().gather(1, actions[:, None])[:, 0]
            )
            loss = torch.nn.functional.smooth_l1_loss(predicted, labels)
        return loss

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
            }
        ),
        flush=True,
    )
    last = {}
    completed = False
    try:
        for step in range(1, cfg["updates"] + 1):
            model.train()
            indices = train[torch.randint(len(train), (cfg["batch_size"],), device=device)]
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
            if step == 1 or step % 50 == 0:
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
                }
                print(json.dumps(last), flush=True)
                with (args.output_dir / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(last) + "\n")
                run.log(last, step=step)
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
