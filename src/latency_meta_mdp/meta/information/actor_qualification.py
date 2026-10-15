"""Bounded real-replay one/two-device equivalence, recovery and learner throughput check."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from latency_meta_mdp.meta.information.actor_critic import initialize_from_encoder, returns_to_go
from latency_meta_mdp.meta.information.actor_learning import fit, initial_state, restore_models
from latency_meta_mdp.meta.information.attention_replay import prepare_attention_replay
from latency_meta_mdp.meta.information.config import load_config
from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.validation import atomic_json


def compare(left, right):
    maximum = 0.0
    for name in left["models"]:
        for key, value in left["models"][name].items():
            other = right["models"][name][key]
            torch.testing.assert_close(value, other, atol=3e-6, rtol=3e-4)
            maximum = max(maximum, float((value - other).abs().max()))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.set_num_threads(2)
    devices = [f"cuda:{i}" for i in range(len(args.gpus.split(",")))]
    if len(devices) != 2:
        raise ValueError("qualification requires two runtime GPUs")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    sources = json.loads(Path(args.sources).read_text())
    replay, receipt = prepare_attention_replay(
        config,
        sources,
        root=".",
        cache_dir=args.cache_dir,
        episode_cache_size=16,
        workers=2,
        cache_max_bytes=32 * 1024**3,
        progress=lambda e: print(json.dumps(e), flush=True),
    )
    norm = json.loads(Path(config["assets"]["paths"]["normalization"]).read_text())
    state = initial_state(config, normalization=norm, seed=17)
    state["binding"] = receipt
    options = dict(
        global_batch=8,
        microbatch=args.microbatch,
        updates=2,
        checkpoint_every=1,
        learning_rate=1e-4,
        tau=0.7,
        ema=0.005,
        seed=19,
        cpu_threads=2,
    )
    results, times = [], []
    for label, gpu_list in [("single", devices[:1]), ("multi", devices)]:
        started = perf_counter()
        results.append(fit("iql", state, replay, out / label, options=options, devices=gpu_list))
        times.append(perf_counter() - started)
    error = compare(*results)
    actor = initialize_from_encoder(restore_models(results[0])["v"])
    critic = initialize_from_encoder(actor, value=True)
    before = model_hash(actor)
    state = dict(
        config=config,
        phase="new_critic",
        step=0,
        epoch=0,
        optimizers={},
        models=dict(actor=actor.state_dict(), v=critic.state_dict()),
    )
    # Full episodes remain intact, but this check fits only one optimizer minibatch.
    count = min(8, len(replay))
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay

    rows = [replay[i] for i in range(count)]
    sample = DecisionReplay([rows])
    returns = []
    for i in range(len(replay.episodes)):
        episode = replay.episodes[i]
        returns.extend(
            returns_to_go([r["task_reward"] for r in episode], [r["terminated"] for r in episode])
        )
    calibrated = fit(
        "critic",
        state,
        sample,
        out / "calibration",
        options=dict(
            global_batch=8, microbatch=args.microbatch, epochs=1, learning_rate=1e-4, cpu_threads=2
        ),
        devices=devices,
        auxiliary=dict(returns=np.array(returns[:count], np.float32)),
    )
    assert model_hash(restore_models(calibrated)["actor"]) == before
    report = dict(
        status="pass",
        one_two_rank_max_abs=error,
        wall_seconds=times,
        real_episodes=len(replay.episodes),
        decisions=len(replay),
        model_width=config["model"]["width"],
        global_batch=8,
        actor_unchanged_during_calibration=True,
        devices=devices,
    )
    atomic_json(out / "complete.json", report)
    print("ACTOR_QUALIFICATION " + json.dumps(report), flush=True)
