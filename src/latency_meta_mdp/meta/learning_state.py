"""Recoverable Meta optimization state, separate from published inference weights."""

import math
import os
import random

import numpy as np
import torch


def learning_rate_at(step, config):
    initial = config["learning_rate"]
    if "lr_decay_end" not in config:
        return initial
    start, end = config["lr_decay_start"], config["lr_decay_end"]
    minimum = config["minimum_learning_rate"]
    if not 0 <= start < end or not 0 < minimum <= initial:
        raise ValueError("invalid Meta learning-rate schedule")
    if step <= start:
        return initial
    if step >= end:
        return minimum
    fraction = (step - start) / (end - start)
    return minimum + (initial - minimum) * (1 + math.cos(math.pi * fraction)) / 2


def save_learning_state(path, model, target, optimizer, step, config):
    numpy_state = np.random.get_state()
    state = {
        "schema": 1,
        "model": model.state_dict(),
        "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_learning_state(path, model, target, optimizer, config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema"] != 1:
        raise ValueError("unsupported learning state")
    # Dataset growth and a later end step are intentional; the objective/dynamics/schedule
    # are not silently changed. Stored normalization is restored with the model weights.
    mutable = {"updates", "replay_manifest", "code_commit", "resume_from", "visual_cache"}
    for key in (set(config) | set(state["config"])) - mutable:
        if config.get(key) != state["config"].get(key):
            raise ValueError(f"Meta resume configuration differs: {key}")
    model.load_state_dict(state["model"], strict=True)
    target.load_state_dict(state["target"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    name, keys, position, has_gauss, cached_gauss = state["numpy_rng"]
    np.random.set_state((name, keys.numpy(), position, has_gauss, cached_gauss))
    return state["step"]
