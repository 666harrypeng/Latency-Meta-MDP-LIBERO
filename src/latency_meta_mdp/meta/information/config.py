"""Strict experiment identity and scene partition checks without model imports."""

import hashlib
import json
from pathlib import Path

import yaml


def validate_binding(expected, actual):
    if expected != actual:
        raise ValueError("information scheduler binding differs")


def validate_splits(splits, *, excluded):
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("all three explicit scene partitions are required")
    seen = set()
    for name, seeds in splits.items():
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f"invalid seeds in {name}")
        if len(seeds) != len(set(seeds)) or seen.intersection(seeds):
            raise ValueError("scene partitions overlap or duplicate")
        if set(seeds).intersection(excluded):
            raise ValueError("scene was previously exposed")
        seen.update(seeds)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    keys = {
        "schema_version",
        "task_id",
        "mode",
        "assets",
        "task_config",
        "rtc_config",
        "splits",
        "excluded_scenes",
        "training",
        "output_dir",
    }
    version = config.get("schema_version") if isinstance(config, dict) else None
    if version == 2:
        keys |= {"model", "memory", "actions"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("information config has missing or unknown fields")
    if version not in (1, 2) or config["task_id"] != "conveyor_sort":
        raise ValueError("unsupported information experiment")
    if config["mode"] not in ("bound", "split"):
        raise ValueError("invalid information mode")
    training = config["training"]
    training_keys = {
        "algorithm",
        "gamma",
        "budget_lambda",
        "learning_rate",
        "fitted_iterations",
        "epochs_per_iteration",
        "conservative_weight",
    }
    if version == 2:
        training_keys |= {"optimizer_updates", "global_batch", "grad_clip", "optimizer_policy"}
        validate_attention_config(config)
    if set(training) != training_keys:
        raise ValueError("training has missing or unknown fields")
    if training["algorithm"] != "fqi" or training["gamma"] != 1 or training["budget_lambda"] != 0:
        raise ValueError(
            "training currently supports only the approved undiscounted success objective"
        )
    if training["conservative_weight"] < 0:
        raise ValueError("invalid training conservative weight")
    if training["learning_rate"] <= 0 or any(
        type(training[k]) is not int or training[k] < 1
        for k in ("fitted_iterations", "epochs_per_iteration")
    ):
        raise ValueError("invalid training budget")
    assets = config["assets"]
    if set(assets) != {"identity", "paths", "files"} or not assets["files"]:
        raise ValueError("invalid asset binding")
    if assets["identity"]["input_mode"] != "current_and_forecast":
        raise ValueError("unexpected conditioned input binding")
    paths, identity, files = assets["paths"], assets["identity"], assets["files"]
    if set(paths) != {"conditioned", "clean", "predictor_weights", "normalization", "decoder_dir"}:
        raise ValueError("asset path binding differs")
    checkpoint = Path(paths["conditioned"])
    if (
        checkpoint.name != str(identity["conditioned"]["step"])
        or checkpoint.parent.name != identity["conditioned"]["revision"]
    ):
        raise ValueError("checkpoint path/revision/step binding differs")
    required = [
        str(checkpoint / p)
        for p in (
            "_CHECKPOINT_METADATA",
            "params/_METADATA",
            "assets/metamdp/conveyor_sort_surface/norm_stats.json",
        )
    ]
    clean = Path(paths["clean"])
    if clean.name != str(identity["clean_baseline"]["step"]):
        raise ValueError("clean bootstrap step/path binding differs")
    required += [
        str(clean / p)
        for p in (
            "_CHECKPOINT_METADATA",
            "params/_METADATA",
            "assets/metamdp/conveyor_sort_surface/norm_stats.json",
        )
    ]
    required += [config["task_config"], config["rtc_config"]]
    if not all(p in files for p in required):
        raise ValueError("actual restore path is outside the asset binding")
    forecast = identity["forecast_identity"]
    for path, expected in (
        (paths["predictor_weights"], forecast["predictor_sha256"]),
        (paths["normalization"], forecast["jepa_normalization_sha256"]),
        (str(Path(paths["decoder_dir"]) / "model.safetensors"), forecast["decoder_sha256"]),
        (
            str(checkpoint / "assets/metamdp/conveyor_sort_surface/norm_stats.json"),
            identity["normalization_sha256"],
        ),
    ):
        if files.get(path) != expected:
            raise ValueError("loaded forecast/normalization asset binding differs")
    validate_splits(config["splits"], excluded=config["excluded_scenes"])
    return config


def validate_attention_config(config):
    from latency_meta_mdp.meta.information.memory_schema import ActionSpec

    if config["mode"] != "bound":
        raise ValueError("attention v2 requires bound Observe/Predict")
    actions = config["actions"]
    if set(actions) != {"version", "sleep_ticks", "starvation_mode", "reserve"}:
        raise ValueError("invalid action fields")
    ActionSpec(**actions)
    model = config["model"]
    if set(model) != {
        "architecture_id",
        "width",
        "num_queries",
        "num_heads",
        "ffn_width",
        "dropout",
    }:
        raise ValueError("invalid model fields")
    if (
        model["architecture_id"] != "attention_memory_v2"
        or model["dropout"] != 0
        or any(
            type(model[k]) is not int or model[k] < 1
            for k in ("width", "num_queries", "num_heads", "ffn_width")
        )
        or model["width"] % model["num_heads"]
        or (model["width"] // model["num_heads"]) % 2
    ):
        raise ValueError("invalid attention model dimensions")
    memory = config["memory"]
    if set(memory) != {
        "visual_frames",
        "recent_frames",
        "execution_records",
        "scheduler_events",
        "spatial_tokens",
        "lag_seconds",
        "tick_seconds",
        "time_scale",
        "episode_ticks",
    }:
        raise ValueError("invalid memory fields")
    if (
        any(
            type(memory[k]) is not int or memory[k] < 1
            for k in (
                "visual_frames",
                "recent_frames",
                "execution_records",
                "scheduler_events",
                "spatial_tokens",
                "episode_ticks",
            )
        )
        or memory["recent_frames"] != 3
        or memory["visual_frames"] < 3 + len(memory["lag_seconds"])
        or any(
            type(x) not in (int, float) or not 0 < x < float("inf")
            for x in [memory["tick_seconds"], memory["time_scale"], *memory["lag_seconds"]]
        )
    ):
        raise ValueError("invalid memory capacities/time scales")
    training = config["training"]
    if (
        any(
            type(training[k]) is not int or training[k] < 1
            for k in ("optimizer_updates", "global_batch")
        )
        or not 0 < training["grad_clip"] < float("inf")
        or training["optimizer_policy"] != "persistent"
    ):
        raise ValueError("invalid attention training budget/optimizer")
